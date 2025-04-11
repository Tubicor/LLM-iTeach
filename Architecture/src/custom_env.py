import os
import numpy as np
from collections import deque
from pyrep.const import RenderMode
from pyrep.errors import IKError, ConfigurationPathError
from rlbench.environment import Environment
from rlbench.task_environment import InvalidActionError, InvalidActionError
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig, CameraConfig
from dataclasses import dataclass, field
from utils import euler_to_quaternion
from typing import List
import tasks

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.environ["COPPELIASIM_ROOT"]


@dataclass
class EnvObject:
    name: str
    position: List[float] = np.array([0, 0, 0])
    quaternion: List[float] = np.array([0, 0, 0, 1])
    normal: List[float] = np.array([0, 0, 1])
    initial_position: List[float] = np.array([0, 0, 0])
    gripped: bool = False
    can_be_gripped: bool = False

    def set_random_values(self):
        self.position = np.random.uniform(-0.2, 0.2, 3)
        self.quaternion = np.random.uniform(-1, 1, 4)
        self.quaternion /= np.linalg.norm(self.quaternion)
        self.gripped = np.random.choice([True, False])
        self.can_be_gripped = np.random.choice([True, False])


@dataclass
class Gripper(EnvObject):
    closed: bool = False
    joint_positions: List[float] = np.array([0, 0, 0, 0, 0, 0, 0])

    def set_random_values(self):
        super().set_random_values()
        self.closed = np.random.choice([True, False])
        self.joint_positions = np.random.uniform(-np.pi, np.pi, 7)


@dataclass
class EnvironmentState:
    task: tasks.Task
    objects: dict = field(default_factory=dict)  # dictionary of string and EnvObject

    # set the objects in the EnvironmentState to task specific objects if not already set
    def __post_init__(self):
        for _object in self.task.objects:
            if _object not in self.objects:
                self.objects[_object] = EnvObject(name=_object)
            if "gripper" not in self.objects:
                self.objects["gripper"] = Gripper(name="gripper")

    def random_state(self):
        for key, _object in self.objects.items():
            _object.set_random_values()


class NoSceneDescriptorException(Exception):
    pass


class LLM_iTeachCustomEnv:
    def __init__(self, config):
        # image_size=(128, 128)
        obs_config = ObservationConfig(
            left_shoulder_camera=CameraConfig(rgb=False, depth=False, mask=False),
            right_shoulder_camera=CameraConfig(rgb=False, depth=False, mask=False),
            front_camera=CameraConfig(rgb=False, depth=False, mask=False),
            wrist_camera=CameraConfig(
                rgb=True, depth=False, mask=False, render_mode=RenderMode.OPENGL
            ),
            joint_positions=True,
            joint_velocities=True,
            joint_forces=False,
            gripper_pose=False,
            task_low_dim_state=False,
        )
        action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_WORLD_FRAME)
        self.env = Environment(
            action_mode,
            obs_config=obs_config,
            static_positions=config["static_env"],
            headless=config["headless_env"],
        )
        self.env.launch()
        highlevel_task_container: tasks.Task = getattr(tasks, config["task"])
        self.task = self.env.get_task(highlevel_task_container.rlb_class)
        self.gripper_open = 0.9
        self.gripper_deque = deque([0.9] * 20, maxlen=20)
        return

    def reset(self):
        self.gripper_open = 0.9
        self.gripper_deque = deque([0.9] * 20, maxlen=20)
        descriptions, obs = self.task.reset()
        camera_obs, proprio_obs = obs_split(obs)
        return camera_obs, proprio_obs

    def step(self, action):
        action_delayed = self.postprocess_action(action)
        could_perform_action = True
        try:
            next_obs, reward, done = self.task.step(action_delayed)
        except (
            IKError,
            InvalidActionError,
            ConfigurationPathError,
            InvalidActionError,
        ):
            could_perform_action = False
            zero_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, action_delayed[-1]]
            next_obs, reward, done = self.task.step(zero_action)
        camera_obs, proprio_obs = obs_split(next_obs)
        return camera_obs, proprio_obs, reward, done, could_perform_action

    def render(self):
        return

    def close(self):
        self.env.shutdown()
        return

    def postprocess_action(self, action):
        delta_position = action[:3] * 0.01
        delta_angle_quat = euler_to_quaternion(action[3:6] * 0.04)
        gripper_delayed = self.delay_gripper(action[-1])
        action_post = np.concatenate(
            (delta_position, delta_angle_quat, [gripper_delayed])
        )
        return action_post

    def delay_gripper(self, gripper_action):
        if gripper_action >= 0.0:
            gripper_action = 0.9
        elif gripper_action < 0.0:
            gripper_action = -0.9
        self.gripper_deque.append(gripper_action)
        if all([x == 0.9 for x in self.gripper_deque]):
            self.gripper_open = 1
        elif all([x == -0.9 for x in self.gripper_deque]):
            self.gripper_open = 0
        return self.gripper_open

    def get_state(self) -> EnvironmentState:
        robot_joints = self.env._robot.arm.get_joint_positions()
        robot_arm_tip_pose = self.env._robot.arm.get_tip().get_pose()
        robot_gripper_closed = (
            self.env._robot.gripper.get_open_amount()[0] < 0.9
        )  # When grabbing Lid closed is ~ 0.75 instead of 0 for closed
        robot_arm = Gripper(
            position=robot_arm_tip_pose[:3],
            quaternion=robot_arm_tip_pose[3:],
            name="robot_arm",
            gripped=robot_gripper_closed,
            closed=robot_gripper_closed,
            joint_positions=robot_joints,
            can_be_gripped=False,
        )

        if self.task.get_name() == "take_lid_off_saucepan":
            scene_objects = self.task._scene._active_task._initial_objs_in_scene
            for object, object_type in scene_objects:
                if object.get_name() == "saucepan_lid_grasp_point":
                    lid_handle_pose = object.get_pose()
                    position = lid_handle_pose[:3]
                    position[2] = position[2]
                    quaternion = [
                        0,
                        0,
                        0,
                        1,
                    ]  # need to reset with init pose since lid_handle_pose[3:] also has rotation of init object
                    can_be_gripped = (
                        self.env._robot.gripper._proximity_sensor.is_detected(object)
                    )
                    is_gripped = object in self.env._robot.gripper.get_grasped_objects()
                    lid = EnvObject(
                        position=position,
                        quaternion=quaternion,
                        name="lid",
                        gripped=is_gripped,
                        can_be_gripped=can_be_gripped,
                    )
            return EnvironmentState(
                task=tasks.TakeLidOffSaucepan,
                objects={"gripper": robot_arm, "lid": lid},
            )
        if self.task.get_name() == "push_button":
            scene_objects = self.task._scene._active_task._initial_objs_in_scene
            for object, object_type in scene_objects:
                if object.get_name() == "push_button_target":
                    button_pose = object.get_pose()
                    position = button_pose[:3]
                    position[2] = position[2] + 0.02
                    quaternion = [0, 0, 0, 1]
                    can_be_gripped = (
                        self.env._robot.gripper._proximity_sensor.is_detected(object)
                    )
                    is_gripped = object in self.env._robot.gripper.get_grasped_objects()
            button = EnvObject(
                position=position,
                quaternion=quaternion,
                name="button",
                gripped=is_gripped,
                can_be_gripped=can_be_gripped,
            )
            return EnvironmentState(
                task=tasks.PushButton, objects={"gripper": robot_arm, "button": button}
            )
        if self.task.get_name() == "close_microwave":
            scene_objects = self.task._scene._active_task._initial_objs_in_scene
            for object, object_type in scene_objects:
                if object.get_name() == "waypoint0":
                    microwave_door_pose = object.get_pose()
                    microwave_door_position = microwave_door_pose[:3]
                    microwave_door_quaternion = microwave_door_pose[3:]
                    microwave_door_can_be_gripped = (
                        self.env._robot.gripper._proximity_sensor.is_detected(object)
                    )
                    microwave_door_is_gripped = (
                        object in self.env._robot.gripper.get_grasped_objects()
                    )
                if object.get_name() == "microwave_frame_resp":
                    microwave = object.get_pose()
                    microwave_position = microwave[:3]
                    microwave_quaternion = microwave[3:]
                    microwave_can_be_gripped = (
                        self.env._robot.gripper._proximity_sensor.is_detected(object)
                    )
                    microwave_is_gripped = (
                        object in self.env._robot.gripper.get_grasped_objects()
                    )
            microwave_normal = (
                microwave_door_position - microwave_position
            ) / np.linalg.norm(microwave_door_position - microwave_position)
            microwave_normal[2] = 0
            microwave_door = EnvObject(
                position=microwave_door_position,
                quaternion=microwave_door_quaternion,
                name="MicrowaveDoor",
                gripped=microwave_door_is_gripped,
                can_be_gripped=microwave_door_can_be_gripped,
                normal=microwave_normal,
            )
            microwave = EnvObject(
                position=microwave_position,
                quaternion=microwave_quaternion,
                name="Microwave",
                gripped=microwave_is_gripped,
                can_be_gripped=microwave_can_be_gripped,
                normal=microwave_normal,
            )
            return EnvironmentState(
                task=tasks.CloseMicrowave,
                objects={
                    "gripper": robot_arm,
                    "microwave door": microwave_door,
                    "microwave": microwave,
                },
            )
        if self.task.get_name() == "unplug_charger":
            scene_objects = self.task._scene._active_task._initial_objs_in_scene
            for object, object_type in scene_objects:
                if object.get_name() == "charger":
                    charger_pose = object.get_pose()
                    charger_position = charger_pose[:3]
                    charger_quaternion = charger_pose[3:]
                    charger_can_be_gripped = (
                        self.env._robot.gripper._proximity_sensor.is_detected(object)
                    )
                    charger_is_gripped = (
                        object in self.env._robot.gripper.get_grasped_objects()
                    )
                if object.get_name() == "waypoint1":
                    charger_target_position = object.get_pose()[:3]
                if object.get_name() == "plug":
                    walloutlet_pose = object.get_pose()
                    walloutlet_position = walloutlet_pose[:3]
                    walloutlet_quaternion = walloutlet_pose[3:]
                    walloutlet_can_be_gripped = (
                        self.env._robot.gripper._proximity_sensor.is_detected(object)
                    )
                    walloutlet_is_gripped = (
                        object in self.env._robot.gripper.get_grasped_objects()
                    )

            charger_normal = (
                charger_target_position - charger_position
            ) / np.linalg.norm(charger_target_position - charger_position)
            charger = EnvObject(
                position=charger_position,
                quaternion=charger_quaternion,
                name="charger",
                normal=charger_normal,
                gripped=charger_is_gripped,
                can_be_gripped=charger_can_be_gripped,
            )
            walloutlet = EnvObject(
                position=walloutlet_position,
                quaternion=walloutlet_quaternion,
                name="walloutlet",
                gripped=walloutlet_is_gripped,
                can_be_gripped=walloutlet_can_be_gripped,
                normal=charger_normal,
            )
            return EnvironmentState(
                task=tasks.UnplugCharger,
                objects={
                    "gripper": robot_arm,
                    "charger": charger,
                    "wall outlet": walloutlet,
                },
            )
        if self.task.get_name() == "put_rubbish_in_bin":
            scene_objects = self.task._scene._active_task._initial_objs_in_scene
            for object, object_type in scene_objects:
                if object.get_name() == "waypoint3":
                    bin_pose = object.get_pose()
                    position = bin_pose[:3]
                    quaternion = [0, 0, 0, 1]
                    can_be_gripped = (
                        self.env._robot.gripper._proximity_sensor.is_detected(object)
                    )
                    is_gripped = object in self.env._robot.gripper.get_grasped_objects()
                    bin = EnvObject(
                        position=position,
                        quaternion=quaternion,
                        name="bin",
                        gripped=is_gripped,
                        can_be_gripped=can_be_gripped,
                    )
                if object.get_name() == "rubbish":
                    rubbish_pose = object.get_pose()
                    position = rubbish_pose[:3]
                    quaternion = [0, 0, 0, 1]
                    can_be_gripped = (
                        self.env._robot.gripper._proximity_sensor.is_detected(object)
                    )
                    is_gripped = object in self.env._robot.gripper.get_grasped_objects()
                    rubbish = EnvObject(
                        position=position,
                        quaternion=quaternion,
                        name="rubbish",
                        gripped=is_gripped,
                        can_be_gripped=can_be_gripped,
                    )
                if object.get_name() == "tomato1":
                    tomato1_pose = object.get_pose()
                    position = tomato1_pose[:3]
                    quaternion = [0, 0, 0, 1]
                    can_be_gripped = (
                        self.env._robot.gripper._proximity_sensor.is_detected(object)
                    )
                    is_gripped = object in self.env._robot.gripper.get_grasped_objects()
                    tomato1 = EnvObject(
                        position=position,
                        quaternion=quaternion,
                        name="tomato1",
                        gripped=is_gripped,
                        can_be_gripped=can_be_gripped,
                    )
                if object.get_name() == "tomato2":
                    tomato2_pose = object.get_pose()
                    position = tomato2_pose[:3]
                    quaternion = [0, 0, 0, 1]
                    can_be_gripped = (
                        self.env._robot.gripper._proximity_sensor.is_detected(object)
                    )
                    is_gripped = object in self.env._robot.gripper.get_grasped_objects()
                    tomato2 = EnvObject(
                        position=position,
                        quaternion=quaternion,
                        name="tomato2",
                        gripped=is_gripped,
                        can_be_gripped=can_be_gripped,
                    )
            return EnvironmentState(
                task=tasks.PutRubbishInBin,
                objects={
                    "gripper": robot_arm,
                    "rubbish": rubbish,
                    "bin": bin,
                    "tomato1": tomato1,
                    "tomato2": tomato2,
                },
            )
        if self.task.get_name() == "pick_and_lift":
            scene_objects = self.task._scene._active_task._initial_objs_in_scene
            for object, object_type in scene_objects:
                if object.get_name() == "pick_and_lift_target":
                    cube_pose = object.get_pose()
                    cube_position = cube_pose[:3]
                    cube_quaternion = [0, 0, 0, 1]
                    cube_can_be_gripped = (
                        self.env._robot.gripper._proximity_sensor.is_detected(object)
                    )
                    cube_is_gripped = (
                        object in self.env._robot.gripper.get_grasped_objects()
                    )
                    cube = EnvObject(
                        position=cube_position,
                        quaternion=cube_quaternion,
                        name="cube",
                        gripped=cube_is_gripped,
                        can_be_gripped=cube_can_be_gripped,
                    )
                if object.get_name() == "waypoint3":
                    target_pose = object.get_pose()
                    target_position = target_pose[:3]
                    target_quaternion = [0, 0, 0, 1]
                    target_can_be_gripped = (
                        self.env._robot.gripper._proximity_sensor.is_detected(object)
                    )
                    target_is_gripped = (
                        object in self.env._robot.gripper.get_grasped_objects()
                    )
                    target = EnvObject(
                        position=target_position,
                        quaternion=target_quaternion,
                        name="target area",
                        gripped=target_is_gripped,
                        can_be_gripped=target_can_be_gripped,
                    )
            return EnvironmentState(
                task=tasks.PickAndLift,
                objects={"gripper": robot_arm, "cube": cube, "target area": target},
            )
        if self.task.get_name() == "reach_target":
            scene_objects = self.task._scene._active_task._initial_objs_in_scene
            for object, object_type in scene_objects:
                if object.get_name() == "target":
                    target_pose = object.get_pose()
                    target_position = target_pose[:3]
                    target_quaternion = [0, 0, 0, 1]
                    target_can_be_gripped = (
                        self.env._robot.gripper._proximity_sensor.is_detected(object)
                    )
                    target_is_gripped = (
                        object in self.env._robot.gripper.get_grasped_objects()
                    )
                    target = EnvObject(
                        position=target_position,
                        quaternion=target_quaternion,
                        name="target",
                        gripped=target_is_gripped,
                        can_be_gripped=target_can_be_gripped,
                    )
            return EnvironmentState(
                task=tasks.ReachTarget, objects={"gripper": robot_arm, "target": target}
            )
        if self.task.get_name() == "stack_blocks":
            scene_objects = self.task._scene._active_task._initial_objs_in_scene
            for object, object_type in scene_objects:
                if object.get_name() == "stack_blocks_target1":
                    block1_pose = object.get_pose()
                    block1_position = block1_pose[:3]
                    block1_quaternion = [0, 0, 0, 1]
                    block1_can_be_gripped = (
                        self.env._robot.gripper._proximity_sensor.is_detected(object)
                    )
                    block1_is_gripped = (
                        object in self.env._robot.gripper.get_grasped_objects()
                    )
                    block1 = EnvObject(
                        position=block1_position,
                        quaternion=block1_quaternion,
                        name="block1",
                        gripped=block1_is_gripped,
                        can_be_gripped=block1_can_be_gripped,
                    )
                if object.get_name() == "stack_blocks_target2":
                    block2_pose = object.get_pose()
                    block2_position = block2_pose[:3]
                    block2_quaternion = [0, 0, 0, 1]
                    block2_can_be_gripped = (
                        self.env._robot.gripper._proximity_sensor.is_detected(object)
                    )
                    block2_is_gripped = (
                        object in self.env._robot.gripper.get_grasped_objects()
                    )
                    block2 = EnvObject(
                        position=block2_position,
                        quaternion=block2_quaternion,
                        name="block2",
                        gripped=block2_is_gripped,
                        can_be_gripped=block2_can_be_gripped,
                    )
                if object.get_name() == "stack_blocks_target3":
                    block3_pose = object.get_pose()
                    block3_position = block3_pose[:3]
                    block3_quaternion = [0, 0, 0, 1]
                    block3_can_be_gripped = (
                        self.env._robot.gripper._proximity_sensor.is_detected(object)
                    )
                    block3_is_gripped = (
                        object in self.env._robot.gripper.get_grasped_objects()
                    )
                    block3 = EnvObject(
                        position=block3_position,
                        quaternion=block3_quaternion,
                        name="block3",
                        gripped=block3_is_gripped,
                        can_be_gripped=block3_can_be_gripped,
                    )
                if object.get_name() == "stack_blocks_target0":
                    block4_pose = object.get_pose()
                    block4_position = block4_pose[:3]
                    block4_quaternion = [0, 0, 0, 1]
                    block4_can_be_gripped = (
                        self.env._robot.gripper._proximity_sensor.is_detected(object)
                    )
                    block4_is_gripped = (
                        object in self.env._robot.gripper.get_grasped_objects()
                    )
                    block4 = EnvObject(
                        position=block4_position,
                        quaternion=block4_quaternion,
                        name="block4",
                        gripped=block4_is_gripped,
                        can_be_gripped=block4_can_be_gripped,
                    )
                if object.get_name() == "stack_blocks_target_plane":
                    target_pose = object.get_pose()
                    target_position = target_pose[:3]
                    target_quaternion = [0, 0, 0, 1]
                    target_can_be_gripped = (
                        self.env._robot.gripper._proximity_sensor.is_detected(object)
                    )
                    target_is_gripped = (
                        object in self.env._robot.gripper.get_grasped_objects()
                    )
                    target = EnvObject(
                        position=target_position,
                        quaternion=target_quaternion,
                        name="target",
                        gripped=target_is_gripped,
                        can_be_gripped=target_can_be_gripped,
                    )
            return EnvironmentState(
                task=tasks.StackBlocks,
                objects={
                    "gripper": robot_arm,
                    "block4": block4,
                    "block1": block1,
                    "block2": block2,
                    "block3": block3,
                    "target": target,
                },
            )

        raise NoSceneDescriptorException(
            f"No descriptor implemented for task: {self.task.get_name()}"
        )


def obs_split(observation):
    camera_obs = observation.wrist_rgb.transpose((2, 0, 1))
    # For using the two camera setup this code was added (!adding the additional information is not covered in this repository!)
    # camera_front_obs = observation.front_rgb.transpose((2,0,1))
    # camera_obs = np.concatenate((camera_front_obs, camera_wrist_obs), axis=2)
    proprio_obs = np.append(observation.joint_positions, observation.gripper_open)
    return camera_obs, proprio_obs
