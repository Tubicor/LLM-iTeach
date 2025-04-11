import numpy as np
from custom_env import EnvironmentState
import tasks

from Codepolicy import Codepolicy


def expert_feedback(state: EnvironmentState):
    """Returns an object of Feedback class"""
    if state.task == tasks.TakeLidOffSaucepan:
        lid_position = state.objects["lid"].position
        robot_arm_position = state.objects["gripper"].position
        gripper_closed = state.objects["gripper"].closed

        close_to_lid = state.objects["lid"].can_be_gripped
        # stage 1 approach lid
        if not close_to_lid and not gripper_closed:
            return np.concatenate(
                (lid_position - robot_arm_position, [0, 0, 0, 1, True])
            )
        # stage 2 grip lid
        elif close_to_lid and not gripper_closed:
            return np.array([0, 0, 0, 0, 0, 0, 1, False])
        # stage 3 lift lid
        else:
            return np.array([0, 0, 0.2, 0, 0, 0, 1, False])
    elif state.task == tasks.PushButton:
        button_position = state.objects["button"].position
        robot_arm_position = state.objects["gripper"].position
        gripper_closed = state.objects["gripper"].closed
        if not gripper_closed:
            return np.array([0, 0, 0, 0, 0, 0, 1, False])
        else:
            return np.concatenate(
                (button_position - robot_arm_position, [0, 0, 0, 1, None])
            )
    elif state.task == tasks.CloseMicrowave:
        microwave_door_position = state.objects["microwave door"].position
        microwave_position = state.objects["microwave"].position
        microwave_door_normal = state.objects["microwave door"].normal
        robot_arm_position = state.objects["gripper"].position
        is_xy_md = (
            np.linalg.norm(microwave_door_position[:2] - robot_arm_position[:2]) < 0.05
        )
        is_above_md = robot_arm_position[2] - microwave_door_position[2] > 0.05
        if is_above_md and not is_xy_md:
            action = microwave_door_position - robot_arm_position
            return np.concatenate((action[:2], [action[2] * 0.05, 0, 0, 0, 1, False]))
        if is_above_md:
            return np.concatenate(
                (microwave_door_position - robot_arm_position, [0, 0, 0, 1, True])
            )
        else:
            return np.concatenate((-microwave_door_normal, [0, 0, 0, 1, False]))
    elif state.task == tasks.UnplugCharger:
        charger_position = state.objects["charger"].position
        robot_arm_position = state.objects["gripper"].position
        can_grab_charger = state.objects["charger"].can_be_gripped
        is_grabbing_charger = state.objects["charger"].gripped
        charger_normal = state.objects["charger"].normal
        if can_grab_charger and not is_grabbing_charger:
            return np.array([0, 0, 0, 0, 0, 0, 1, False])
        elif is_grabbing_charger:
            return np.concatenate((charger_normal, [0, 0, 0, 1, False]))
        else:
            return np.concatenate(
                (charger_position - robot_arm_position, [0, 0, 0, 1, True])
            )
    elif state.task == tasks.PutRubbishInBin:
        bin_position = state.objects["Bbin"].position
        rubbish_position = state.objects["rubbish"].position
        robot_arm_position = state.objects["gripper"].position
        gripping_rubbish = state.objects["rubbish"].gripped
        rubbish_can_be_gripped = state.objects["rubbish"].can_be_gripped
        at_position_of_bin = np.linalg.norm(bin_position - robot_arm_position) < 0.05
        if at_position_of_bin:
            return np.array([0, 0, 0, 0, 0, 0, 1, True])
        if not gripping_rubbish and not rubbish_can_be_gripped:
            return np.concatenate(
                (rubbish_position - robot_arm_position, [0, 0, 0, 1, True])
            )
        elif not gripping_rubbish and rubbish_can_be_gripped:
            return np.array([0, 0, 0, 0, 0, 0, 1, False])
        elif gripping_rubbish and not at_position_of_bin:
            return np.concatenate(
                (bin_position - robot_arm_position, [0, 0, 0, 1, False])
            )
    elif state.task == tasks.PushButtons:
        button_0_position = state.objects["button0"].position
        robot_arm_position = state.objects["gripper"].position
        gripper_closed = state.objects["gripper"].closed
        if not gripper_closed:
            return np.array([0, 0, 0, 0, 0, 0, 1, False])
        else:
            return np.concatenate(
                (button_0_position - robot_arm_position, [0, 0, 0, 1, None])
            )
    elif state.task == tasks.ReachTarget:
        target_0_position = state.objects["target0"].position
        robot_arm_position = state.objects["gripper"].position
        return np.concatenate(
            (target_0_position - robot_arm_position, [0, 0, 0, 1, None])
        )

    return np.array([0, 0, 0, 0, 0, 0, 1, False])


class CodePolicyAPI:
    """This class is responsible for interfacing the feedback generation from CodePolicy with a given state"""

    def __init__(self, task: tasks.Task):
        self.codepolicy: Codepolicy = Codepolicy.load(
            task, filenames=["CodePolicy_1.py"]
        )[0]

    def feedback(self, state: EnvironmentState):
        corrective_feedback = np.array([0, 0, 0, 0, 0, 0, 1, None])
        action, done = self.codepolicy.execute(state.objects)
        corrective_feedback[:3] = action[:3]
        corrective_feedback[-1] = (
            False if action[-1] == -1 else (True if action[-1] == 1 else None)
        )
        return corrective_feedback
