import os
import time
import torch
import numpy as np
from argparse import ArgumentParser
from utils import (
    TrajectoriesDataset,
    log_to_csv,
    path,
    quaternion_to_euler,
    Timer,
    print_progressbar,
)
from custom_env import LLM_iTeachCustomEnv, EnvironmentState

# from functools import partial
# from pynput import keyboard
from feedback import CodePolicyAPI


def human_feedback(keyboard_obs, action, feedback_type):
    if feedback_type == "evaluative":
        feedback = keyboard_obs.get_label()

    elif feedback_type == "dagger":
        if keyboard_obs.has_joints_cor() or keyboard_obs.has_gripper_update():
            action = correct_action(keyboard_obs, action)
            feedback = -1  # corrected
        else:
            feedback = 0  # bad

    elif feedback_type == "iwr":
        if keyboard_obs.has_joints_cor() or keyboard_obs.has_gripper_update():
            action = correct_action(keyboard_obs, action)
            feedback = -1  # corrected
        else:
            feedback = 1  # good

    elif feedback_type == "LLM_iTeach_full":
        if keyboard_obs.has_joints_cor() or keyboard_obs.has_gripper_update():
            action = correct_action(keyboard_obs, action)
            feedback = -1  # corrected
        else:
            feedback = keyboard_obs.get_label()

    elif feedback_type == "LLM_iTeach_partial":
        if keyboard_obs.has_joints_cor() or keyboard_obs.has_gripper_update():
            action = correct_action(keyboard_obs, action, full_control=False)
            feedback = -1  # corrected
        else:
            feedback = keyboard_obs.get_label()

    else:
        raise NotImplementedError("Feedback type not supported!")
    return action, feedback


def correct_action(keyboard_obs, action, full_control=True):
    if full_control:
        action[:-1] = keyboard_obs.get_ee_action()
    elif keyboard_obs.has_joints_cor():
        ee_step = keyboard_obs.get_ee_action()
        action[:-1] = action[:-1] * 0.5 + ee_step
        action = np.clip(action, -0.9, 0.9)
    if keyboard_obs.has_gripper_update():
        action[-1] = keyboard_obs.get_gripper()
    return action


# TODO human
# class KeyboardObserver:
#     def __init__(self):
#         self.reset()
#         self.hotkeys = keyboard.GlobalHotKeys(
#             {
#                 "c": partial(self.set_gripper, -0.9),  # close
#                 "v": partial(self.set_gripper, 0.9),  # open
#                 "x": self.set_reset,
#                 "ö": self.set_stop,
#             }
#         )
#         self.hotkeys.start()
#         self.direction = np.array([0, 0, 0, 0, 0, 0])
#         self.listener = keyboard.Listener(
#             on_press=self.set_direction, on_release=self.reset_direction
#         )
#         self.key_mapping = {
#             "a": (1, 1),  # left
#             "d": (1, -1),  # right
#             "s": (0, -1),  # backward
#             "w": (0, 1),  # forward
#             "q": (2, 1),  # down
#             "e": (2, -1),  # up
#             "j": (3, -1),  # look left
#             "l": (3, 1),  # look right
#             "i": (4, 1),  # look up
#             "k": (4, -1),  # look down
#             "u": (5, -1),  # rotate left
#             "o": (5, 1),  # rotate right
#         }
#         self.listener.start()
#         return
#     def set_stop(self):
#         self.stop_button = True
#         return
#     def set_reset(self):
#         self.reset_button = True
#         return

#     def set_gripper(self, value):
#         self.gripper_open = value
#         return

#     def get_gripper(self):
#         return self.gripper_open

#     def set_direction(self, key):
#         try:
#             idx, value = self.key_mapping[key.char]
#             self.direction[idx] = value
#         except (KeyError, AttributeError):
#             pass
#         return

#     def reset_direction(self, key):
#         try:
#             idx, _ = self.key_mapping[key.char]
#             self.direction[idx] = 0
#         except (KeyError, AttributeError):
#             pass
#         return

#     def has_joints_cor(self):
#         return self.direction.any()

#     def has_gripper_update(self):
#         return self.get_gripper() is not None

#     def get_ee_action(self):
#         return self.direction * 0.9

#     def reset(self):
#         self.set_gripper(0.9)
#         self.reset_button = False
#         self.stop_button = False
#         return


def LLM_iTeach_action_adapter(action, state: EnvironmentState):
    """Formats an action to the input of the LLM_iTeach algorithm (Controller)"""
    xyz = np.array(action[0:3])
    euler = np.array(quaternion_to_euler(action[3:-1]))
    gripper_open = action[-1]
    pos_th = 0.01  # positional threshold
    rot_th = 0.04  # rotational threshold
    conform_xyz = np.where(xyz > pos_th, 1, np.where(xyz < -pos_th, -1, 0)) * 0.9
    conform_euler = np.where(euler > rot_th, 1, np.where(euler < -rot_th, -1, 0)) * 0.9
    # Map to EE_POSE_EE_FRAME
    # conform_xyz[0] *= -1
    # conform_xyz[2] *= -1

    if gripper_open is None:
        gripper_open = not state.objects["gripper"].closed
    conform_gripper = 0.9 if gripper_open else -0.9
    conform_action = np.concatenate([conform_xyz, conform_euler, [conform_gripper]])
    return conform_action


# TODO delete simulate.py
def main(config):
    env = LLM_iTeachCustomEnv(config)
    # keyboard_obs = KeyboardObserver()
    replay_memory = TrajectoriesDataset(config["sequence_len"])
    demonstrations_count = 0
    total_demonstrations = 0
    total_timer = Timer()
    total_time_on_success = 0
    total_steps_on_success = 0
    total_err_steps = 0
    while demonstrations_count < config["demonstrations"]:
        if not config["verbose"]:
            print_progressbar(
                demonstrations_count,
                config["demonstrations"],
                "Demonstrating",
                f" of {config['demonstrations']} demonstrations",
            )
        episode_timer = Timer()
        total_demonstrations += 1
        gripper_open = 0.9
        camera_obs, proprio_obs = env.reset()
        step_count = 0
        err_steps = 0
        teacher = CodePolicyAPI(env.get_state().task)
        # keyboard_obs.reset()
        done = False
        while step_count < 3000 and not done and err_steps < 200:
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_open])
            state = env.get_state()
            if config["control"] == "human":
                # if keyboard_obs.has_joints_cor():
                #     action[:6] = keyboard_obs.get_ee_action()
                # if keyboard_obs.has_gripper_update():
                #     gripper_open = keyboard_obs.get_gripper()
                #     action[-1] = gripper_open
                pass
            elif config["control"] == "llm":
                corrective_action = teacher.feedback(state)
                action = LLM_iTeach_action_adapter(corrective_action, state)

            next_camera_obs, next_proprio_obs, reward, done, could_perform = env.step(
                action
            )
            if config["save_demos"]:
                replay_memory.add(camera_obs, proprio_obs, action, [1])
                camera_obs, proprio_obs = next_camera_obs, next_proprio_obs

            if not could_perform:
                if config["control"] == "llm":
                    err_steps += 1
                if config["control"] == "human":
                    print(
                        "-" * np.random.randint(1, 4),
                        "> Could not perform action",
                        sep="",
                    )

            # if config["control"] == "human":
            #     if keyboard_obs.stop_button:
            #         return
            #     if keyboard_obs.reset_button:
            #         break
            if config["control"] == "llm":
                step_count += 1
        if done:
            demonstrations_count += 1
            if config["save_demos"]:
                replay_memory.save_current_traj()
            total_steps_on_success += step_count
            total_time_on_success += episode_timer.elapsed()
            total_err_steps += err_steps

    if config["save_demos"]:
        save_path = (
            path + "demonstrations/" + config["control"] + "/" + config["task"] + "/"
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = "demos_" + str(config["demonstrations"]) + ".dat"
        torch.save(replay_memory, save_path + file_name)
    result_dict = {
        "Task": config["task"],
        "Demonstrations": config["demonstrations"],
        "FailedDemonstrations": total_demonstrations - config["demonstrations"],
        "ElapsedTime": total_timer.elapsed(),
        "AvgTimeOnSuccess": total_time_on_success / config["demonstrations"],
        "AvgStepsOnSuccess": total_steps_on_success / config["demonstrations"],
        "AvgErrStepsOnSuccess": total_err_steps / config["demonstrations"],
        "Control": config["control"],
    }
    log_to_csv(result_dict, path, "demonstrate.csv")
    print(
        "\r",
        " " * 100,
        "\nFinished demonstration of task: ",
        config["task"],
        " using ",
        config["control"],
        " control",
    )
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        default="PushButton",
        help="options: CloseMicrowave, PushButton, TakeLidOffSaucepan, UnplugCharger, PutRubbishInBin",
    )
    parser.add_argument(
        "-d",
        "--demonstrations",
        dest="demonstrations",
        default=10,
        type=int,
        help="number of demonstrations to collect",
    )
    parser.add_argument(
        "-c",
        "--control",
        dest="control",
        default="llm",
        help="options: llm, human",
    )
    parser.add_argument(
        "-s",
        "--show",
        dest="show",
        action="store_true",
        help="Show GUI",
    )
    parser.add_argument(
        "-r",
        "--record",
        dest="record",
        action="store_true",
        help="Record demonstrations",
    )
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument(
        "--seqlen",
        dest="seqlen",
        default=150,
        type=int,
    )
    args = parser.parse_args()
    config = {
        "control": args.control,
        "task": args.task,
        "static_env": False,
        "headless_env": not args.show,
        "demonstrations": args.demonstrations,
        "sequence_len": args.seqlen,
        "save_demos": args.record,
        "verbose": args.verbose,
    }
    main(config)
