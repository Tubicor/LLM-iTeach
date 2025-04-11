import os
import time
import torch
import numpy as np
from argparse import ArgumentParser
from utils import TrajectoriesDataset, loop_sleep, path
from custom_env import LLM_iTeachCustomEnv
from pynput import keyboard
from functools import partial


class KeyboardObserver:
    def __init__(self):
        self.reset()
        self.hotkeys = keyboard.GlobalHotKeys(
            {
                "g": partial(self.set_label, 1),  # good
                "b": partial(self.set_label, 0),  # bad
                "c": partial(self.set_gripper, -0.9),  # close
                "v": partial(self.set_gripper, 0.9),  # open
                "f": partial(self.set_gripper, None),  # gripper free
                "x": self.reset_episode,
            }
        )
        self.hotkeys.start()
        self.direction = np.array([0, 0, 0, 0, 0, 0])
        self.listener = keyboard.Listener(
            on_press=self.set_direction, on_release=self.reset_direction
        )
        self.key_mapping = {
            "a": (1, 1),  # left
            "d": (1, -1),  # right
            "s": (0, 1),  # backward
            "w": (0, -1),  # forward
            "q": (2, 1),  # down
            "e": (2, -1),  # up
            "j": (3, -1),  # look left
            "l": (3, 1),  # look right
            "i": (4, -1),  # look up
            "k": (4, 1),  # look down
            "u": (5, -1),  # rotate left
            "o": (5, 1),  # rotate right
        }
        self.listener.start()
        return

    def set_label(self, value):
        self.label = value
        print("label set to: ", value)
        return

    def get_label(self):
        return self.label

    def set_gripper(self, value):
        self.gripper_open = value
        print("gripper set to: ", value)
        return

    def get_gripper(self):
        return self.gripper_open

    def set_direction(self, key):
        try:
            idx, value = self.key_mapping[key.char]
            self.direction[idx] = value
        except (KeyError, AttributeError):
            pass
        return

    def reset_direction(self, key):
        try:
            idx, _ = self.key_mapping[key.char]
            self.direction[idx] = 0
        except (KeyError, AttributeError):
            pass
        return

    def has_joints_cor(self):
        return self.direction.any()

    def has_gripper_update(self):
        return self.get_gripper() is not None

    def get_ee_action(self):
        return self.direction * 0.9

    def reset_episode(self):
        self.reset_button = True
        return

    def reset(self):
        self.set_label(1)
        self.set_gripper(None)
        self.reset_button = False
        return


def main(config):
    save_path = path + "/demonstrations/human/" + config["task"] + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    env = LLM_iTeachCustomEnv(config)
    keyboard_obs = KeyboardObserver()
    replay_memory = TrajectoriesDataset(config["sequence_len"])
    camera_obs, proprio_obs = env.reset()
    gripper_open = 0.9
    time.sleep(5)
    print("Go!")
    episodes_count = 0
    while episodes_count < config["episodes"]:
        start_time = time.time()
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_open])
        if keyboard_obs.has_joints_cor():
            action[:-1] = keyboard_obs.get_ee_action()
        if keyboard_obs.has_gripper_update():
            gripper_open = keyboard_obs.get_gripper()
            action[-1] = gripper_open
        next_camera_obs, next_proprio_obs, reward, done, _ = env.step(action)
        replay_memory.add(camera_obs, proprio_obs, action, [1])
        camera_obs, proprio_obs = next_camera_obs, next_proprio_obs
        if keyboard_obs.reset_button:
            replay_memory.reset_current_traj()
            camera_obs, proprio_obs = env.reset()
            gripper_open = 0.9
            keyboard_obs.reset()
        elif done:
            replay_memory.save_current_traj()
            camera_obs, proprio_obs = env.reset()
            gripper_open = 0.9
            episodes_count += 1
            print("Episode", episodes_count, "done")
            keyboard_obs.reset()
            done = False
        else:
            loop_sleep(start_time)

    file_name = "demos_" + str(config["episodes"]) + ".dat"
    if config["save_demos"]:
        torch.save(replay_memory, save_path + file_name)
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        default="PushButton",
        help="options: CloseMicrowave, PushButton, TakeLidOffSaucepan, UnplugCharger",
    )
    args = parser.parse_args()
    config = {
        "task": args.task,
        "static_env": False,
        "headless_env": False,
        "save_demos": True,
        "episodes": 10,
        "sequence_len": 150,
    }
    main(config)
