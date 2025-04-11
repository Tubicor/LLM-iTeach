import torch
import time
from models import Policy
from custom_env import LLM_iTeachCustomEnv
from utils import loop_sleep
from argparse import ArgumentParser
import numpy as np
from utils import log_to_csv, print_progressbar, path, Timer, device


def run_simulation(env, policy, config):
    successes = 0
    timer = Timer()
    total_steps = 0
    for episode in range(config["episodes"]):
        if not config["verbose"]:
            print_progressbar(
                episode,
                config["episodes"],
                "Evaluating policy",
                f" of {config['episodes']} episodes",
            )
        steps = 0
        done = False
        camera_obs, proprio_obs = env.reset()
        lstm_state = None
        err_steps = 0
        while not done and steps < 1000 and err_steps < 30:
            start_time = time.time()
            action, lstm_state = policy.predict(camera_obs, proprio_obs, lstm_state)
            next_camera_obs, next_proprio_obs, reward, done, could_perform_action = (
                env.step(action)
            )
            if not could_perform_action:
                err_steps += 1
            camera_obs, proprio_obs = next_camera_obs, next_proprio_obs
            steps += 1
            loop_sleep(start_time)
        if done:
            total_steps += steps
            successes += 1

    result_dict = {
        "Successrate": successes / config["episodes"],
        "Task": config["task"],
        "Demonstrations": config["control_policy"].split("_")[-1],
        "Evaluations": config["episodes"],
        "ElapsedTime": timer.elapsed(),
        "AvgStepsOnSuccess": np.nan if successes == 0 else total_steps / successes,
    }
    print(
        f"\rFinished evaluation of task: {config['task']} trained on {config['control_policy'].split('_')[1]} demonstrations",
        " " * 100,
    )
    log_to_csv(result_dict, path=config["metrics_path"], file_name="evaluation.csv")


def main(config):
    policy = Policy(config).to(device)
    control = config["control_policy"].split("_")[0]
    config["metrics_path"] = f"{path}/Metrics/{config['experiment_name']}"
    if control == "LLM_iTeach":
        file_path = f"{config['metrics_path']}/LearnedPolicies/{config['task']}/{config['control_policy']}_policy.pt"
    elif control == "bc":
        demos = config["control_policy"].split("_")[-1]
        control_type = config["control_policy"].split("_")[-2]  # llm or human
        file_path = f"{path}/LearnedPolicies/{control_type}/{config['task']}/bc_policy_{demos}demos_{config['bcsteps']}steps.pt"
    else:
        raise ValueError("Invalid control policy")

    policy.load_state_dict(torch.load(file_path))
    policy.eval()
    env = LLM_iTeachCustomEnv(config)
    run_simulation(env, policy, config)
    return


if __name__ == "__main__":
    # set_seeds(1996)
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--control_policy",
        dest="control_policy",
        default="""
        LLM_iTeach: celing_xxx (xxx is the number of episodes) --> LLM_iTeach_600
        BEHAVIOR CLONING: bc_zzz_qqq ( zzz is the number of demonstrations qqq number of steps) --> bc_policy_10_800steps.pt""",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        default="TakeLidOffSaucepan",
        help="options: CloseMicrowave, PushButton, TakeLidOffSaucepan, UnplugCharger, PutRubbishInBin",
    )
    parser.add_argument(
        "-s",
        "--show",
        dest="show",
        action="store_true",
        help="Show GUI",
    )
    parser.add_argument("-e", "--episodes", dest="episodes", type=int, default=100)
    parser.add_argument(
        "-x",
        "--xperiment_name",
        dest="experiment_name",
        default="standard",
    )
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--bcsteps", dest="bcsteps", type=int, default=2000)
    args = parser.parse_args()

    config_defaults = {
        "control_policy": args.control_policy,
        "task": args.task,
        "episodes": args.episodes,
        "static_env": False,
        "headless_env": not args.show,
        "proprio_dim": 8,
        "action_dim": 7,
        "visual_embedding_dim": 256,
        "learning_rate": 3e-4,
        "weight_decay": 3e-6,
        "batch_size": 16,
        "experiment_name": args.experiment_name,
        "verbose": args.verbose,
        "bcsteps": args.bcsteps,
    }
    main(config_defaults)
