import threading
import time
import torch
import numpy as np
from argparse import ArgumentParser
from custom_env import LLM_iTeachCustomEnv, EnvironmentState
from demonstrate import LLM_iTeach_action_adapter
from feedback import expert_feedback, CodePolicyAPI
from models import Policy
from utils import (
    TrajectoriesDataset,
    path,
    print_progressbar,
    loop_sleep,
    Timer,
    device,
    set_seeds,
    log_to_csv,
)
import os
import pandas as pd


def train_step(policy, replay_memory, config, stop_flag, thread_return):
    iterations = 0
    data = pd.DataFrame(columns=["Loss", "Demonstrations"])
    while not stop_flag.isSet() or iterations < (2000 + config["episodes"] * 20):
        if len(replay_memory) >= 1:
            batch = replay_memory.sample(config["batch_size"])
            camera_batch, proprio_batch, action_batch, feedback_batch = batch
            total_loss = policy.update_params(
                camera_batch, proprio_batch, action_batch, feedback_batch
            )
            iterations += 1

            data.loc[len(data)] = {
                "Loss": total_loss["loss"].item(),
                "Demonstrations": len(replay_memory),
            }
        else:
            time.sleep(1)
    # get filename to save training metrics
    max_i = 0
    tm_path = config["metrics_path"] + "/TrainingMetrics/"
    # If necessary create folder structure
    if not os.path.exists(tm_path):
        os.makedirs(tm_path)
    for filename in os.listdir(tm_path):
        if filename.startswith("TrainingMetrics_") and filename.endswith(".csv"):
            i = int(filename.split("_")[1].split(".")[0])
            if i > max_i:
                max_i = i
    filename = "TrainingMetrics_" + str(max_i + 1) + ".csv"
    file_path = tm_path + filename
    data.to_csv(file_path, index=True)
    thread_return["TrainingMetricsFile"] = filename
    thread_return["TrainingSteps"] = iterations


def run_env_simulation(env, policy, replay_memory, config):
    camera_obs, proprio_obs = env.reset()
    episode_counter = 0
    successes = 0
    total_steps = 0
    total_timer = Timer()
    total_elapsed_time_success = 0
    evaluation_rates = []
    lengths = []
    angles = []
    while successes < config["episodes"]:
        if not config["verbose"]:
            print_progressbar(
                successes, config["episodes"], "Episodes with LLM_iTeach:"
            )
        step_count = 0
        done = False
        lstm_state = None
        evaluations = 0
        length_sum = 0
        angle_sum = 0
        err_steps = 0
        episode_timer = Timer()
        teacher = CodePolicyAPI(env.get_state().task)
        while (
            not done and step_count < 3000 and err_steps < 30
        ):  # config["sequence_len"]:
            start_time = time.time()
            step_count += 1
            """ PREDICT ACTION """
            action, lstm_state = policy.predict(camera_obs, proprio_obs, lstm_state)
            """ GET GUIDANCE BY TEACHER """
            env_state: EnvironmentState = env.get_state()
            # corrected_action = expert_feedback(env_state)
            teacher_action = teacher.feedback(env_state)
            corrected_action = LLM_iTeach_action_adapter(teacher_action, env_state)
            label = 0
            if config["evaluative"]:
                """ EVALUATIVE FEEDBACK """
                dot_product = np.dot(action[:3], corrected_action[:3])
                norm_a = np.linalg.norm(action[:3])
                norm_ca = np.linalg.norm(corrected_action[:3])
                # check gripper
                length_sum += norm_a - norm_ca
                if norm_a == 0 or norm_ca == 0:
                    if norm_ca == norm_a:
                        label = 1
                        evaluations += 1
                else:
                    cos_theta = dot_product / (norm_a * norm_ca)
                    angle = np.arccos(cos_theta) * 180 / np.pi
                    if angle < config["angle"]:
                        label = 1
                        evaluations += 1
                        angle_sum += angle

            if config["corrective"]:
                if label == 0:
                    action = corrected_action
                    label = -1

            next_camera_obs, next_proprio_obs, reward, done, could_perform_action = (
                env.step(action)
            )
            if not could_perform_action:
                err_steps += 1
            replay_memory.add(camera_obs, proprio_obs, action, [label])
            camera_obs, proprio_obs = next_camera_obs, next_proprio_obs
            loop_sleep(start_time)
        """ SAVE TRAJECTORY """
        camera_obs, proprio_obs = env.reset()
        if done:
            replay_memory.save_current_traj()
            successes += 1
            total_steps += step_count
            evaluation_rates += [evaluations / step_count]
            lengths += [length_sum / step_count]
            angles += [angle_sum / step_count]
            total_elapsed_time_success += episode_timer.elapsed()
        else:
            replay_memory.reset_current_traj()
        episode_counter += 1

        # abbruchbedingung wenn evaluative keine successfull task hinbekommt
        if config["evaluative"]:
            if episode_counter > config["episodes"] * 10:
                break

    # Save evaluation rates to csv
    data = pd.DataFrame(
        {"EvaluationRate": evaluation_rates, "Length": lengths, "Angle": angles}
    )
    max_i = 0
    tm_path = config["metrics_path"] + "/EvaluationMetrics/"
    # If necessary create folder structure
    if not os.path.exists(tm_path):
        os.makedirs(tm_path)
    for filename in os.listdir(tm_path):
        if filename.startswith("EvaluationMetrics_") and filename.endswith(".csv"):
            i = int(filename.split("_")[1].split(".")[0])
            if i > max_i:
                max_i = i
    filename = "EvaluationMetrics_" + str(max_i + 1) + ".csv"
    file_path = tm_path + filename
    data.to_csv(file_path, index=True)

    return {
        "Task": config["task"],
        "SuccessfullDemonstrations": successes,
        "TotalDemonstrations": episode_counter,
        "ElapsedTime": total_timer.elapsed(),
        "AvgTimeOnSuccess": -1
        if successes == 0
        else total_elapsed_time_success / successes,
        "EvaluationRateOnSuccess": np.mean(evaluation_rates),
        "EvaluationMetricsFilename": filename,
        "AvgStepsOnSuccess": -1 if successes == 0 else total_steps / successes,
    }


def main(config):
    env = LLM_iTeachCustomEnv(config)
    policy = Policy(config).to(device)
    if config["warmup"] == "None":
        replay_memory = TrajectoriesDataset(config["sequence_len"])
    else:
        warmup_control = config["warmup"].split("_")[0]
        warmup_demos = config["warmup"].split("_")[1]
        replay_memory = torch.load(
            f"{path}demonstrations/{warmup_control}/{config['task']}/demos_{warmup_demos}.dat"
        )
        model_path = f"{path}LearnedPolicies/{warmup_control}/{config['task']}/bc_policy_{warmup_demos}demos_800steps.pt"
        policy.load_state_dict(torch.load(model_path))
    policy.train()

    stop_flag = threading.Event()
    thread_return_dict = {"TrainingSteps": 0, "TrainingMetricsFile": "Not given"}

    training_loop = threading.Thread(
        target=train_step,
        args=(policy, replay_memory, config, stop_flag, thread_return_dict),
    )
    training_loop.start()
    result_dict = run_env_simulation(env, policy, replay_memory, config)

    if not config["verbose"]:
        print("\rFinished demonstrating, training for 60 more seconds.", end="")
    time.sleep(600)

    # Stop training
    stop_flag.set()
    training_loop.join()

    # Save policy
    policy_path = config["metrics_path"] + "/LearnedPolicies/" + config["task"] + "/"
    policy_filename = "LLM_iTeach_" + str(config["episodes"]) + "_policy.pt"
    if not os.path.exists(policy_path):
        os.makedirs(policy_path)
    torch.save(policy.state_dict(), policy_path + policy_filename)

    # Save metrics
    result_dict["TrainingSteps"] = thread_return_dict["TrainingSteps"]
    result_dict["TrainingMetrics"] = thread_return_dict["TrainingMetricsFile"]
    log_to_csv(result_dict, path=config["metrics_path"], file_name="training.csv")

    print(
        f"\rFinished training policy for task: {config['task']} with {config['episodes']} demonstrations",
        f" with warmup_policy {config['warmup']}" if config["warmup"] != "None" else "",
        f" and evaluation (Rate {result_dict['EvaluationRateOnSuccess']})"
        if config["evaluative"]
        else "",
        " " * 100,
    )


if __name__ == "__main__":
    set_seeds(1996)
    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        default="PushButton",
        help="options: PushButton, TakeLidOffSaucepan, UnplugCharger, CloseMicrowave, PutRubbishInBin",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        dest="episodes",
        default=100,
        type=int,
    )
    parser.add_argument(
        "-w",
        "--warmup",
        dest="warmup",
        default="None",
        help="use warmup policy specify by control_demonstrations --> None, llm_10, human_10, llm_100",
    )
    parser.add_argument(
        "-v",
        "--evaluative",
        dest="evaluative",
        action="store_true",
    )
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument(
        "-c",
        "--corrective",
        dest="corrective",
        action="store_true",
    )
    parser.add_argument(
        "-x",
        "--xperiment_name",
        dest="experiment_name",
        default="standard",
    )
    parser.add_argument(
        "-s",
        "--show",
        dest="show",
        action="store_true",
        help="Show GUI",
    )
    parser.add_argument(
        "--angle",
        dest="angle",
        default=20,
        type=float,
        help="Threshold for the similarity of the angle between corrective action and predicted action",
    )
    parser.add_argument(
        "--length",
        dest="length",
        default=3.0,
        type=float,
        help="Threshold for the similarity of the length between corrective action and predicted action",
    )
    args = parser.parse_args()

    metrics_path = path + "/Metrics/" + args.experiment_name
    config_defaults = {
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
        "sequence_len": 150,
        "warmup": args.warmup,
        "evaluative": args.evaluative,
        "metrics_path": metrics_path,
        "corrective": args.corrective,
        "verbose": args.verbose,
        "angle": args.angle,
        "length": args.length,
    }
    main(config_defaults)
