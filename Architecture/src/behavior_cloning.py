import torch
from models import Policy
from utils import TrajectoriesDataset  # noqa: F401
from utils import device, set_seeds, print_progressbar, path
from argparse import ArgumentParser
import os
import pandas as pd


def train_step(policy, replay_memory, config):
    steps = config["steps"]
    data = pd.DataFrame(columns=["Loss","Step"])
    for step in range(steps):
        if not config["verbose"]:
            print_progressbar(step, steps, prefix="Training policy", suffix="Complete")
        batch = replay_memory.sample(config["batch_size"])
        camera_batch, proprio_batch, action_batch, feedback_batch = batch
        training_metrics = policy.update_params(
            camera_batch, proprio_batch, action_batch, feedback_batch
        )
        data.loc[len(data)] = {"Loss":training_metrics["loss"].item(),"Step":step}
    filepath =  path+f"Metrics/BC_OtherTasks/TrainingMetrics/"
    filename = f"{config['task']}_{config['demonstrations']}.csv"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    data.to_csv(filepath+filename)
    print("Behavior Cloning Training Complete for Task: ", config["task"]," "*100)
    return


def main(config):
    replay_memory = torch.load(f"{path}/demonstrations/{config['control']}/{config['task']}/demos_{config['demonstrations']}.dat")
    policy = Policy(config).to(device)
    train_step(policy, replay_memory, config)
    file_folder = path+"/LearnedPolicies/" + config["control"] + "/" + config["task"] + "/"
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)
    torch.save(policy.state_dict(), file_folder + f"bc_policy_{config['demonstrations']}demos_{config['steps']}steps.pt")
    return


if __name__ == "__main__":
    set_seeds(1)
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--control",
        dest="control",
        default="human",
        help="control type of the recorded demos options: llm, human",
    )
    parser.add_argument(
        "-d",
        "--demonstrations",
        dest="demonstrations",
        default=10,
        type=int,
        help="number of demonstrations to perform behavior cloning on",
    )
    parser.add_argument(
        "-s",
        "--steps",
        dest="steps",
        default=800,
        type=int,
        help="steps to train the policy",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        default="PushButton",
        help="options: CloseMicrowave, PushButton, TakeLidOffSaucepan, UnplugCharger, PutRubbishInBin",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true"
    )
    args = parser.parse_args()
    config = {
        "control": args.control,
        "task": args.task,
        "proprio_dim": 8,
        "action_dim": 7,
        "visual_embedding_dim": 256,
        "learning_rate": 3e-4,
        "weight_decay": 3e-6,
        "batch_size": 16,
        "steps": args.steps,
        "demonstrations": args.demonstrations,#
        "verbose": args.verbose
    }
    main(config)
