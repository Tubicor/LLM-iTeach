import re
import signal
import sys
from io import StringIO
import random
import torch
import time
import numpy as np
from torch.utils.data import Dataset 
import pandas as pd
import os



path = os.getcwd()+"/Architecture/"
if not os.path.exists(path):
    raise Exception("Command not executed from Master Thesis folder")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
try: 
    device = torch.device(os.environ["CUDA_DEVICE"])
except:
    pass
class Timer():
    def __init__(self):
        self.start_time = time.time()
    def reset(self):
        self.start_time = time.time()
    def elapsed(self, decimals=1):
        return round(time.time() - self.start_time, decimals)

def log_to_file(line:str,file_name:str="results.log"):
    with open(path+file_name, "a") as file:
        file.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')} - {line}\n")

def log_to_csv(new_row:dict,path:str,file_name:str):
    new_row["Time"] = pd.to_datetime(time.strftime('%Y-%m-%d %H:%M:%S'))
    if not os.path.exists(path+"/"+file_name):
        # Create folder structure if necessary
        if not os.path.exists(path):
            os.makedirs(path)
        data = pd.DataFrame([new_row])
    else:
        data = pd.read_csv(path+"/"+file_name, delimiter=',')
        data.loc[len(data)] = new_row
    data.to_csv(path+"/"+file_name, index=False)

def vector_from_text(text: str):
    """Returns the first Vector from string as a list of float. If not findable returns False"""
    result = []
    pattern = r"-?\d*\.?\d+[,\s*-?\d*\.?\d+]*"
    match = re.search(pattern, text)
    try:
        if match:
            numbers = re.findall(r'[-]?\d*\.?\d+', match.group())
            decimal_numbers = [float(num) for num in numbers]
            result = decimal_numbers
    except:
        pass
    return result

def find_tags_content(text: str, start_tag: str, end_tag: str):
    """Returns the content of all tag pairs in the text"""
    pattern = f"{start_tag}(.*?){end_tag}"
    findings = re.findall(pattern, text,re.DOTALL)
    findings = [x.strip(" \n") for x in findings]
    return findings

class TimeoutError(Exception):
    pass

def execute_string(code,functions,suppress_output=True): 
    """ Executes string
        Returns True if successful
        Returns error message if not successful
    """ 
    original_stdout = None
    original_stderr = None
    fake_stdout = None
    fake_stderr = None

    if suppress_output:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        fake_stdout = StringIO()
        fake_stderr = StringIO()
        sys.stdout = fake_stdout
        sys.stderr = fake_stderr
    try:
        def _handle_timeout(signum, frame):
            raise TimeoutError("Function call timed out")
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(10)
        # with open(path+"execution_time.txt", "a") as file:
        #     file.write(f"{signal.getitimer(signal.ITIMER_REAL)[0]:.2f}\n")
        exec(code, functions)

    # except TimeoutError as e:
    #     return "Timeout Error with executing code"
    # except SyntaxError as e:
    #     return f"SyntaxError with executing code {e}"
    # except Exception as e:
    #     return f"Exception with executing code {e}"
    finally:
        signal.alarm(0)
        if suppress_output:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            # suppressed_output = fake_stdout.getvalue()

    return True

def distance_between_vectors(vector1, vector2):
    return sum([(vector1[i] - vector2[i])**2 for i in range(3)])**0.5

class TrajectoriesDataset(Dataset):
    def __init__(self, sequence_len):
        self.sequence_len = sequence_len
        self.camera_obs = []
        self.proprio_obs = []
        self.action = []
        self.feedback = []
        self.reset_current_traj()
        self.pos_count = 0
        self.cor_count = 0
        self.neg_count = 0
        return

    def __getitem__(self, idx):
        if self.cor_count < 10:
            alpha = 1
        else:
            alpha = (self.pos_count + self.neg_count + self.cor_count ) / self.cor_count
        weighted_feedback = [
            alpha if value == -1 else value for value in self.feedback[idx]
        ]
        weighted_feedback = torch.tensor(weighted_feedback).unsqueeze(1)
        return (
            self.camera_obs[idx],
            self.proprio_obs[idx],
            self.action[idx],
            weighted_feedback,
        )

    def __len__(self):
        return len(self.proprio_obs)

    def add(self, camera_obs, proprio_obs, action, feedback):
        self.current_camera_obs.append(camera_obs)
        self.current_proprio_obs.append(proprio_obs)
        self.current_action.append(action)
        self.current_feedback.append(feedback)
        if feedback[0] == 1:
            self.pos_count += 1
        elif feedback[0] == -1:
            self.cor_count += 1
        elif feedback[0] == 0:
            self.neg_count += 1
        return

    def save_current_traj(self):
        camera_obs = downsample_traj(self.current_camera_obs, self.sequence_len)
        proprio_obs = downsample_traj(self.current_proprio_obs, self.sequence_len)
        action = downsample_traj(self.current_action, self.sequence_len)
        feedback = downsample_traj(self.current_feedback, self.sequence_len)
        camera_obs_th = torch.tensor(camera_obs, dtype=torch.float32)
        proprio_obs_th = torch.tensor(proprio_obs, dtype=torch.float32)
        action_th = torch.tensor(action, dtype=torch.float32)
        feedback_th = torch.tensor(feedback, dtype=torch.float32)
        self.camera_obs.append(camera_obs_th)
        self.proprio_obs.append(proprio_obs_th)
        self.action.append(action_th)
        self.feedback.append(feedback_th)
        self.reset_current_traj()
        return

    def reset_current_traj(self):
        self.current_camera_obs = []
        self.current_proprio_obs = []
        self.current_action = []
        self.current_feedback = []
        return

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self))
        indeces = random.sample(range(len(self)), batch_size)
        batch = zip(*[self[i] for i in indeces])
        camera_batch = torch.stack(next(batch), dim=1)
        proprio_batch = torch.stack(next(batch), dim=1)
        action_batch = torch.stack(next(batch), dim=1)
        feedback_batch = torch.stack(next(batch), dim=1)
        return camera_batch, proprio_batch, action_batch, feedback_batch

def downsample_traj(traj, target_len):
    if len(traj) == target_len:
        return np.array(traj)
    elif len(traj) < target_len:
        return np.array(traj + [traj[-1]] * (target_len - len(traj)))
    else:
        indeces = np.linspace(start=0, stop=len(traj) - 1, num=target_len)
        indeces = np.round(indeces).astype(int)
        return np.array([traj[i] for i in indeces])


def loop_sleep(start_time):
    dt = 0.05
    sleep_time = dt - (time.time() - start_time)
    if sleep_time > 0.0:
        time.sleep(sleep_time)
    return


def euler_to_quaternion(euler_angle):
    roll, pitch, yaw = euler_angle
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [qx, qy, qz, qw]

def quaternion_to_euler(quaternion):
    qx, qy, qz, qw = quaternion
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = np.arcsin(2 * (qw * qy - qz * qx))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return [roll, pitch, yaw]

def set_seeds(seed=0):
    """Sets all seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def print_progressbar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
            """
            Call in a loop to create terminal progress bar.

            Args:
                iteration (int): Current iteration.
                total (int): Total iterations.
                prefix (str, optional): Prefix string. Defaults to ''.
                suffix (str, optional): Suffix string. Defaults to ''.
                decimals (int, optional): Positive number of decimals in percent complete. Defaults to 1.
                length (int, optional): Character length of bar. Defaults to 50.
                fill (str, optional): Bar fill character. Defaults to '█'.
            """
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filled_length = int(length * iteration // total)
            bar = fill * filled_length + '-' * (length - filled_length)
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
            # Print New Line on Complete
            if iteration == total:
                print()

from collections import Counter
def most_frequent(arr):
    # Convert the NumPy array to a list
    arr_list = arr.tolist()
    # Use Counter to count occurrences
    counter = Counter(arr_list)
    # Find the most common element
    most_common_element, _ = counter.most_common(1)[0]
    return most_common_element
