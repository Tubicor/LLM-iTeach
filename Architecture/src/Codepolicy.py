import os
from utils import execute_string, path, TimeoutError
import numpy as np
import os
import tasks
import LMP
import time


class ExecutionError(Exception):
    pass


class WrongFormatError(Exception):
    pass


class CodepolicyError(Exception):
    # This class of Exceptions provide description for the LLM to reprompt
    pass


class Codepolicy:
    """Class to encapsule the Canidate that holds a reasoning within a python code snippet"""

    def __init__(self, task: tasks.Task, lmp: LMP, filename: str):
        self.task = task
        self.filename = filename
        self.lmp = lmp
        # set during execution
        self.current_step = None
        self.done = None
        self.gripper_default_position = None
        self.init_objects = None

    def execute(self, objects: dict):
        """Returns
        action: [dx:float ,dy:float ,dz: float ,roll: float ,pitch:float ,yaw: float ,gripper_open: bool]
        done: bool
        """

        # First executing => initialize
        if self.current_step == None:
            self.current_step = 1
            self.done = False
            self.gripper_default_position = (
                objects["gripper"].position - np.array([0, 0, 0.3])
            )  # The start position is at maximum height and the used algorithm to translate from xyz to joint angles is not able to reliably reach that height again after moving down
            self.init_objects = objects

        # Get global variables and callables for LMP
        variables = {
            "np": np,
            "current_step": self.current_step,
            "done": self.done,
            "gripper_default_position": self.gripper_default_position,
        }
        sim_api = SimulationAPI(
            objects, self.done, self.current_step, self.init_objects
        )
        callables = {
            k: getattr(sim_api, k)
            for k in dir(sim_api)
            if callable(getattr(sim_api, k)) and not k.startswith("_")
        }
        globals = {**variables, **callables}

        # Execute LMP
        try:
            execute_string(self.lmp, globals, False)
        except TimeoutError:
            with open(path + "execution_time.txt", "a") as file:
                file.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}, Timeout Error with executing code in Task: {self.task.name}\n"
                )
        # If state has changed => update variables
        if sim_api._get_current_step() != self.current_step:
            self.current_step = sim_api._get_current_step()
            self.init_objects = objects

        # Check weather CodePolicy has reached final state
        if sim_api._get_done():
            self.done = sim_api._get_done()

        # Return action
        return sim_api._get_action(), self.done

    # ======================================================
    # == functions handeling Persistence of Codepolicies
    # ======================================================
    _filename_prexif = "CodePolicy_"
    _filename_suffix = ".py"
    _path_suffix = "CodePolicies/"

    def _get_path(task: tasks.Task):
        return path + Codepolicy._path_suffix + "/" + task.name + "/"

    def __str__(self):
        return f"{self.filename}"

    def create(task: tasks.Task, lmp: str):
        # If necessary create folder structure
        path = Codepolicy._get_path(task)
        if not os.path.exists(path):
            os.makedirs(path)
        # get available Codepolicy number
        max_i = 0
        for filename in os.listdir(path):
            if filename.startswith(Codepolicy._filename_prexif) and filename.endswith(
                Codepolicy._filename_suffix
            ):
                i = int(filename.split("_")[1].split(".")[0])
                if i > max_i:
                    max_i = i
        filename = (
            Codepolicy._filename_prexif + str(max_i + 1) + Codepolicy._filename_suffix
        )
        # Save code to file
        with open(path + filename, "w") as file:
            file.write(lmp)
        return Codepolicy(task, lmp, filename)

    def load(task: tasks.Task, filenames=[]):
        """Loads a List of Canidates supplied in the provided List of filenames
        If No filenames are provided all Codepolicys in the path are loaded.
        """
        codepolicies = []
        path = Codepolicy._get_path(task)
        # if no filenames are provided load all
        if len(filenames) == 0:
            filenames = os.listdir(path)

        for filename in filenames:
            # check correct filename format
            if filename.startswith(Codepolicy._filename_prexif) and filename.endswith(
                Codepolicy._filename_suffix
            ):
                if os.path.isfile(path + filename):
                    with open(path + filename, "r") as file:
                        code = file.read()
                        codepolicies.append(Codepolicy(task, code, filename))
        return codepolicies


class SimulationAPI:
    """Class to provide API functions for Codepolicy and API for accessing return values"""

    def __init__(self, objects: dict, done, current_step, init_objects):
        self.objects: dict = objects
        self._action = [
            0,
            0,
            0,
            0,
        ]  # [dx,dy,dz,gripper_action] gripperaction = -1:close, 0:do nothing, 1:open
        self._done = done
        self._current_step = current_step
        self._init_objects = init_objects

    # ======================================================
    # == API functions for CodePolicy
    # ======================================================

    def get_object(self, name: str):
        for key, value in self.objects.items():
            if key.lower() == name.lower():
                value.initial_position = self._init_objects[key].position
                return value

    def is_gripper_open(self):
        return not self.objects["gripper"].closed

    def move_gripper(self, action):
        if len(action) == 3:
            self._action[:3] = action
        else:
            pass

    def set_gripper(self, action):
        if action == True:
            self._action[-1] = 1
        elif action == False:
            self._action[-1] = -1
        else:
            pass

    def set_done(self):
        self._done = True

    def increment_current_step(self):
        self._current_step += 1

    # ======================================================
    # == API functions for accessing values of CodePolicy's execution
    # ======================================================

    def _get_action(self):
        return self._action

    def _get_current_step(self):
        return self._current_step

    def _get_done(self):
        return self._done


def generate_codepolicy(task: tasks.Task) -> Codepolicy:
    """Generates a new Codepolicy for the provided Task
    returns codepolicy if successfull, else None
    """
    # generate LMP
    lmp = LMP.hierarchial_code_generation(task.description, task.objects)
    # check executability
    executable = True  # TODO
    if not executable:
        return None
    # create Codepolicy
    codepolicy = Codepolicy.create(task, lmp)
    return codepolicy


if __name__ == "__main__":
    generate_codepolicy(tasks.PushButton)
    print("Done with PushButton")
