from dataclasses import dataclass
from enum import Enum
import rlbench.tasks as rlbench_tasks
import rlbench.backend.task as rlbench_backend

@dataclass
class Task:
    rlb_class: rlbench_backend.Task
    objects: list
    description: str
    name: str

class PushButton(Task):
    rlb_class = rlbench_tasks.PushButton
    name = "PushButton"
    objects = ["button"]
    description = "push button"

class TakeLidOffSaucepan(Task):
    rlb_class = rlbench_tasks.TakeLidOffSaucepan
    name = "TakeLidOffSaucepan"
    objects = ["lid"]#, "saucepan"]
    description = "lift the lid by 20cm" # Not the same because task needs to be transformed if Simulation API for CodePolicy should work 

class UnplugCharger(Task):
    rlb_class = rlbench_tasks.UnplugCharger
    name = "UnplugCharger"
    objects = ["charger","wall outlet"]
    description = "unplug charger by 20cm"

class CloseMicrowave(Task):
    rlb_class = rlbench_tasks.CloseMicrowave
    name = "CloseMicrowave"
    objects = ["microwave", "microwave door"]
    description = "close microwave 30cm"
class PutRubbishInBin(Task):
    rlb_class = rlbench_tasks.PutRubbishInBin
    name = "PutRubbishInBin"
    objects = ["rubbish", "bin", "tomato1", "tomato2"]
    description = "drop the rubbish into the bin"

class PickAndLift(Task):
    rlb_class = rlbench_tasks.PickAndLift
    name = "PickAndLift"
    objects = ["cube","target area"]
    description = "move the cube to the target area"

class ReachTarget(Task):
    rlb_class = rlbench_tasks.ReachTarget
    name = "ReachTarget"
    objects = ["target"]
    description = "reach the target"

class StackBlocks(Task):
    rlb_class = rlbench_tasks.StackBlocks
    name = "StackBlocks"
    objects = ["block1", "block2", "block3", "block4", "target"]
    description = "stack the 5cm blocks on each other at the target"
