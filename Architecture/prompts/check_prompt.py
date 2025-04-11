import numpy as np
from env_utils import increment_current_step, get_object, gripper_default_position

# Check action: push in drawer by 25cm
drawer = get_object('drawer')
# specified direction is in so negative normal
goal_position = drawer.initial_position + drawer.normal * -0.25
if np.linalg.norm(drawer.position - goal_position) < 0.02:
    increment_current_step()
# done check

# Check action: move gripper 20cm at the right side of the bowl
bowl = get_object('bowl')
# right side so negative y-axis
goal_position = bowl.position + np.array([0, -0.2, 0])
gripper = get_object('gripper')
if np.linalg.norm(gripper.position - goal_position) < 0.02:
    increment_current_step()
# done check

# Check action: close gripper
gripper = get_object('gripper')
if gripper.closed:
    increment_current_step()
# done check

# Check action: push the switch down by 5cm
switch = get_object('switch')
gripper = get_object('gripper')
# down so negative z-axis
goal_position = switch.position + np.array([0, 0, -0.05])
if np.linalg.norm(gripper.position - goal_position) < 0.02:
    increment_current_step()
# done check

# Check action: push the blue block 10cm in front
blue_block = get_object('blue block')
gripper = get_object('gripper')
# in front so x axis
goal_position = blue_block.position + np.array([0.1, 0, 0])
if np.linalg.norm(gripper.position - goal_position) < 0.02:
    increment_current_step()
# done check

# Check action: move 5cm above the yellow block
yellow_block = get_object('yellow block')
gripper = get_object('gripper')
# above so z-axis
goal_position = yellow_block.position + np.array([0, 0, 0.05])
if np.linalg.norm(gripper.position - goal_position) < 0.02:
    increment_current_step()
# done check

# Check action: pull the cable away from the outlet by 25cm
outlet = get_object('outlet')
plug = get_object('cable')
# no specified direction so absolute distance over all axes
if np.linalg.norm(plug.position - outlet.position) > (0.25):
    increment_current_step()
# done check

# Check action: back to default pose
gripper = get_object('gripper')
if np.linalg.norm(gripper.position - gripper_default_position) < 0.02:
    increment_current_step()
# done check

# Check action: move up by 50cm
gripper = get_object('gripper')
# up so z-axis
goal_position = gripper.inital_position + np.array([0, 0, 0.5])
if np.linalg.norm(gripper.position - goal_position) < 0.02:
    increment_current_step()
# done check

# Check action: pull back by 10cm
gripper = get_object('gripper')
# back so negative in x-axis
goal_position = gripper.inital_position + np.array([-0.1, 0, 0])
if np.linalg.norm(gripper.position - goal_position) < 0.02:
    increment_current_step()
# done check

# Check action: move to the lemon
lemon = get_object('lemon')
gripper = get_object('gripper')
if np.linalg.norm(gripper.position - lemon.position) < 0.02:
    increment_current_step()
# done check

# Check action: open gripper
gripper = get_object('gripper')
if not gripper.closed:
    increment_current_step()
# done check

# Check action: move to center of the switch
switch = get_object('switch')
gripper = get_object('gripper')
if np.linalg.norm(gripper.position - switch.position) < 0.02:
    increment_current_step()
# done check
