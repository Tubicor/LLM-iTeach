import numpy as np
from env_utils import get_object, gripper_default_position, move_gripper, set_gripper

# Action: push in drawer by 25cm
gripper = get_object('gripper')
drawer = get_object('drawer')
# in direction of drawer so negative normal
goal_position = drawer.position + drawer.normal * -0.25
action = goal_position - gripper.position
move_gripper(action)
# action done

# Action: move gripper 20cm at the right side of the bowl
bowl = get_object('bowl')
gripper = get_object('gripper')
# right side so negative y-axis
goal_position = bowl.position + np.array([0, -0.2, 0])
action = goal_position - gripper.position
move_gripper(action)
# action done

# Action: close gripper
set_gripper(False)
# action done

# Action: open gripper
set_gripper(True)
# action done

# Action: push the blue block 10cm in front
blue_block = get_object('blue block')
gripper = get_object('gripper')
# in front so x axis
goal_position = blue_block.position + np.array([0.1, 0, 0])
action =  goal_position - gripper.position
move_gripper(action)
# action done

# Action: move to the door
door = get_object('door')
gripper = get_object('gripper')
action = door.position - gripper.position
move_gripper(action)
# action done

# Action: move 5cm above the yellow block
yellow_block = get_object('yellow block')
gripper = get_object('gripper')
# 5cm above so we add to z-axis
goal_position = yellow_block.position + np.array([0, 0, 0.05])
action = goal_position - gripper.position
move_gripper(action)
# action done

# Action: pull out from the drawer by 25cm
drawer = get_object('drawer')
gripper = get_object('gripper')
# out from the drawer so positive normal
goal_position = drawer.position + drawer.normal * 0.25
action = goal_position - gripper.position
move_gripper(action)
# action done

# Action: back to default pose
gripper = get_object('gripper')
goal_position = gripper_default_position
action = goal_position - gripper.position
move_gripper(action)
# action done

# Action: move 20cm above the table
table = get_object('table')
gripper = get_object('gripper')
# up so z-axis
goal_position = table.position + np.array([0, 0, 0.2])
action = goal_position - gripper.position
move_gripper(action)
# action done

# Action: move back by 10cm
gripper = get_object('gripper')
# back so negative in x-axis
goal_position = gripper.position + np.array([-0.1, 0, 0])
action = goal_position - gripper.position
move_gripper(action)
# action done

# Action: move to the lemon
lemon = get_object('lemon')
gripper = get_object('gripper')
action = lemon.position - gripper.position
move_gripper(action)
# action done

# Action: move to the switch
switch = get_object('switch')
gripper = get_object('gripper')
action = switch.position - gripper.position
move_gripper(action)
# action done
