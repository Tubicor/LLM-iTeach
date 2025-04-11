# Query: stack the 5cm blocks on each other at the target
objects = ['block1', 'block2', 'block3', 'block4', 'target']
# Call Planner LMP: planner("stack the 5cm blocks on each other at the target")
if current_step == 1:
    # Call Action LMP: composer("move to block1")
    block1 = get_object('block1')
    gripper = get_object('gripper')
    action = block1.position - gripper.position
    move_gripper(action)
    # action done
    # Call Check LMP: composer("move to block1")
    block1 = get_object('block1')
    gripper = get_object('gripper')
    if np.linalg.norm(gripper.position - block1.position) < 0.02:
        increment_current_step()
    # done check
elif current_step == 2:
    # Call Action LMP: composer("close gripper")
    set_gripper(False)
    # Call Check LMP: composer("close gripper")
    gripper = get_object('gripper')
    if gripper.closed:
        increment_current_step()
elif current_step == 3:
    # Call Action LMP: composer("back to default pose")
    gripper = get_object('gripper')
    goal_position = gripper_default_position
    action = goal_position - gripper.position
    move_gripper(action)
    # Call Check LMP: composer("back to default pose")
    gripper = get_object('gripper')
    if np.linalg.norm(gripper.position - gripper_default_position) < 0.02:
        increment_current_step()
elif current_step == 4:
    # Call Action LMP: composer("move 5cm above the target")
    target = get_object('target')
    gripper = get_object('gripper')
    goal_position = target.position + np.array([0, 0, 0.05])
    action = goal_position - gripper.position
    move_gripper(action)
    # action done
    # Call Check LMP: composer("move 5cm above the target")
    target = get_object('target')
    gripper = get_object('gripper')
    goal_position = target.position + np.array([0, 0, 0.05])
    if np.linalg.norm(gripper.position - goal_position) < 0.02:
        increment_current_step()
    # done check
elif current_step == 5:
    # Call Action LMP: composer("open gripper")
    set_gripper(True)
    # Call Check LMP: composer("open gripper")
    gripper = get_object('gripper')
    if not gripper.closed:
        increment_current_step()
elif current_step == 6:
    # Call Action LMP: composer("back to default pose")
    gripper = get_object('gripper')
    goal_position = gripper_default_position
    action = goal_position - gripper.position
    move_gripper(action)
    # Call Check LMP: composer("back to default pose")
    gripper = get_object('gripper')
    if np.linalg.norm(gripper.position - gripper_default_position) < 0.02:
        increment_current_step()
elif current_step == 7:
    # Call Action LMP: composer("move to block2")
    block2 = get_object('block2')
    gripper = get_object('gripper')
    action = block2.position - gripper.position
    move_gripper(action)
    # action done
    # Call Check LMP: composer("move to block2")
    block2 = get_object('block2')
    gripper = get_object('gripper')
    if np.linalg.norm(gripper.position - block2.position) < 0.02:
        increment_current_step()
    # done check
elif current_step == 8:
    # Call Action LMP: composer("close gripper")
    set_gripper(False)
    # Call Check LMP: composer("close gripper")
    gripper = get_object('gripper')
    if gripper.closed:
        increment_current_step()
elif current_step == 9:
    # Call Action LMP: composer("back to default pose")
    gripper = get_object('gripper')
    goal_position = gripper_default_position
    action = goal_position - gripper.position
    move_gripper(action)
    # Call Check LMP: composer("back to default pose")
    gripper = get_object('gripper')
    if np.linalg.norm(gripper.position - gripper_default_position) < 0.02:
        increment_current_step()
elif current_step == 10:
    # Call Action LMP: composer("move 5cm above the block1 on target")
    block1 = get_object('block1')
    gripper = get_object('gripper')
    goal_position = block1.position + np.array([0, 0, 0.05])
    action = goal_position - gripper.position
    move_gripper(action)
    # action done
    # Call Check LMP: composer("move 5cm above the block1 on target")
    block1 = get_object('block1')
    gripper = get_object('gripper')
    goal_position = block1.position + np.array([0, 0, 0.05])
    if np.linalg.norm(gripper.position - goal_position) < 0.02:
        increment_current_step()
elif current_step == 11:
    # Call Action LMP: composer("open gripper")
    set_gripper(True)
    # Call Check LMP: composer("open gripper")
    gripper = get_object('gripper')
    if not gripper.closed:
        increment_current_step()
elif current_step == 12:
    # Call Action LMP: composer("back to default pose")
    gripper = get_object('gripper')
    goal_position = gripper_default_position
    action = goal_position - gripper.position
    move_gripper(action)
    # Call Check LMP: composer("back to default pose")
    gripper = get_object('gripper')
    if np.linalg.norm(gripper.position - gripper_default_position) < 0.02:
        increment_current_step()
else:
    set_done()
