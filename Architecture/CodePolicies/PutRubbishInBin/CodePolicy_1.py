# Query: drop the rubbish into the bin
objects = ['rubbish', 'bin', 'tomato1', 'tomato2']
# Call Planner LMP: planner("drop the rubbish into the bin")
if current_step == 1:
    # Call Action LMP: composer("move to the rubbish")
    rubbish = get_object('rubbish')
    gripper = get_object('gripper')
    action = rubbish.position - gripper.position
    move_gripper(action)
    # action done
    # Call Check LMP: composer("move to the rubbish")
    rubbish = get_object('rubbish')
    gripper = get_object('gripper')
    if np.linalg.norm(gripper.position - rubbish.position) < 0.02:
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
    # Call Action LMP: composer("move above the bin by 10cm")
    bin = get_object('bin')
    gripper = get_object('gripper')
    goal_position = bin.position + np.array([0, 0, 0.1])
    action = goal_position - gripper.position
    move_gripper(action)
    # action done
    # Call Check LMP: composer("move above the bin by 10cm")
    bin = get_object('bin')
    gripper = get_object('gripper')
    goal_position = bin.position + np.array([0, 0, 0.1])
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
else:
    set_done()
