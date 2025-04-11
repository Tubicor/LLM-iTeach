# Query: push button
objects = ['button']
# Call Planner LMP: planner("push button")
if current_step == 1:
    # Call Action LMP: composer("move to the button")
    button = get_object('button')
    gripper = get_object('gripper')
    action = button.position - gripper.position
    move_gripper(action)
    # action done
    # Call Check LMP: composer("move to the button")
    button = get_object('button')
    gripper = get_object('gripper')
    if np.linalg.norm(gripper.position - button.position) < 0.02:
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
    # Call Action LMP: composer("push down by 5cm")
    gripper = get_object('gripper')
    goal_position = gripper.position + np.array([0, 0, -0.05])
    action = goal_position - gripper.position
    move_gripper(action)
    # action done
    # Call Check LMP: composer("push down by 5cm")
    gripper = get_object('gripper')
    goal_position = gripper.position + np.array([0, 0, -0.05])
    if np.linalg.norm(gripper.position - goal_position) < 0.02:
        increment_current_step()
    # done check
else:
    set_done()
