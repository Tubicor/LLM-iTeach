# Query: lift the lid by 20cm
objects = ['lid']
# Call Planner LMP: planner("lift the lid by 20cm")
if current_step == 1:
    # Call Action LMP: composer("move to the lid")
    lid = get_object('lid')
    gripper = get_object('gripper')
    action = lid.position - gripper.position
    move_gripper(action)
    # action done
    # Call Check LMP: composer("move to the lid")
    lid = get_object('lid')
    gripper = get_object('gripper')
    if np.linalg.norm(gripper.position - lid.position) < 0.02:
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
    # Call Action LMP: composer("move 20cm up")
    gripper = get_object('gripper')
    goal_position = gripper.position + np.array([0, 0, 0.2])
    action = goal_position - gripper.position
    move_gripper(action)
    # action done
    # Call Check LMP: composer("move 20cm up")
    gripper = get_object('gripper')
    goal_position = gripper.position + np.array([0, 0, 0.2])
    if np.linalg.norm(gripper.position - goal_position) < 0.02:
        increment_current_step()
    # done check
else:
    set_done()
