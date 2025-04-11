# Query: reach the target
objects = ['target']
# Call Planner LMP: planner("reach the target")
if current_step == 1:
    # Call Action LMP: composer("move to the target")
    target = get_object('target')
    gripper = get_object('gripper')
    action = target.position - gripper.position
    move_gripper(action)
    # action done
    # Call Check LMP: composer("move to the target")
    target = get_object('target')
    gripper = get_object('gripper')
    if np.linalg.norm(gripper.position - target.position) < 0.02:
        increment_current_step()
else:
    set_done()
