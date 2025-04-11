# Query: move the cube to the target area
objects = ['cube', 'target area']
# Call Planner LMP: planner("move the cube to the target area")
if current_step == 1:
    # Call Action LMP: composer("move to the cube")
    cube = get_object('cube')
    gripper = get_object('gripper')
    action = cube.position - gripper.position
    move_gripper(action)
    # action done
    # Call Check LMP: composer("move to the cube")
    cube = get_object('cube')
    gripper = get_object('gripper')
    if np.linalg.norm(gripper.position - cube.position) < 0.02:
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
    # Call Action LMP: composer("move to the target area")
    target_area = get_object('target area')
    gripper = get_object('gripper')
    action = target_area.position - gripper.position
    move_gripper(action)
    # action done
    # Call Check LMP: composer("move to the target area")
    target_area = get_object('target area')
    gripper = get_object('gripper')
    if np.linalg.norm(gripper.position - target_area.position) < 0.02:
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
