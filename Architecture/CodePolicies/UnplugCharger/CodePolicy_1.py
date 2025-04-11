# Query: unplug charger by 20cm
objects = ['charger', 'wall outlet']
# Call Planner LMP: planner("unplug charger by 20cm")
if current_step == 1:
    # Call Action LMP: composer("move to the charger")
    charger = get_object('charger')
    gripper = get_object('gripper')
    action = charger.position - gripper.position
    move_gripper(action)
    # action done
    # Call Check LMP: composer("move to the charger")
    charger = get_object('charger')
    gripper = get_object('gripper')
    if np.linalg.norm(gripper.position - charger.position) < 0.02:
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
    # Call Action LMP: composer("pull out from wall outlet by 20cm")
    wall_outlet = get_object('wall outlet')
    gripper = get_object('gripper')
    goal_position = wall_outlet.position + wall_outlet.normal * 0.2
    action = goal_position - gripper.position
    move_gripper(action)
    # action done
    # Call Check LMP: composer("pull out from wall outlet by 20cm")
    wall_outlet = get_object('wall outlet')
    gripper = get_object('gripper')
    goal_position = wall_outlet.position + np.array([0.2, 0, 0])
    if np.linalg.norm(gripper.position - goal_position) < 0.02:
        increment_current_step()
    # done check
else:
    set_done()
