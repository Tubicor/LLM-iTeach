# Query: close microwave 30cm
objects = ['microwave', 'microwave door']
# Call Planner LMP: planner("close microwave 30cm")
if current_step == 1:
    # Call Action LMP: composer("move to the microwave door")
    microwave_door = get_object('microwave door')
    gripper = get_object('gripper')
    action = microwave_door.position - gripper.position
    move_gripper(action)
    # action done
    # Call Check LMP: composer("move to the microwave door")
    microwave_door = get_object('microwave door')
    gripper = get_object('gripper')
    if np.linalg.norm(gripper.position - microwave_door.position) < 0.02:
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
    # Call Action LMP: composer("push in microwave door by 30cm")
    microwave = get_object('microwave')
    gripper = get_object('gripper')
    goal_position = microwave.position + microwave.normal * -0.3
    action = goal_position - gripper.position
    move_gripper(action)
    # action done
    # Call Check LMP: composer("push in microwave door by 30cm")
    microwave = get_object('microwave')
    goal_position = microwave.position + np.array([0.3, 0, 0])
    gripper = get_object('gripper')
    if np.linalg.norm(gripper.position - goal_position) < 0.02:
        increment_current_step()
    # done check
else:
    set_done()
