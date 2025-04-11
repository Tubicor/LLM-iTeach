import numpy as np
from env_utils import current_step, set_done
from action_utils import composer

objects = ['blue block', 'yellow block', 'mug']
# Query: place the blue block on the yellow block.
if current_step == 1:
    composer("move to blue block")
elif current_step == 2:
    composer("close gripper")
elif current_step == 3:
    composer("back to default pose")
elif current_step == 4:
    composer("move 5cm above the yellow block")
elif current_step == 5:
    composer("open gripper")
else:
    set_done()
# done

objects = ['table', 'apple']
# Query: lift apple off by 20cm
if current_step == 1:
    composer("move to the apple")
elif current_step == 2:
    composer("close gripper")
elif current_step == 3:
    composer("move 20cm up")
else:
    set_done()
# done

objects = ['airpods', 'drawer', 'drawer handle']
# Query: open the drawer.
if current_step == 1:
    composer("move to the drawer handle")
elif current_step == 2:
    composer("close gripper")
elif current_step == 3:
    composer("pull out from drawer by 25cm")
else:
    set_done()
# done

objects = ['tissue box', 'tissue', 'bowl']
# Query: place tissue next to the bowl?
if current_step == 1:
    composer("move to tissue")
elif current_step == 2:
    composer("close gripper")
elif  current_step == 3:
    composer("back to default pose")
elif  current_step == 4:
    composer("move gripper 20cm at the right side of the bowl")
else:
    set_done()
# done

objects = ['drawer', 'drawer handle']
# Query: close drawer 25cm
if current_step == 1:
    composer("move to the drawer handle")
elif  current_step == 2:
    composer("close gripper")
elif  current_step == 3:
    composer("push in drawer by 25cm")
else:
    set_done()
# done

objects = ['orange', 'QR code', 'lemon', 'drawer', 'drawer handle']
# Query: put the sour fruit into the drawer.
if current_step == 1:
    composer("move to the drawer handle")
elif  current_step == 2:
    composer("close gripper")
elif  current_step == 3:
    composer("move away from the drawer by 25cm")
elif  current_step == 4:
    composer("open gripper")
elif  current_step == 5:
    composer("back to default pose")
elif  current_step == 6:
    composer("move to the lemon")
elif  current_step == 7:
    composer("close gripper")
elif  current_step == 8:
    composer("move in the drawer")
elif  current_step == 9:
    composer("open gripper")
elif current_step == 10:
    composer("back to default pose")
else:    
    set_done()
# done

objects = ['closet', 'closet door']
# Query: close the closet
if current_step == 1:
    composer("move to the closet door")
elif  current_step == 2:
    composer("close gripper")
elif  current_step == 3:
    composer("push in closet door by 30cm")
else:
    set_done()
# done

objects = ['cup1', 'cup2' 'blue area']
# Query: move the cups to the blue area
if current_step == 1:
    composer("move to the cup1")
elif  current_step == 2:
    composer("close gripper")
elif current_step == 3:
    composer("back to default pose")
elif current_step == 4:
    composer("move to the blue area")
elif current_step == 5:
    composer("open gripper")
elif current_step == 6:
    composer("return to default pose")
elif current_step == 7:
    composer("move to the cup2")
elif current_step == 8:
    composer("close gripper")
elif current_step == 9:
    composer("back to default pose")
elif current_step == 10:
    composer("move to the blue area")
elif current_step == 11:
    composer("open gripper")
elif current_step == 12:
    composer("back to default pose")
else:
    set_done()
# done

objects = ['lamp', 'switch']
# Query: Turn off the lamp.
if current_step == 1:
    composer("close the gripper")
elif  current_step == 2:
    composer("move to the center of the switch")
elif  current_step == 3:
    composer("push down by 5cm")
else:
    set_done()
# done