import numpy as np


def is_finger_closed(lm, origin_id, finger_tip_id, use_3d=False):
    if use_3d:
        origin = np.array([lm[origin_id].x, lm[origin_id].y, lm[origin_id].z])
        tip = np.array([lm[finger_tip_id].x, lm[finger_tip_id].y, lm[finger_tip_id].z])
        crease = np.array([lm[finger_tip_id - 2].x, lm[finger_tip_id - 2].y, lm[finger_tip_id - 2].z])
    else:
        origin = np.array([lm[origin_id].x, lm[origin_id].y])
        tip = np.array([lm[finger_tip_id].x, lm[finger_tip_id].y])
        crease = np.array([lm[finger_tip_id - 2].x, lm[finger_tip_id - 2].y])

    d_origin_tip = np.linalg.norm(tip - origin)
    d_origin_crease = np.linalg.norm(crease - origin)
    ratio = d_origin_tip / (d_origin_crease + 1e-6)
    return ratio < 0.975


def are_fingers_closed(lm):
    return is_finger_closed(lm, 0, 20) \
        and is_finger_closed(lm, 0, 16) \
        and is_finger_closed(lm, 0, 12)


def is_correct_gesture(hand):
    lm = hand.landmark
    return is_finger_closed(lm, 4, 8, True) and are_fingers_closed(lm)
