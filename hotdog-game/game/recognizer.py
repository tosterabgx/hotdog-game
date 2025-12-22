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
    return ratio < 1


def are_fingers_closed(lm):
    return is_finger_closed(lm, 0, 20) \
        and is_finger_closed(lm, 0, 16) \
        and is_finger_closed(lm, 0, 12)

def is_front(lm, is_left):
    thumb_pos = lm[4].x
    pinky_pos = lm[20].x
    if is_left:
        return thumb_pos > pinky_pos
    return thumb_pos < pinky_pos


def is_correct_gesture(hand, is_left):
    lm = hand.landmark
    return is_finger_closed(lm, 4, 8, True) and are_fingers_closed(lm) and is_front(lm, is_left)


def is_mouth_open(face):
    lm = face.landmark

    upper_lip_top = 0
    upper_lip_bottom = 13
    bottom_lip_top = 14

    upper_lip_size = abs(lm[upper_lip_top].y - lm[upper_lip_bottom].y)
    mouth_size = abs(lm[upper_lip_bottom].y - lm[bottom_lip_top].y)

    ratio = upper_lip_size / (mouth_size + 1e-6)
    return ratio < 1.5
