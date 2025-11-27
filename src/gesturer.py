import numpy as np


def is_correct_gesture(hand_landmarks):
    lm = hand_landmarks.landmark

    index_tip = np.array([lm[8].x, lm[8].y])
    thumb_tip = np.array([lm[4].x, lm[4].y])
    index_mcp = np.array([lm[6].x, lm[6].y])
    d_tip_tip = np.linalg.norm(index_tip - thumb_tip)
    d_thumb_crease = np.linalg.norm(index_mcp - thumb_tip)

    ratio = d_tip_tip / (d_thumb_crease + 1e-6)
    return ratio < 0.8
