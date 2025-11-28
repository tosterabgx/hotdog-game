import numpy as np
import cv2


def get_points(landmark, shape):
    points = []
    for mark in landmark:
        points.append([mark.x * shape[1], mark.y * shape[0]])
    return np.array(points, dtype=np.int32)


def insert_image_in_hand(bg, img, hand):
    img = img.copy()
    (x, y), r = cv2.minEnclosingCircle(get_points(hand.landmark, bg.shape))

    dd = int(max(1, int(r * 2)) * 0.9)
    img = cv2.resize(img, (int(img.shape[1] / img.shape[0] * dd), dd))

    x -= img.shape[1] / 2
    y -= img.shape[0] / 2

    return insert_image(bg, img, int(x), int(y))


def insert_image(bg, img, x, y):
    bg = bg.copy()

    bh, bw = bg.shape[:2]
    ih, iw = img.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + iw, bw), min(y + ih, bh)

    if x1 >= x2 or y1 >= y2:
        return bg

    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    overlay = img[oy1:oy2, ox1:ox2]
    roi = bg[y1:y2, x1:x2]

    mask = overlay[:, :, 3] > 0
    roi[mask] = overlay[:, :, :3][mask]

    bg[y1:y2, x1:x2] = roi
    return bg
