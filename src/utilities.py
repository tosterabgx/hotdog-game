def insert_image(background, image, x, y):
    bg = background.copy()

    bh, bw = bg.shape[:2]
    ih, iw = image.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + iw, bw), min(y + ih, bh)

    if x1 >= x2 or y1 >= y2:
        return bg

    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    overlay = image[oy1:oy2, ox1:ox2]
    roi = bg[y1:y2, x1:x2]

    mask = overlay[:, :, 3] > 0

    roi[mask] = overlay[:, :, :3][mask]

    bg[y1:y2, x1:x2] = roi
    return bg
