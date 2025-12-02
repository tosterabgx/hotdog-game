import cv2
import mediapipe as mp
import numpy as np

from pathlib import Path

from recognizer import is_correct_gesture, is_mouth_open
from utilities import insert_image_in_hand, find_circle

hotdog_path = Path(__file__).parent.parent / "assets" / "hotdog.png"

hotdog_image = cv2.imread(str(hotdog_path), cv2.IMREAD_UNCHANGED)
hotdog_image = cv2.cvtColor(hotdog_image, cv2.COLOR_BGRA2RGBA)

hands_detector = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
)
face_detector = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

open_mouth_confidence = 0

cooldown = [False, False]
score = 0

while cap.isOpened():
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord("q") or not ret:
        break

    flipped = np.fliplr(frame)
    flipped_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

    faces = face_detector.process(flipped_rgb).multi_face_landmarks
    hands = hands_detector.process(flipped_rgb)
    hands, hands_info = hands.multi_hand_landmarks, hands.multi_handedness

    if faces:
        face = faces[0]
        lm = face.landmark

        if is_mouth_open(face):
            open_mouth_confidence = min(100, open_mouth_confidence + 5)
        else:
            open_mouth_confidence = max(0, open_mouth_confidence - 5)

        if open_mouth_confidence > 50:
            (x_mouth, y_mouth), r_mouth = find_circle((lm[0], lm[17]), flipped_rgb.shape, 2)

            cv2.circle(img=flipped_rgb, center=(int(x_mouth), int(y_mouth)), radius=int(r_mouth), color=(255, 0, 0))
    else:
        open_mouth_confidence = max(0, open_mouth_confidence - 5)

    if hands:
        for hand, hand_info in zip(hands, hands_info):
            is_left = "Left" in str(hand_info)

            if is_left:
                i = 0
            else:
                i = 1

            if is_correct_gesture(hand, is_left):
                if not cooldown[i]:
                    flipped_rgb = insert_image_in_hand(flipped_rgb, hotdog_image, hand, is_left)

                (x, y), r = find_circle(hand.landmark, flipped_rgb.shape, 1.25)
                cv2.circle(img=flipped_rgb, center=(int(x), int(y)), radius=int(r), color=(255, 0, 0))

                if faces and open_mouth_confidence > 50:
                    hand_pos = np.array([x, y])
                    mouth_pos = np.array([x_mouth, y_mouth])

                    dd_hand_mouth = np.linalg.norm(hand_pos - mouth_pos)

                    if dd_hand_mouth < r + r_mouth:
                        if not cooldown[i]:
                            cooldown[i] = True

                            score += 1
                                
                            print(f"ate hotdog ({i})")
                    else:
                        cooldown[i] = False
                else:
                    cooldown[i] = False

    cv2.putText(flipped_rgb, f"I ate {score} hotdog(s)", (10, 50), cv2.QT_FONT_NORMAL, 2, (0, 0, 0), thickness=2)

    res_image = cv2.cvtColor(flipped_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)

hands_detector.close()
