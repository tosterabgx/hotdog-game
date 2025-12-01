import cv2
import mediapipe as mp
import numpy as np

from recognizer import is_correct_gesture, is_mouth_open
from utilities import insert_image_in_hand, find_circle

hotdog_image = cv2.imread("../assets/hotdog.png", cv2.IMREAD_UNCHANGED)
hotdog_image = cv2.cvtColor(hotdog_image, cv2.COLOR_BGRA2RGBA)

hands_detector = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
)
face_detector = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

open_mouth_confidence = 0

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
            (x, y), r = find_circle((lm[0], lm[17]), flipped_rgb.shape, 1.5)

            cv2.circle(img=flipped_rgb, center=(int(x), int(y)), radius=int(r), color=(255, 0, 0))

    if hands:
        for hand, hand_info in zip(hands, hands_info):
            is_left = "Left" in str(hand_info)

            if is_correct_gesture(hand, is_left):
                flipped_rgb = insert_image_in_hand(flipped_rgb, hotdog_image, hand, is_left)

                (x, y), r = find_circle(hand.landmark, flipped_rgb.shape, 1.5)
                cv2.circle(img=flipped_rgb, center=(int(x), int(y)), radius=int(r), color=(255, 0, 0))

    res_image = cv2.cvtColor(flipped_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)

hands_detector.close()
