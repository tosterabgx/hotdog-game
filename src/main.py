import cv2
import mediapipe as mp
import numpy as np

from recognizer import is_correct_gesture, is_mouth_open
from utilities import insert_image_in_hand

hotdog_image = cv2.imread("../assets/hotdog.png", cv2.IMREAD_UNCHANGED)
hotdog_image = cv2.cvtColor(hotdog_image, cv2.COLOR_BGRA2RGBA)

hands_detector = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
)
face_detector = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord("q") or not ret:
        break

    flipped = np.fliplr(frame)
    flipped_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

    faces = face_detector.process(flipped_rgb).multi_face_landmarks
    hands_result = hands_detector.process(flipped_rgb)
    hands = hands_result.multi_hand_landmarks

    if faces:
        face = faces[0]
        lm = face.landmark

        # for debug
        # cv2.circle(img=flipped_rgb, center=(int(lm[13].x * flipped_rgb.shape[1]), int(lm[13].y * flipped_rgb.shape[0])),
        #            radius=3, color=(0, 255, 0))
        # cv2.circle(img=flipped_rgb, center=(int(lm[14].x * flipped_rgb.shape[1]), int(lm[14].y * flipped_rgb.shape[0])),
        #            radius=3, color=(255, 0, 0))

        if is_mouth_open(face):
            cv2.putText(flipped_rgb,
                        "Mouth open",
                        (10, 100),
                        cv2.FONT_HERSHEY_DUPLEX,
                        2,
                        (125, 246, 55),
                        2)

    if hands:
        for hand, hand_info in zip(hands, hands_result.multi_handedness):
            is_left = "Left" in str(hand_info)

            # mp.solutions.drawing_utils.draw_landmarks(flipped_rgb, hand, mp.solutions.hands.HAND_CONNECTIONS)

            if is_correct_gesture(hand, is_left):
                flipped_rgb = insert_image_in_hand(flipped_rgb, hotdog_image, hand, is_left)

    res_image = cv2.cvtColor(flipped_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)

hands_detector.close()
