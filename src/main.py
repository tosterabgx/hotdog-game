import cv2
import mediapipe as mp
import numpy as np

from recognizer import is_correct_gesture, is_mouth_open

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

    face = face_detector.process(flipped_rgb).multi_face_landmarks
    hands = hands_detector.process(flipped_rgb).multi_hand_landmarks

    if face:
        face = face[0]
        lm = face.landmark

        # debug output
        cv2.circle(img=flipped_rgb, center=(int(lm[13].x * flipped_rgb.shape[1]), int(lm[13].y * flipped_rgb.shape[0])),
                   radius=3, color=(0, 255, 0))
        cv2.circle(img=flipped_rgb, center=(int(lm[14].x * flipped_rgb.shape[1]), int(lm[14].y * flipped_rgb.shape[0])),
                   radius=3, color=(255, 0, 0))

        if is_mouth_open(face):
            cv2.putText(flipped_rgb,
                        "Mouth open",
                        (10, 100),
                        cv2.FONT_HERSHEY_DUPLEX,
                        2,
                        (125, 246, 55),
                        2)

    if hands:
        for hand in hands:
            mp.solutions.drawing_utils.draw_landmarks(flipped_rgb, hand, mp.solutions.hands.HAND_CONNECTIONS)

            if is_correct_gesture(hand):
                lm = hand.landmark
                text_position = (int(lm[0].x * flipped_rgb.shape[1]), int(lm[0].y * flipped_rgb.shape[0]))

                cv2.putText(
                    flipped_rgb,
                    "Correct",
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

    res_image = cv2.cvtColor(flipped_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)

hands_detector.close()
