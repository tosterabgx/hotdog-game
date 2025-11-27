import cv2
import mediapipe as mp
import numpy as np

from gesturer import is_correct_gesture

handsDetector = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord("q") or not ret:
        break

    flipped = np.fliplr(frame)
    flipped_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flipped_rgb)

    if results.multi_hand_landmarks is not None:
        for hand in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(flipped_rgb, hand, mp.solutions.hands.HAND_CONNECTIONS)

            if is_correct_gesture(hand):
                lm = hand.landmark
                text_position = (int(lm[0].x * flipped_rgb.shape[0]), int(lm[0].y * flipped_rgb.shape[1] - 150))

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

handsDetector.close()
