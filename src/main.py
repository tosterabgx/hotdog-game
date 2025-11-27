import cv2
import mediapipe as mp
import numpy as np

handsDetector = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord("q") or not ret:
        break

    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)

    if results.multi_hand_landmarks is not None:
        for hand in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                flippedRGB, hand
            )

    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)

handsDetector.close()
