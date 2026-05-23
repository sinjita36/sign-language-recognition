import cv2
import mediapipe as mp

hands_module = mp.solutions.hands
draw = mp.solutions.drawing_utils

hands = hands_module.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

cap = cv2.VideoCapture(0)


def detect_gesture(landmarks):
    thumb = 1 if landmarks[4].x < landmarks[3].x else 0
    index = 1 if landmarks[8].y < landmarks[6].y else 0
    middle = 1 if landmarks[12].y < landmarks[10].y else 0
    ring = 1 if landmarks[16].y < landmarks[14].y else 0
    pinky = 1 if landmarks[20].y < landmarks[18].y else 0

    total = thumb + index + middle + ring + pinky

    if total == 0:
        return "FIST"

    elif total == 5:
        return "OPEN PALM"

    elif index and not middle and not ring and not pinky:
        return "ONE"

    elif index and middle and not ring and not pinky:
        return "PEACE"

    elif index and middle and ring and not pinky:
        return "THREE"

    elif index and middle and ring and pinky and not thumb:
        return "FOUR"

    elif thumb and not index and not middle and not ring and not pinky:
        return "THUMBS UP"

    elif total == 2:
        return "TWO"

    else:
        return "UNKNOWN"


while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=30)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    gesture = "No Hand"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            draw.draw_landmarks(
                frame,
                hand_landmarks,
                hands_module.HAND_CONNECTIONS
            )

            gesture = detect_gesture(hand_landmarks.landmark)

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()