import cv2
import joblib
import numpy as np
import mediapipe as mp
from collections import Counter, deque
from pathlib import Path

ROOT = Path.cwd()
MODEL_PATH = ROOT / "models" / "landmark_AZ_model.joblib"

if not MODEL_PATH.exists():
    print("ERROR: landmark_AZ_model.joblib not found.")
    raise SystemExit

loaded = joblib.load(MODEL_PATH)

print("Model file type:", type(loaded))

# Handle dictionary-saved model
if isinstance(loaded, dict):

    print("Dictionary keys:", list(loaded.keys()))

    model = None

    for key in ["model", "classifier", "clf", "rf", "random_forest"]:
        if key in loaded:
            model = loaded[key]
            break

    if model is None:
        for value in loaded.values():
            if hasattr(value, "predict") and hasattr(value, "predict_proba"):
                model = value
                break

    if model is None:
        print("ERROR: Classifier not found inside model file.")
        raise SystemExit

else:
    model = loaded

print("Classifier:", type(model))

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def normalize_landmarks(hand_landmarks):

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )

    # Same preprocessing used during training
    points = points - points[0]

    scale = np.max(np.abs(points))

    if scale > 0:
        points = points / scale

    return points.flatten().reshape(1, -1)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    raise SystemExit

prediction_history = deque(maxlen=8)

print("=" * 60)
print("A-Z SIGN LANGUAGE LIVE TEST")
print("=" * 60)
print("GREEN LETTER = MODEL PREDICTION")
print("Show one alphabet sign at a time.")
print("Press Q to quit.")
print("=" * 60)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.60,
    min_tracking_confidence=0.60
) as hands:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Webcam frame error.")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]

            # Draw 21 landmarks
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            features = normalize_landmarks(hand)

            probabilities = model.predict_proba(features)[0]

            best_index = int(np.argmax(probabilities))

            raw_letter = str(model.classes_[best_index])

            confidence = float(probabilities[best_index]) * 100

            prediction_history.append(raw_letter)

            if len(prediction_history) >= 3:
                predicted_letter = Counter(
                    prediction_history
                ).most_common(1)[0][0]
            else:
                predicted_letter = raw_letter

            # GREEN prediction box
            cv2.rectangle(
                frame,
                (15, 15),
                (450, 115),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                f"LETTER: {predicted_letter}",
                (30, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.15,
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                f"CONFIDENCE: {confidence:.1f}%",
                (30, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        else:

            prediction_history.clear()

            cv2.rectangle(
                frame,
                (15, 15),
                (450, 70),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (30, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.putText(
            frame,
            "Show A-Z sign | Press Q to quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "A-Z Sign Language - Live Prediction",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print("Webcam test completed.")
