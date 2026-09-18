import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, callbacks

ROOT = Path.cwd()
DATA = ROOT / "word_landmark_data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

WORDS = ["hello", "water", "help", "no", "yes"]
SEQ_LEN = 30
FEATURES = 63

print("=" * 60)
print("LOADING WORD DATASET")
print("=" * 60)

X = []
y = []

for word in WORDS:
    files = sorted((DATA / word).glob("*.npy"))

    for f in files:
        data = np.load(f)

        if data.shape != (SEQ_LEN, FEATURES):
            print(f"Skipping invalid file: {f}")
            continue

        X.append(data)
        y.append(word)

    print(f"{word.upper():10s}: {len(files)} sequences")

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"\nDataset shape: {X.shape}")
print(f"Total samples: {len(X)}")

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nClasses:")
for i, name in enumerate(encoder.classes_):
    print(f"{i}: {name.upper()}")

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

# Model
model = models.Sequential([
    layers.Input(shape=(SEQ_LEN, FEATURES)),

    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.30),

    layers.LSTM(64),
    layers.Dropout(0.30),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.20),

    layers.Dense(len(WORDS), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\n" + "=" * 60)
print("TRAINING WORD MODEL")
print("=" * 60)

early_stop = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=8,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=0.00001
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("\n" + "=" * 60)
print("MODEL TEST")
print("=" * 60)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")

# Save model
model_path = MODEL_DIR / "word_sequence_model.keras"
labels_path = MODEL_DIR / "word_labels.npy"

model.save(model_path)
np.save(labels_path, encoder.classes_)

print("\nModel saved:")
print(model_path)
print(labels_path)

# ---------------------------------------------------------
# LIVE WEBCAM
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("STARTING LIVE WORD DETECTION")
print("=" * 60)
print("Perform: HELLO / WATER / HELP / NO / YES")
print("Press Q to quit.")
print("=" * 60)

import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def normalize_landmarks(hand):
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand.landmark],
        dtype=np.float32
    )

    points = points - points[0]

    scale = np.max(np.abs(points))

    if scale > 0:
        points = points / scale

    return points.flatten()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam open aagala.")
    raise SystemExit

sequence = deque(maxlen=SEQ_LEN)
prediction_history = deque(maxlen=7)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        word_text = "DETECTING..."
        confidence_text = ""

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            features = normalize_landmarks(hand)
            sequence.append(features)

            if len(sequence) == SEQ_LEN:

                input_data = np.expand_dims(
                    np.array(sequence, dtype=np.float32),
                    axis=0
                )

                prediction = model.predict(
                    input_data,
                    verbose=0
                )[0]

                class_id = int(np.argmax(prediction))
                confidence = float(prediction[class_id])

                predicted_word = encoder.classes_[class_id]

                prediction_history.append(predicted_word)

                # Majority vote
                common_word, count = Counter(
                    prediction_history
                ).most_common(1)[0]

                word_text = common_word.upper()
                confidence_text = f"CONFIDENCE: {confidence * 100:.1f}%"

        else:
            sequence.clear()
            prediction_history.clear()
            word_text = "HAND NOT DETECTED"

        # Green word display
        cv2.rectangle(
            frame,
            (0, 0),
            (700, 105),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"WORD: {word_text}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            confidence_text,
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.imshow("Sign Language - Word Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print("\nWord detection finished.")
