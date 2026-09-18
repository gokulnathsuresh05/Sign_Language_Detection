import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import time

ROOT = Path.cwd()
DATA = ROOT / "word_landmark_data"

WORDS = {
    "h": "hello",
    "w": "water",
    "p": "help",
    "n": "no",
    "y": "yes"
}

SEQUENCE_LENGTH = 30
TARGET_SEQUENCES = 100

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

counts = {}
for word in WORDS.values():
    counts[word] = len(list((DATA / word).glob("*.npy")))

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam open aagala.")
    raise SystemExit

print("=" * 60)
print("WORD DATASET COLLECTION")
print("=" * 60)
print("H = HELLO")
print("W = WATER")
print("P = HELP")
print("N = NO")
print("Y = YES")
print("SPACE = Start one sequence")
print("Q = Quit")
print("=" * 60)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    current_word = None
    collecting = False
    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            features = normalize_landmarks(hand)

            if collecting:
                sequence.append(features)

                if len(sequence) >= SEQUENCE_LENGTH:
                    counts[current_word] += 1

                    filename = (
                        DATA / current_word /
                        f"{current_word}_{counts[current_word]:03d}.npy"
                    )

                    np.save(filename, np.array(sequence, dtype=np.float32))

                    print(
                        f"{current_word.upper()} -> "
                        f"{counts[current_word]}/{TARGET_SEQUENCES} saved"
                    )

                    collecting = False
                    sequence = []

        cv2.rectangle(frame, (0, 0), (700, 90), (0, 0, 0), -1)

        if current_word:
            status = f"WORD: {current_word.upper()}"
        else:
            status = "SELECT WORD: H/W/P/N/Y"

        cv2.putText(
            frame,
            status,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        if collecting:
            progress = f"Recording: {len(sequence)}/{SEQUENCE_LENGTH}"
        else:
            progress = "Press SPACE to record"

        cv2.putText(
            frame,
            progress,
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        y = 120
        for word in WORDS.values():
            text = f"{word.upper():10s}: {counts[word]}/{TARGET_SEQUENCES}"
            cv2.putText(
                frame,
                text,
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )
            y += 25

        cv2.imshow("Word Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key in [ord(k) for k in WORDS]:
            current_word = WORDS[chr(key)]
            sequence = []
            collecting = False
            print(f"\nSelected: {current_word.upper()}")

        elif key == 32:
            if current_word is None:
                print("First select a word using H/W/P/N/Y.")
            elif counts[current_word] >= TARGET_SEQUENCES:
                print(f"{current_word.upper()} already completed.")
            else:
                sequence = []
                collecting = True
                print(
                    f"Recording {current_word.upper()}... "
                    f"Perform the complete sign now."
                )

cap.release()
cv2.destroyAllWindows()

print("\nFINAL DATASET COUNTS")
for word in WORDS.values():
    print(f"{word.upper():10s}: {counts[word]}/{TARGET_SEQUENCES}")
