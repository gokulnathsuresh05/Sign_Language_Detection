import cv2
import mediapipe as mp
import numpy as np
import csv
from pathlib import Path

ROOT = Path.cwd()
DATA_DIR = ROOT / "landmark_data"
DATA_DIR.mkdir(exist_ok=True)

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SAMPLES_PER_CLASS = 300

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    raise SystemExit

print("=" * 70)
print("SIGN LANGUAGE A-Z LANDMARK DATA COLLECTION")
print("=" * 70)
print("Each alphabet = 300 samples")
print("Show the SAME hand sign clearly.")
print("SPACE = start/collect current alphabet")
print("N = next alphabet")
print("Q = quit")
print("=" * 70)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    for class_index, label in enumerate(CLASSES):

        csv_path = DATA_DIR / f"{label}.csv"

        existing = 0

        if csv_path.exists():
            with open(csv_path, "r", newline="") as f:
                existing = sum(1 for _ in f)

        collecting = False

        print(f"\n[{class_index+1}/26] LETTER: {label}")
        print(f"Existing samples: {existing}")

        while existing < SAMPLES_PER_CLASS:

            ret, frame = cap.read()

            if not ret:
                print("ERROR: Could not read webcam.")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            landmarks = None

            if result.multi_hand_landmarks:

                hand = result.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                coords = []

                # Wrist-relative + scale-normalized coordinates
                wrist = hand.landmark[0]

                for lm in hand.landmark:
                    coords.extend([
                        lm.x - wrist.x,
                        lm.y - wrist.y,
                        lm.z - wrist.z
                    ])

                coords = np.array(coords, dtype=np.float32)

                scale = np.max(np.abs(coords))

                if scale > 0:
                    coords = coords / scale

                landmarks = coords.tolist()

                cv2.putText(
                    frame,
                    "HAND DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "SHOW ONE HAND",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.putText(
                frame,
                f"LETTER: {label}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 255, 0),
                3
            )

            cv2.putText(
                frame,
                f"SAMPLES: {existing}/{SAMPLES_PER_CLASS}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "SPACE = COLLECT | N = NEXT | Q = QUIT",
                (20, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("A-Z Landmark Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print("\nCollection stopped.")
                raise SystemExit

            if key == ord("n"):
                print(f"Skipping {label}")
                break

            if key == 32:
                collecting = True

            if collecting and landmarks is not None:

                with open(
                    csv_path,
                    "a",
                    newline=""
                ) as f:

                    writer = csv.writer(f)

                    if existing == 0:
                        writer.writerow(
                            ["label"] +
                            [f"f{i}" for i in range(63)]
                        )

                    writer.writerow([label] + landmarks)

                existing += 1

                # Small delay between samples
                cv2.waitKey(20)

        if existing >= SAMPLES_PER_CLASS:
            print(f"{label}: {existing} samples completed.")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 70)
print("A-Z LANDMARK DATA COLLECTION COMPLETED")
print("=" * 70)
print(f"Dataset location: {DATA_DIR}")
