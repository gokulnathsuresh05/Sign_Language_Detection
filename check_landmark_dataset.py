import pandas as pd
from pathlib import Path

DATA_DIR = Path("landmark_data")
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

print("=" * 65)
print("A-Z LANDMARK DATASET VALIDATION")
print("=" * 65)

total = 0
all_ok = True

for label in CLASSES:

    file = DATA_DIR / f"{label}.csv"

    if not file.exists():
        print(f"{label}: FILE MISSING")
        all_ok = False
        continue

    df = pd.read_csv(file)

    rows = len(df)
    features = len(df.columns) - 1

    status = "OK"

    if rows != 300:
        status = f"WRONG SAMPLE COUNT ({rows})"
        all_ok = False

    if features != 63:
        status = f"WRONG FEATURES ({features})"
        all_ok = False

    if df.isnull().values.any():
        status = "NULL VALUES FOUND"
        all_ok = False

    total += rows

    print(
        f"{label}: {rows:3d} samples | "
        f"{features:2d} features | {status}"
    )

print()
print("=" * 65)
print(f"TOTAL SAMPLES: {total}")
print("=" * 65)

if total == 7800 and all_ok:
    print("DATASET CHECK PASSED")
    print("26 classes x 300 samples = 7800 samples")
    print("63 landmark features per sample")
    print("READY FOR A-Z MODEL TRAINING")
else:
    print("DATASET CHECK FAILED")
    print("Please check the reported class.")

print("=" * 65)
