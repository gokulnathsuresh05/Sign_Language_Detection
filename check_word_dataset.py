from pathlib import Path
import numpy as np

ROOT = Path.cwd() / "word_landmark_data"
WORDS = ["hello", "water", "help", "no", "yes"]

total = 0
failed = 0

print("=" * 60)
print("WORD DATASET VALIDATION")
print("=" * 60)

for word in WORDS:
    files = sorted((ROOT / word).glob("*.npy"))
    valid = 0

    for f in files:
        try:
            x = np.load(f)

            if x.shape == (30, 63) and np.isfinite(x).all():
                valid += 1
            else:
                failed += 1
                print(f"INVALID: {f.name} -> {x.shape}")
        except Exception as e:
            failed += 1
            print(f"ERROR: {f.name} -> {e}")

    total += valid
    print(f"{word.upper():10s}: {valid:3d} / 100 valid")

print("=" * 60)
print(f"TOTAL VALID SEQUENCES : {total}")
print(f"FAILED SEQUENCES      : {failed}")

if total == 500 and failed == 0:
    print("DATASET CHECK PASSED")
else:
    print("DATASET CHECK FAILED")
print("=" * 60)
