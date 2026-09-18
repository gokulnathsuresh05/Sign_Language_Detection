import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

ROOT = Path.cwd()
DATA_DIR = ROOT / "landmark_data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

X = []
y = []

print("=" * 70)
print("FULL A-Z LANDMARK MODEL TRAINING")
print("=" * 70)

for label in CLASSES:

    file = DATA_DIR / f"{label}.csv"

    df = pd.read_csv(file)

    features = df.iloc[:, 1:].values.astype(np.float32)

    X.extend(features)
    y.extend([label] * len(features))

    print(f"{label}: {len(features)} samples")

X = np.array(X, dtype=np.float32)
y = np.array(y)

print()
print("Total samples :", len(X))
print("Features      :", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print()
print("Training Random Forest...")
print("Please wait...")

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print()
print("=" * 70)
print(f"TEST ACCURACY: {accuracy * 100:.2f}%")
print("=" * 70)

print()
print("CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_test, pred))

model_path = MODEL_DIR / "landmark_AZ_model.joblib"

joblib.dump(
    {
        "model": model,
        "classes": CLASSES
    },
    model_path
)

print()
print("MODEL SAVED:")
print(model_path)
print()
print("TRAINING COMPLETED")
