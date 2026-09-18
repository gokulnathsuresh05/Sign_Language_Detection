import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset" / "alphabet"

TRAIN_DIR = DATASET / "train"
VAL_DIR = DATASET / "val"
TEST_DIR = DATASET / "test"

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

print("=" * 60)
print("SIGN LANGUAGE DETECTION - 26 ALPHABET TRAINING")
print("=" * 60)

print("TensorFlow:", tf.__version__)

# -----------------------------
# LOAD DATASETS
# -----------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:", class_names)
print("Number of classes:", num_classes)

if num_classes != 26:
    raise ValueError(
        f"Expected 26 classes, but found {num_classes}: {class_names}"
    )

# Save labels
with open(MODEL_DIR / "labels.json", "w") as f:
    json.dump(class_names, f, indent=4)

# Performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# -----------------------------
# DATA AUGMENTATION
# -----------------------------

augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.10, 0.10),
])

# -----------------------------
# MOBILE NET V2
# -----------------------------

base = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,))

x = augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# -----------------------------
# PHASE 1
# -----------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n" + "=" * 60)
print("PHASE 1 - CLASSIFIER TRAINING")
print("=" * 60)

callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6
    ),

    ModelCheckpoint(
        MODEL_DIR / "sign_language_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks
)

# -----------------------------
# PHASE 2 - FINE TUNING
# -----------------------------

print("\n" + "=" * 60)
print("PHASE 2 - FINE TUNING")
print("=" * 60)

base.trainable = True

# Freeze earlier layers
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    callbacks=callbacks
)

# -----------------------------
# FINAL MODEL
# -----------------------------

model.save(
    MODEL_DIR / "sign_language_model.keras"
)

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print("\nModel:")
print(MODEL_DIR / "sign_language_model.keras")

print("\nLabels:")
print(MODEL_DIR / "labels.json")
