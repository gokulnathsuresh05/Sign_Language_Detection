import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import json
import numpy as np
from pathlib import Path
from collections import deque, Counter
import tensorflow as tf
import joblib
import mediapipe as mp

# ============================================================
# PATHS
# ============================================================

ROOT = Path.cwd()
MODEL_DIR = ROOT / "models"

IMAGE_MODEL_PATH = MODEL_DIR / "sign_language_model.keras"
IMAGE_LABELS_PATH = MODEL_DIR / "labels.json"

AZ_MODEL_PATH = MODEL_DIR / "landmark_AZ_model.joblib"

WORD_MODEL_PATH = MODEL_DIR / "word_sequence_model.keras"
WORD_LABELS_PATH = MODEL_DIR / "word_labels.npy"



IMG_SIZE = (224, 224)

# ============================================================
# MODEL + RUNTIME INITIALIZATION
# ============================================================

print("Loading models...")

# Image model
image_model = tf.keras.models.load_model(
    IMAGE_MODEL_PATH
)

with open(IMAGE_LABELS_PATH, "r", encoding="utf-8") as f:
    image_labels = json.load(f)

# Handle labels.json stored as list or dictionary
if isinstance(image_labels, dict):
    image_labels = list(image_labels.values())

# A-Z landmark model
az_loaded = joblib.load(AZ_MODEL_PATH)

if isinstance(az_loaded, dict):
    if "model" in az_loaded:
        az_model = az_loaded["model"]
    elif "classifier" in az_loaded:
        az_model = az_loaded["classifier"]
    else:
        az_model = next(
            (v for v in az_loaded.values() if hasattr(v, "predict")),
            az_loaded
        )
else:
    az_model = az_loaded

# Word sequence model
word_model = tf.keras.models.load_model(
    WORD_MODEL_PATH
)

word_labels = np.load(
    WORD_LABELS_PATH,
    allow_pickle=True
)

word_labels = [str(x).upper() for x in word_labels]

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Runtime variables
cap = None
running = False
mode = None

letter_history = deque(maxlen=8)
word_history = deque(maxlen=7)
word_sequence = deque(maxlen=30)

current_photo = None

print("All models loaded successfully.")


# ============================================================
# MODEL + RUNTIME INITIALIZATION
# ============================================================

print("Loading models...")

# Image model
image_model = tf.keras.models.load_model(
    IMAGE_MODEL_PATH
)

with open(IMAGE_LABELS_PATH, "r", encoding="utf-8") as f:
    image_labels = json.load(f)

# Handle labels.json stored as list or dictionary
if isinstance(image_labels, dict):
    image_labels = list(image_labels.values())

# A-Z landmark model
az_loaded = joblib.load(AZ_MODEL_PATH)

if isinstance(az_loaded, dict):
    if "model" in az_loaded:
        az_model = az_loaded["model"]
    elif "classifier" in az_loaded:
        az_model = az_loaded["classifier"]
    else:
        az_model = next(
            (v for v in az_loaded.values() if hasattr(v, "predict")),
            az_loaded
        )
else:
    az_model = az_loaded

# Word sequence model
word_model = tf.keras.models.load_model(
    WORD_MODEL_PATH
)

word_labels = np.load(
    WORD_LABELS_PATH,
    allow_pickle=True
)

word_labels = [str(x).upper() for x in word_labels]

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Runtime variables
cap = None
running = False
mode = None

letter_history = deque(maxlen=8)
word_history = deque(maxlen=7)
word_sequence = deque(maxlen=30)

current_photo = None

print("All models loaded successfully.")


# ============================================================
# TIME RESTRICTION
# ============================================================

def allowed_time():
    return True

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

# ============================================================
# TIME CHECK
# ============================================================

def check_time():
    # Timing restriction disabled for testing.
    return True


# ============================================================
# STATUS
# ============================================================

def set_status(text):

    status_label.config(text=text)

# ============================================================
# STOP CAMERA
# ============================================================

def stop_camera():

    global cap, running, mode

    running = False
    mode = None

    if cap is not None:
        cap.release()
        cap = None

    word_sequence.clear()
    word_history.clear()
    letter_history.clear()

# ============================================================
# START ALPHABET WEBCAM
# ============================================================

def start_alphabet():

    global cap, running, mode

    if not check_time():
        return

    # Reset previous mode before starting Alphabet
    stop_camera()

    prediction_label.config(text="Prediction: -")
    confidence_label.config(text="Confidence: -")

    display_label.config(
        image="",
        text="Starting Alphabet Webcam..."
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        messagebox.showerror(
            "Webcam Error",
            "Webcam open aagala."
        )

        return

    mode = "alphabet"
    running = True

    set_status("Alphabet Webcam - Ready")

    update_camera()

# ============================================================
# START WORD DETECTION
# ============================================================

def start_word():

    global cap, running, mode

    if not check_time():
        return

    # Reset previous mode before starting Word Detection
    stop_camera()

    prediction_label.config(text="Prediction: -")
    confidence_label.config(text="Confidence: -")

    display_label.config(
        image="",
        text="Starting Word Detection..."
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        messagebox.showerror(
            "Webcam Error",
            "Webcam open aagala."
        )

        return

    mode = "word"
    running = True

    set_status("Word Detection - Ready")

    update_camera()

# ============================================================
# UPLOAD IMAGE
# ============================================================

def upload_image():

    if not check_time():
        return

    # Stop and reset previous webcam mode
    stop_camera()

    prediction_label.config(text="Prediction: -")
    confidence_label.config(text="Confidence: -")

    display_label.config(
        image="",
        text="Select an image..."
    )

    file_path = filedialog.askopenfilename(
        title="Select Sign Language Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")
        ]
    )

    if not file_path:
        return

    try:

        # Display original image
        image = Image.open(file_path)
        image.thumbnail((700, 430))

        global current_photo

        current_photo = ImageTk.PhotoImage(image)

        display_label.config(
            image=current_photo,
            text=""
        )

        # Prediction
        img = tf.keras.utils.load_img(
            file_path,
            target_size=IMG_SIZE
        )

        arr = tf.keras.utils.img_to_array(img)

        # IMPORTANT:
        # Same preprocessing used during image model testing.
        arr = np.expand_dims(arr, axis=0)

        prediction = image_model.predict(
            arr,
            verbose=0
        )[0]

        class_id = int(np.argmax(prediction))
        confidence = float(prediction[class_id]) * 100

        predicted = image_labels[class_id]

        prediction_label.config(
            text=f"LETTER: {predicted}"
        )

        confidence_label.config(
            text=f"CONFIDENCE: {confidence:.1f}%"
        )

        set_status("Image Prediction Complete")

    except Exception as e:

        messagebox.showerror(
            "Image Error",
            str(e)
        )

# ============================================================
# CAMERA UPDATE
# ============================================================

def update_camera():

    global running, cap

    if not running or cap is None:
        return

    ret, frame = cap.read()

    if not ret:

        set_status("Unable to read webcam")
        root.after(30, update_camera)
        return

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)

    prediction_text = ""
    confidence_text = ""

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        features = normalize_landmarks(hand)

        # ====================================================
        # ALPHABET MODE
        # ====================================================

        if mode == "alphabet":

            try:

                if hasattr(
                    az_model,
                    "predict_proba"
                ):

                    probabilities = az_model.predict_proba(
                        [features]
                    )[0]

                    class_id = int(
                        np.argmax(probabilities)
                    )

                    confidence = float(
                        probabilities[class_id]
                    )

                    predicted = az_model.classes_[class_id]

                else:

                    predicted = az_model.predict(
                        [features]
                    )[0]

                    confidence = 1.0

                letter_history.append(
                    str(predicted).upper()
                )

                stable_letter = Counter(
                    letter_history
                ).most_common(1)[0][0]

                prediction_text = (
                    f"LETTER: {stable_letter}"
                )

                confidence_text = (
                    f"CONFIDENCE: {confidence * 100:.1f}%"
                )

                prediction_label.config(
                    text=f"Prediction: {stable_letter}"
                )

                confidence_label.config(
                    text=f"Confidence: {confidence * 100:.1f}%"
                )

                set_status(
                    "Alphabet Webcam Detection"
                )

            except Exception as e:

                prediction_text = "LETTER: ERROR"
                confidence_text = str(e)

        # ====================================================
        # WORD MODE
        # ====================================================

        elif mode == "word":

            word_sequence.append(features)

            if len(word_sequence) == 30:

                sequence_array = np.expand_dims(
                    np.array(
                        word_sequence,
                        dtype=np.float32
                    ),
                    axis=0
                )

                prediction = word_model.predict(
                    sequence_array,
                    verbose=0
                )[0]

                class_id = int(
                    np.argmax(prediction)
                )

                confidence = float(
                    prediction[class_id]
                )

                predicted_word = word_labels[
                    class_id
                ]

                # Only accept reasonably confident predictions
                if confidence >= 0.60:

                    word_history.append(
                        predicted_word
                    )

                if word_history:

                    stable_word = Counter(
                        word_history
                    ).most_common(1)[0][0]

                    prediction_text = (
                        f"WORD: {stable_word}"
                    )

                else:

                    prediction_text = (
                        "WORD: DETECTING..."
                    )

                confidence_text = (
                    f"CONFIDENCE: {confidence * 100:.1f}%"
                )

                if word_history:
                    prediction_label.config(
                        text=f"Prediction: {stable_word}"
                    )
                else:
                    prediction_label.config(
                        text="Prediction: Detecting..."
                    )

                confidence_label.config(
                    text=f"Confidence: {confidence * 100:.1f}%"
                )

                set_status(
                    "Word Detection"
                )

    else:

        letter_history.clear()

        if mode == "word":

            word_sequence.clear()
            word_history.clear()

        prediction_text = "HAND NOT DETECTED"
        confidence_text = ""

        set_status(
            "Place your hand inside the camera"
        )

    # ========================================================
    # DRAW PREDICTION ON CAMERA
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (760, 90),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        prediction_text,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        confidence_text,
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2
    )

    # Convert frame for Tkinter
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(frame_rgb)

    image = image.resize(
        (760, 430)
    )

    photo = ImageTk.PhotoImage(image)

    display_label.config(
        image=photo,
        text=""
    )

    display_label.image = photo

    root.after(
        15,
        update_camera
    )

# ============================================================
# RESET GUI
# ============================================================

def reset_gui():

    global current_photo

    # Stop webcam and clear all detection buffers
    stop_camera()

    # Clear displayed image
    current_photo = None

    display_label.config(
        image="",
        text="WEBCAM / IMAGE\n\nSelect an option below"
    )

    display_label.image = None

    # Clear prediction
    prediction_label.config(
        text="Prediction: -"
    )

    confidence_label.config(
        text="Confidence: -"
    )

    set_status(
        "Ready - Select an option"
    )

# ============================================================
# EXIT
# ============================================================


def exit_app():

    stop_camera()

    hands.close()

    root.destroy()

# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "Sign Language Detection"
)

root.geometry(
    "1000x720"
)

root.minsize(
    900,
    650
)

root.configure(
    bg="#f2f2f2"
)

# ============================================================
# TITLE
# ============================================================

title_label = tk.Label(
    root,
    text="SIGN LANGUAGE DETECTION",
    font=("Arial", 24, "bold"),
    bg="#f2f2f2"
)

title_label.pack(
    pady=(18, 5)
)

subtitle_label = tk.Label(
    root,
    text="Alphabet and Word Recognition System",
    font=("Arial", 11),
    bg="#f2f2f2"
)

subtitle_label.pack(
    pady=(0, 12)
)

# ============================================================
# MIDDLE DISPLAY AREA
# ============================================================

display_frame = tk.Frame(
    root,
    bg="black",
    width=760,
    height=430
)

display_frame.pack(
    pady=5
)

display_frame.pack_propagate(False)

display_label = tk.Label(
    display_frame,
    text="WEBCAM / IMAGE\n\nSelect an option below",
    font=("Arial", 18, "bold"),
    fg="white",
    bg="black",
    justify="center"
)

display_label.pack(
    expand=True,
    fill="both"
)

# ============================================================
# BUTTON AREA
# ============================================================

button_frame = tk.Frame(
    root,
    bg="#f2f2f2"
)

button_frame.pack(
    pady=15
)

alphabet_button = tk.Button(
    button_frame,
    text="Alphabet Webcam",
    command=start_alphabet,
    font=("Arial", 12, "bold"),
    width=18,
    height=2
)

alphabet_button.grid(
    row=0,
    column=0,
    padx=8
)

word_button = tk.Button(
    button_frame,
    text="Word Detection",
    command=start_word,
    font=("Arial", 12, "bold"),
    width=18,
    height=2
)

word_button.grid(
    row=0,
    column=1,
    padx=8
)

upload_button = tk.Button(
    button_frame,
    text="Upload Image",
    command=upload_image,
    font=("Arial", 12, "bold"),
    width=18,
    height=2
)

upload_button.grid(
    row=0,
    column=2,
    padx=8
)

reset_button = tk.Button(
    button_frame,
    text="RESET",
    command=reset_gui,
    font=("Arial", 12, "bold"),
    width=12,
    height=2
)

reset_button.grid(
    row=0,
    column=3,
    padx=8
)

# ============================================================
# RESULT
# ============================================================

prediction_label = tk.Label(
    root,
    text="Prediction: -",
    font=("Arial", 16, "bold"),
    bg="#f2f2f2"
)

prediction_label.pack(
    pady=(0, 3)
)

confidence_label = tk.Label(
    root,
    text="Confidence: -",
    font=("Arial", 12),
    bg="#f2f2f2"
)

confidence_label.pack()

status_label = tk.Label(
    root,
    text="Ready - Testing Mode",
    font=("Arial", 10),
    bg="#f2f2f2"
)

status_label.pack(
    pady=8
)

# ============================================================
# EXIT BUTTON
# ============================================================

exit_button = tk.Button(
    root,
    text="EXIT",
    command=exit_app,
    font=("Arial", 11, "bold"),
    width=10
)

exit_button.pack(
    pady=(0, 10)
)

root.protocol(
    "WM_DELETE_WINDOW",
    exit_app
)

root.mainloop()

