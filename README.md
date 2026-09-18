# Sign Language Detection
 
A machine learning based Sign Language Detection system for recognizing English alphabet signs and selected sign language words using image upload and real-time webcam detection.
 
## Features
 
- Upload Image - Detect alphabet signs from an image
 - Alphabet Webcam - Real-time A-Z detection
 - Word Detection - HELLO, WATER, HELP, NO, YES
 - Hand landmark based recognition
 - MobileNetV2 image classification
 - Random Forest alphabet classification
 - LSTM word sequence recognition
 - Tkinter GUI
 
## Models
 
### Image Alphabet Model
MobileNetV2 Transfer Learning 
 Input: 224 x 224 
 Classes: A-Z 
 Test Accuracy: 85.17%
Model: models/sign_language_model.keras
 
### Webcam Alphabet Model
MediaPipe Hands + Random Forest 
 21 landmarks = 63 features 
 Classes: A-Z
Model: models/landmark_AZ_model.joblib
 
### Word Recognition Model
MediaPipe Hands + LSTM 
 30 frames per sequence 
 Words: HELLO, WATER, HELP, NO, YES
Model: models/word_sequence_model.keras
 
## Requirements
 
- Windows 10/11
 - Python 3.11.x
 - Webcam
 - Git
 - Internet connection
 
## Installation on Another System
 
### 1. Clone the Repository
`powershell
git clone https://github.com/gokulnathsuresh05/Sign_Language_Detection.git
cd Sign_Language_Detection
`
 
### 2. Create Virtual Environment
`powershell
py -3.11 -m venv .venv
`
 
### 3. Activate Virtual Environment
`powershell
.\.venv\Scripts\Activate.ps1
`
If activation is blocked, run Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass and activate again.
 
### 4. Install Packages
`powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
`
 
### 5. Run the Application
`powershell
python app.py
`
 
## GUI Usage
 
**Upload Image:** Select a hand sign image and detect the alphabet.
 
**Alphabet Webcam:** Detect A-Z signs in real time.
 
**Word Detection:** Recognize HELLO, WATER, HELP, NO and YES using webcam sequences.
 
**RESET:** Reset the GUI.
 
## Time Restriction
 
The application works from **6:00 PM to before 10:00 PM**.
 
Outside this period all buttons are blocked and the application displays:
Time is not valid. Try again between 6 PM - 10 PM.
 
## Dataset
 
The training dataset can be downloaded from Google Drive:
 
https://drive.google.com/file/d/1Wlsfo2FrilHzl3UsWO1gcnl7kZc1pDFo/view?usp=sharing
 
The dataset is not included in GitHub because of its size. The trained models are already included in the models folder, so downloading the dataset is not required for normal usage.
 
## Dataset Information
 
- Alphabet dataset: 10,873 images, A-Z classes
 - Landmark alphabet dataset: 7,800 samples, 26 classes
 - Word dataset: 500 valid sequences, 100 per word, 30 frames per sequence
 
## Training
 
Training scripts are included: 	rain.py, 	rain_landmark_AZ.py, 	rain_word_model.py.
 
Pre-trained models are included, so training is not required for normal usage.
 
## Technologies
 
Python, TensorFlow, Keras, OpenCV, MediaPipe, Scikit-learn, NumPy, Pandas, Joblib, Tkinter, MobileNetV2, Random Forest, LSTM
 
## Important Notes
 
1. Use Python 3.11.x for compatibility.
2. Webcam is required for real-time detection.
3. Install dependencies using equirements.txt.
4. Models are already included in models.
5. Dataset download is required only for retraining or development.
 
## Troubleshooting
 
If webcam does not open, check camera connection and Windows camera permissions.
 
If MediaPipe errors occur, reinstall dependencies using pip install -r requirements.txt.
 
## GitHub
 
https://github.com/gokulnathsuresh05/Sign_Language_Detection
 
## Author
 
**Gokulnath Suresh**

