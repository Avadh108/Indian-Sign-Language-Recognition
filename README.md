# ANUVAAD - Indian-Sign-Language-Recognition
Word Level Indian Sign Language Recognition using CNN-LSTM along with sentence formation.
ANUVAAD is a Flask-based web application that translates Indian Sign Language gestures into English words and sentences using computer vision and deep learning.
The project combines hand-landmark detection, a CNN–LSTM model, and a grammar-correction pipeline to generate meaningful English output from ISL signs.​

**Features**
Real-time sign recognition from webcam or uploaded image sequence.

Deep learning model (CNN + LSTM) trained on custom ISL gesture dataset.​

Automatic sentence formation and grammar correction for predicted words.​

Web version with separate tabs for Live Camera and Upload Image.​

**Tech Stack**
Backend: Python, Flask, TensorFlow / Keras, MediaPipe, OpenCV.​

Frontend: HTML5, CSS3, JavaScript (fetch API), responsive design.​

Others: NumPy, JSON-based word mapping, custom ISL dataset stored as landmark arrays.​

**Project Structure**
app.py – Flask application, API endpoints for prediction and sentence generation.​

templates/index.html – Main web interface for live camera and uploads.​

static/script.js – Frontend logic for webcam capture, image upload, and API calls.​​

models/best_model.h5 – Trained CNN–LSTM model for ISL word prediction.

models/word_mapping.json – Mapping between class indices and ISL words.

**How It Works**
Data collection

ISL gestures are recorded via webcam as short video clips.

MediaPipe extracts 3D hand landmarks for each frame and saves them as sequences.​

Model training

Landmark sequences are used to train a CNN–LSTM model that classifies each sequence into an ISL word.​

Training logs and accuracy curves are stored in models/training_log.csv and models/training_history.png.

Inference pipeline (web app)

User signs using webcam or uploads a sequence of images.​

Backend extracts landmarks, fills a buffer of 30 frames, and runs the trained model to predict the word.​

Predicted words are appended to a sentence buffer and passed through the grammar-correction module to form natural English sentences.​

**Setup and Installation**
Clone the repository

bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Create environment and install dependencies

bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
(Requirements should include flask, tensorflow, mediapipe, opencv-python, numpy, etc.)​

Place model files

Copy best_model.h5 and word_mapping.json into the models/ folder if they are not already tracked in the repo.

Run the web

bash
python app.py
# or (if configured)
flask --app app run
Open http://127.0.0.1:5000 in the browser to use the web interface.​

**Implementation Steps (Summary)**
Designed ISL vocabulary and collected gesture videos for each word.

Extracted hand landmarks using MediaPipe and stored them as sequences.

Trained a CNN–LSTM model on the processed landmark dataset and evaluated accuracy.​

Built a Flask API that loads the trained model, receives frames, and returns predictions.​

Developed web UI with webcam integration and image upload support.​

Added sentence formation and grammar correction to convert predicted words into fluent English.​
