from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
import json
from collections import deque
import requests
import os

# FIXED: Set template and static folder to current directory
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Configuration
PERPLEXITY_API_KEY = "YOUR_AI_API_Key"
API_URL = "https://api.perplexity.ai/chat/completions"

# Global variables
model = None
word_mapping = None
mp_hands = None
landmark_buffer = deque(maxlen=30)

class PerplexityGrammarCorrector:
    """Grammar correction using Perplexity API"""
    def __init__(self, api_key=PERPLEXITY_API_KEY, model_name="Your AI MODEL"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = API_URL
        self.available = bool(api_key) and api_key != "YOUR_PERPLEXITY_API_KEY_HERE"

    def correct_sentence(self, words):
        """Correct grammar from list of words"""
        if not self.available:
            return " ".join(words).capitalize() + "."

        joined_words = " ".join(words)
        prompt = f"""Form a grammatically correct English sentence using these ISL words: {joined_words}
Requirements:
- Only use the provided words
- Make it grammatically correct
- Add necessary articles, prepositions if needed
- Keep it natural
- Return only the sentence"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 60,
            "temperature": 0.2
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                corrected = result["choices"][0]["message"]["content"].strip()
                if corrected and not corrected.endswith(('.', '!', '?')):
                    corrected += "."
                return corrected
            else:
                return " ".join(words).capitalize() + "."
        except Exception as e:
            print(f"API Error: {e}")
            return " ".join(words).capitalize() + "."

def extract_landmarks(frame):
    """Extract hand landmarks from frame using MediaPipe"""
    if mp_hands is None:
        return None, []

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb_frame)
        h, w, c = frame.shape
        landmarks_list = []
        landmarks_coords = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    # Normalized coordinates
                    landmarks_list.extend([landmark.x, landmark.y, landmark.z])
                    # Pixel coordinates for drawing on frontend
                    x_pixel = int(landmark.x * w)
                    y_pixel = int(landmark.y * h)
                    confidence = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                    landmarks_coords.append({
                        "index": idx,
                        "x": x_pixel,
                        "y": y_pixel,
                        "z": float(landmark.z),
                        "confidence": float(confidence)
                    })

            # Pad to 126 (2 hands * 21 landmarks * 3 coords)
            while len(landmarks_list) < 126:
                landmarks_list.append(0.0)

            return np.array(landmarks_list[:126], dtype=np.float32), landmarks_coords

        return None, []
    except Exception as e:
        print(f"Error extracting landmarks: {e}")
        return None, []

def normalize_landmarks(landmarks):
    """Normalize landmarks relative to wrist"""
    try:
        landmarks = landmarks.reshape(-1, 3)
        for hand_idx in range(2):
            start_idx = hand_idx * 21
            end_idx = start_idx + 21
            hand_landmarks = landmarks[start_idx:end_idx]
            if np.any(hand_landmarks):
                wrist = hand_landmarks[0]
                hand_landmarks -= wrist
                distances = np.linalg.norm(hand_landmarks, axis=1)
                max_distance = np.max(distances)
                if max_distance > 0:
                    hand_landmarks /= max_distance
                landmarks[start_idx:end_idx] = hand_landmarks
        return landmarks.flatten()
    except Exception as e:
        print(f"Error normalizing landmarks: {e}")
        return landmarks.flatten()

def predict_word(confidence_threshold=0.5):
    """Predict word from full buffer (30 frames)"""
    global landmark_buffer
    try:
        if len(landmark_buffer) < 30:
            return None, 0.0

        sequence = np.array(list(landmark_buffer))
        sequence_input = sequence.reshape(1, 30, 126)
        predictions = model.predict(sequence_input, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        idx_to_word = {v: k for k, v in word_mapping.items()}
        word = idx_to_word.get(int(predicted_class), "Unknown")

        return word, float(confidence)
    except Exception as e:
        print(f"Error predicting word: {e}")
        return "Error", 0.0

# Routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'mediapipe_loaded': mp_hands is not None,
        'api_available': bool(PERPLEXITY_API_KEY) and PERPLEXITY_API_KEY != "YOUR_PERPLEXITY_API_KEY_HERE"
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict word from image with landmarks"""
    global landmark_buffer
    try:
        # Check if image is in request
        if 'image' not in request.files:
            print("❌ No image in request")
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'error': 'No file selected'}), 400

        # Read and decode image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("❌ Could not decode image")
            return jsonify({
                'word': 'Invalid image',
                'hand_detected': False,
                'landmarks': []
            })

        print(f"✓ Image received: {frame.shape}")

        # Extract landmarks AND coordinates
        landmarks, landmarks_coords = extract_landmarks(frame)

        if landmarks is None:
            print("❌ No hand detected in image")
            return jsonify({
                'word': 'No hand',
                'confidence': 0.0,
                'hand_detected': False,
                'buffer_full': False,
                'buffer_progress': len(landmark_buffer) / 30.0,
                'landmarks': []
            })

        print(f"✓ Hand detected! Landmarks: {len(landmarks_coords)}")

        # Normalize and add to buffer
        normalized = normalize_landmarks(landmarks)
        landmark_buffer.append(normalized)
        buffer_progress = len(landmark_buffer) / 30.0

        print(f"✓ Buffer: {len(landmark_buffer)}/30 frames")

        # Check if buffer is full
        if len(landmark_buffer) == 30:
            print("✓ Buffer full! Predicting...")
            word, confidence = predict_word(0.5)
            print(f"✓ Prediction: {word} ({confidence:.2%})")

            # Clear buffer after prediction
            landmark_buffer.clear()

            return jsonify({
                'word': word if word != "Unknown" else "No match",
                'confidence': float(confidence),
                'hand_detected': True,
                'buffer_full': True,
                'buffer_progress': 1.0,
                'landmarks': landmarks_coords
            })
        else:
            # Still buffering
            return jsonify({
                'word': '',
                'confidence': 0.0,
                'hand_detected': True,
                'buffer_full': False,
                'buffer_progress': buffer_progress,
                'landmarks': landmarks_coords
            })

    except Exception as e:
        print(f"❌ Error in predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/sentence', methods=['POST'])
def sentence():
    """Form sentence from words"""
    try:
        data = request.json
        words = data.get('words', [])
        if not words:
            return jsonify({'error': 'No words provided'}), 400

        print(f"✓ Forming sentence from: {words}")
        corrector = PerplexityGrammarCorrector()
        corrected_sentence = corrector.correct_sentence(words)
        print(f"✓ Corrected: {corrected_sentence}")

        return jsonify({
            'original': ' '.join(words),
            'corrected': corrected_sentence
        })
    except Exception as e:
        print(f"❌ Error in sentence: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset buffers"""
    global landmark_buffer
    landmark_buffer.clear()
    print("✓ Buffer reset")
    return jsonify({'status': 'reset'})

@app.route('/api/words', methods=['GET'])
def words_list():
    """Get list of recognized words"""
    if word_mapping:
        return jsonify({'words': list(word_mapping.keys())})
    return jsonify({'words': []})

def initialize_models():
    """Load models on startup"""
    global model, word_mapping, mp_hands
    try:
        print("\n" + "="*70)
        print("🚀 LOADING ISL MODELS...")
        print("="*70)

        # Check if model files exist
        if not os.path.exists('models/best_model.h5'):
            print("❌ ERROR: models/best_model.h5 not found!")
            return False

        if not os.path.exists('models/word_mapping.json'):
            print("❌ ERROR: models/word_mapping.json not found!")
            return False

        print("📦 Loading Keras model...")
        model = keras.models.load_model('models/best_model.h5', compile=False)
        print("✓ Model loaded")

        print("📋 Loading word mapping...")
        with open('models/word_mapping.json', 'r') as f:
            word_mapping = json.load(f)
        print(f"✓ Words loaded: {list(word_mapping.keys())[:5]}... ({len(word_mapping)} total)")

        print("👐 Initializing MediaPipe...")
        mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✓ MediaPipe initialized")

        print("\n✅ ALL MODELS LOADED SUCCESSFULLY!")
        print("="*70)
        print("🌐 Starting web server at http://localhost:5500")
        print("="*70 + "\n")
        return True

    except Exception as e:
        print(f"\n❌ ERROR loading models: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure:")
        print(" - models/best_model.h5 exists")
        print(" - models/word_mapping.json exists")
        print(" - TensorFlow and MediaPipe are installed")
        print("="*70 + "\n")
        return False

if __name__ == '__main__':
    if initialize_models():
        port = int(os.environ.get("PORT", 10000))
        print("\n🚀 Server starting...")
        print("📱 Open http://localhost:5500 in your browser\n")
        app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
    else:
        print("❌ Failed to initialize models. Server not started.")
