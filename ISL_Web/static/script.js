const state = {
    wordBuffer: [],
    sentenceHistory: [],
    isProcessing: false,
    cameraActive: false,
    frameCount: 0,
    currentMode: 'camera',
    landmarks: []
};

const webcamEl = document.getElementById('webcam');
const canvasEl = document.getElementById('canvas');
const videoContainer = document.getElementById('videoContainer');
const uploadZone = document.getElementById('uploadZone');
const imageInput = document.getElementById('imageInput');
const recordingInd = document.getElementById('recordingInd');
const predictedWordEl = document.getElementById('predictedWord');
const confidenceFillEl = document.getElementById('confidenceFill');
const confidenceTextEl = document.getElementById('confidenceText');
const bufferStatusEl = document.getElementById('bufferStatus');
const bufferFillEl = document.getElementById('bufferFill');
const wordBufferEl = document.getElementById('wordBuffer');
const sentenceOutputEl = document.getElementById('sentenceOutput');
const historyGridEl = document.getElementById('historyGrid');
const startCameraBtn = document.getElementById('startCameraBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');
const formSentenceBtn = document.getElementById('formSentenceBtn');
const removeLastBtn = document.getElementById('removeLastBtn');

let drawCanvas = null;
let drawCtx = null;

const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20]
];

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    setupLandmarkCanvas();
});

function setupEventListeners() {
    uploadZone.addEventListener('click', () => imageInput.click());
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            imageInput.files = e.dataTransfer.files;
            handleImageUpload();
        }
    });
    imageInput.addEventListener('change', handleImageUpload);
}

function setupLandmarkCanvas() {
    drawCanvas = document.createElement('canvas');
    drawCanvas.id = 'landmarkCanvas';
    drawCanvas.style.position = 'absolute';
    drawCanvas.style.top = '0';
    drawCanvas.style.left = '0';
    drawCanvas.style.cursor = 'pointer';
    drawCanvas.style.background = 'transparent';
    videoContainer.appendChild(drawCanvas);
    drawCtx = drawCanvas.getContext('2d');
}

function drawHandLandmarks(landmarks) {
    if (!drawCanvas || !drawCtx || landmarks.length === 0) return;
    drawCanvas.width = videoContainer.offsetWidth;
    drawCanvas.height = videoContainer.offsetHeight;
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    const scaleX = drawCanvas.width / webcamEl.videoWidth;
    const scaleY = drawCanvas.height / webcamEl.videoHeight;

    drawCtx.strokeStyle = '#00D8FF';
    drawCtx.lineWidth = 2;
    for (const [start, end] of HAND_CONNECTIONS) {
        if (start < landmarks.length && end < landmarks.length) {
            const lmStart = landmarks[start];
            const lmEnd = landmarks[end];
            if (lmStart && lmEnd && lmStart.confidence > 0.3 && lmEnd.confidence > 0.3) {
                drawCtx.beginPath();
                drawCtx.moveTo(lmStart.x * scaleX, lmStart.y * scaleY);
                drawCtx.lineTo(lmEnd.x * scaleX, lmEnd.y * scaleY);
                drawCtx.stroke();
            }
        }
    }

    for (let i = 0; i < landmarks.length; i++) {
        const landmark = landmarks[i];
        if (!landmark || landmark.confidence < 0.3) continue;
        const x = landmark.x * scaleX;
        const y = landmark.y * scaleY;
        if (i === 0) {
            drawCtx.fillStyle = '#FF073A';
        } else if ([4, 8, 12, 16, 20].includes(i)) {
            drawCtx.fillStyle = '#00FF41';
        } else {
            drawCtx.fillStyle = '#00D8FF';
        }
        drawCtx.beginPath();
        drawCtx.arc(x, y, 5, 0, 2 * Math.PI);
        drawCtx.fill();
        drawCtx.strokeStyle = '#FFFFFF';
        drawCtx.lineWidth = 1;
        drawCtx.stroke();
    }
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'user',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        });
        webcamEl.srcObject = stream;
        state.cameraActive = true;
        startCameraBtn.style.display = 'none';
        stopCameraBtn.style.display = 'block';
        recordingInd.style.display = 'flex';
        webcamEl.style.transform = 'scaleX(-1)';
        console.log('✓ Camera started');
        startAutoPrediction();
    } catch (error) {
        console.error('Camera error:', error);
        alert('Could not access camera.');
    }
}

function stopCamera() {
    if (webcamEl.srcObject) {
        webcamEl.srcObject.getTracks().forEach(track => track.stop());
    }
    state.cameraActive = false;
    startCameraBtn.style.display = 'block';
    stopCameraBtn.style.display = 'none';
    recordingInd.style.display = 'none';
    webcamEl.style.transform = 'none';
    if (autoPredictionInterval) clearInterval(autoPredictionInterval);
    console.log('✓ Camera stopped');
}

let autoPredictionInterval = null;
function startAutoPrediction() {
    if (autoPredictionInterval) clearInterval(autoPredictionInterval);
    autoPredictionInterval = setInterval(() => {
        if (state.cameraActive) captureFrame();
    }, 150);
}

function captureFrame() {
    if (!state.cameraActive || !webcamEl.srcObject) return;
    canvasEl.width = webcamEl.videoWidth;
    canvasEl.height = webcamEl.videoHeight;
    const ctx = canvasEl.getContext('2d');
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(webcamEl, -canvasEl.width, 0);
    ctx.restore();
    canvasEl.toBlob(blob => {
        predictFromImage(blob, 'webcam.jpg');
    }, 'image/jpeg', 0.9);
}

function handleImageUpload() {
    const file = imageInput.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            canvasEl.width = img.width;
            canvasEl.height = img.height;
            const ctx = canvasEl.getContext('2d');
            ctx.drawImage(img, 0, 0);
            canvasEl.toBlob(blob => {
                predictFromImage(blob, file.name);
            }, 'image/jpeg', 0.9);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function predictFromImage(blob, fileName = 'image.jpg') {
    if (state.isProcessing) return;
    state.isProcessing = true;
    const formData = new FormData();
    formData.append('image', blob, fileName);
    fetch('/api/predict', {
        method: 'POST',
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        handlePredictionResponse(data);
    })
    .catch(e => {
        console.error('Prediction error:', e);
        predictedWordEl.textContent = 'ERROR';
    })
    .finally(() => {
        state.isProcessing = false;
    });
}

function handlePredictionResponse(data) {
    if (data.landmarks && data.landmarks.length > 0) {
        state.landmarks = data.landmarks;
        drawHandLandmarks(state.landmarks);
    }
    if (data.confidence !== undefined) {
        const confidence = data.confidence * 100;
        confidenceTextEl.textContent = confidence.toFixed(0) + '%';
        confidenceFillEl.style.width = confidence + '%';
    }
    if (data.buffer_progress !== undefined) {
        const progress = data.buffer_progress * 100;
        bufferFillEl.style.width = progress + '%';
        bufferStatusEl.textContent = Math.round(progress / 10) + '/10';
    }
    if (data.buffer_full && data.word && data.word !== 'No hand detected' && data.word !== 'Unknown') {
        predictedWordEl.textContent = data.word.toUpperCase();
        predictedWordEl.style.color = '#00D8FF';
        addWord(data.word, data.confidence);
    } else if (data.hand_detected && !data.buffer_full) {
        predictedWordEl.textContent = 'BUFFERING...';
        predictedWordEl.style.color = '#FFD700';
    } else {
        predictedWordEl.textContent = 'NO HAND';
        predictedWordEl.style.color = '#888888';
    }
}

function addWord(word, confidence = 1.0) {
    if (!word || word === 'Unknown') return;
    if (state.wordBuffer.length > 0 && state.wordBuffer[state.wordBuffer.length - 1] === word) {
        return;
    }
    state.wordBuffer.push(word);
    updateWordDisplay();
    updateButtonStates();
}

function removeLastWord() {
    if (state.wordBuffer.length > 0) {
        state.wordBuffer.pop();
        updateWordDisplay();
        updateButtonStates();
    }
}

function updateWordDisplay() {
    if (state.wordBuffer.length === 0) {
        wordBufferEl.innerHTML = '<span class="empty-message">Words will appear here...</span>';
    } else {
        wordBufferEl.innerHTML = state.wordBuffer
            .map((word, idx) => `<span class="word-tag" onclick="removeWord(${idx})">${word.toUpperCase()} ✕</span>`)
            .join('');
    }
}

function removeWord(idx) {
    state.wordBuffer.splice(idx, 1);
    updateWordDisplay();
    updateButtonStates();
}

function updateButtonStates() {
    formSentenceBtn.disabled = state.wordBuffer.length === 0;
    removeLastBtn.disabled = state.wordBuffer.length === 0;
}

async function formSentence() {
    if (state.wordBuffer.length === 0) {
        alert('No words in buffer!');
        return;
    }
    const wordsToSend = [...state.wordBuffer];
    sentenceOutputEl.textContent = 'PROCESSING...';
    sentenceOutputEl.className = 'sentence-text';
    try {
        const response = await fetch('/api/sentence', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ words: wordsToSend })
        });
        const data = await response.json();
        const sentence = data.corrected || data.original || wordsToSend.join(' ');
        sentenceOutputEl.textContent = sentence;
        sentenceOutputEl.className = 'sentence-text success';
        addToHistory(sentence, wordsToSend);
        state.wordBuffer = [];
        updateWordDisplay();
        updateButtonStates();
    } catch (error) {
        console.error('Error:', error);
        sentenceOutputEl.textContent = 'ERROR';
    }
}

function addToHistory(sentence, words) {
    state.sentenceHistory.unshift({
        sentence,
        words,
        timestamp: new Date()
    });
    updateHistory();
}

function updateHistory() {
    if (state.sentenceHistory.length === 0) {
        historyGridEl.innerHTML = '<div style="grid-column: 1 / -1; text-align: center; color: var(--steel);">No translations yet...</div>';
    } else {
        historyGridEl.innerHTML = state.sentenceHistory
            .map((item, idx) => `
                <div class="history-item">
                    <div class="history-num">Translation #${state.sentenceHistory.length - idx}</div>
                    <div class="history-text">${item.sentence}</div>
                    <div class="history-time">${item.timestamp.toLocaleString()}</div>
                </div>
            `).join('');
    }
}

function switchMode(mode) {
    state.currentMode = mode;
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    if (mode === 'camera') {
        event.target.classList.add('active');
        videoContainer.style.display = 'block';
        uploadZone.style.display = 'none';
    } else {
        event.target.classList.add('active');
        videoContainer.style.display = 'none';
        uploadZone.style.display = 'block';
        if (state.cameraActive) stopCamera();
    }
}

function resetAll() {
    state.wordBuffer = [];
    updateWordDisplay();
    updateButtonStates();
    sentenceOutputEl.textContent = 'Grammatically correct sentence will appear here...';
    sentenceOutputEl.className = 'sentence-text empty';
}
