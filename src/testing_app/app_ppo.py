from flask import Flask, request, Response, render_template, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import tempfile
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import torch

app = Flask(__name__, template_folder='templates')

# Model paths
yolo_model_path = Path("C:/Users/amari/OneDrive/Desktop/EdgeAI_Project/runs/train/intrusion_detection_v05_e80_20250804_1427/weights/best.pt")
ppo_model_path = Path("C:/Users/amari/OneDrive/Desktop/EdgeAI_Project/models/ppo_threshold_policy.zip")

# Load models
yolo_model = YOLO(yolo_model_path)
ppo_model = None
if ppo_model_path.exists():
    # Define a minimal environment for PPO (only observation space needed)
    class MinimalEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = Dict({
                'image': Box(low=0, high=255, shape=(3, 224, 224), dtype=np.uint8),
                'thresholds': Box(low=0.1, high=0.9, shape=(3,), dtype=np.float32)
            })
            self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        def reset(self, **kwargs):
            return {'image': np.zeros((3, 224, 224), dtype=np.uint8), 'thresholds': np.array([0.3, 0.3, 0.3], dtype=np.float32)}, {}

        def step(self, action):
            return self.reset()[0], 0.0, False, False, {}

    env = MinimalEnv()
    ppo_model = PPO.load(ppo_model_path, env=env, device='cuda' if torch.cuda.is_available() else 'cpu', weights_only=True)
    print(f"PPO model loaded from {ppo_model_path}")
else:
    print(f"Warning: PPO model not found at {ppo_model_path}, using default thresholds [0.3, 0.3, 0.3]")

def preprocess_image_for_ppo(img):
    """Preprocess image for PPO model (same as YOLOThresholdEnv._load_image)."""
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))  # CHW format
    return img

def get_ppo_thresholds(img):
    """Predict thresholds using PPO model."""
    if ppo_model is None:
        return np.array([0.3, 0.3, 0.3], dtype=np.float32)
    obs = {
        'image': preprocess_image_for_ppo(img),
        'thresholds': np.array([0.3, 0.3, 0.3], dtype=np.float32)
    }
    action, _ = ppo_model.predict(obs, deterministic=True)
    thresholds = np.clip((action + 1) * 0.4 + 0.1, 0.1, 0.9)
    return thresholds

def apply_thresholds(results, thresholds):
    """Filter YOLO predictions with class-specific thresholds."""
    filtered_boxes = []
    for box in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = box
        if conf >= thresholds[int(cls)]:
            filtered_boxes.append(box)
    results.boxes.data = torch.tensor(filtered_boxes, device=results.boxes.data.device)
    return results

def generate_frames(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nError: Could not open video file\r\n'
        return

    frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / 5)  # 5 FPS
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        # Get PPO-predicted thresholds
        thresholds = get_ppo_thresholds(frame)
        results = yolo_model.predict(source=frame, conf=0.01, classes=[0, 1, 2], imgsz=320, verbose=False)[0]
        results = apply_thresholds(results, thresholds)
        annotated_frame = results.plot()
        if annotated_frame is None or annotated_frame.size == 0:
            continue

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ['.mp4', '.avi']:
        return jsonify({"error": "Unsupported file format"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_path = temp_file.name
        file.save(temp_path)

    try:
        return Response(generate_frames(temp_path),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    finally:
        os.unlink(temp_path)

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ['.jpg', '.jpeg', '.png']:
        return jsonify({"error": "Unsupported file format"}), 400

    img_array = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img_array is None:
        return jsonify({"error": "Failed to decode image"}), 400

    thresholds = get_ppo_thresholds(img_array)
    results = yolo_model.predict(source=img_array, conf=0.01, classes=[0, 1, 2], imgsz=320, verbose=False)[0]
    results = apply_thresholds(results, thresholds)
    annotated_img = results.plot()
    if annotated_img is None or annotated_img.size == 0:
        return jsonify({"error": "No valid annotation generated"}), 500
    ret, buffer = cv2.imencode('.jpg', annotated_img)
    if not ret:
        return jsonify({"error": "Failed to encode image"}), 500
    return Response(buffer.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    if not yolo_model_path.exists():
        print(f"Error: YOLO model file not found at {yolo_model_path}")
    else:
        print(f"YOLO model loaded from {yolo_model_path}")
    app.run(debug=True, host='0.0.0.0', port=5000)