from flask import Flask, request, Response, render_template, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import tempfile
import json
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')
model_path = Path("C:/Users/amari/OneDrive/Desktop/EdgeAI_Project/runs/train/intrusion_detection_v07_e100_20250806_0959/weights/best.pt")

# Initialize model
try:
    model = YOLO(model_path)  # Load the checkpoint
    model.model.eval()
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Class names: {model.names}")
except Exception as e:
    logger.error(f"Error loading model from {model_path}: {e}")
    model = None

# Class names fallback
class_names = {0: "person", 1: "violence", 2: "weapon"}

def get_class_names():
    """Retrieve class names from the model or use fallback."""
    return getattr(model, 'names', class_names) if model else class_names

def generate_frames(file_path):
    if not model:
        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nError: Model not loaded\r\n'
        return

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nError: Could not open video file\r\n'
        return

    frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / 5)  # 5 FPS
    frame_count = 0
    frame_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        try:
            results = model.predict(source=frame, conf=0.3, classes=[0, 1, 2], imgsz=416)
            detections = []
            for result in results:
                annotated_frame = result.plot()
                if annotated_frame is None or annotated_frame.size == 0:
                    logger.warning(f"No valid annotation for frame {frame_count}")
                    continue

                boxes = result.boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
                height, width = frame.shape[:2]
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box
                    cls = int(cls)
                    class_name = get_class_names().get(cls, "unknown")
                    detections.append({
                        "class": class_name,
                        "confidence": float(conf),
                        "bbox": [float(x1)/width, float(y1)/height, float(x2)/width, float(y2)/height],
                        "frame": frame_count // frame_interval
                    })
                    logger.info(f"Frame {frame_count}: Detected {class_name} with conf {conf:.4f} at bbox {[float(x1)/width, float(y1)/height, float(x2)/width, float(y2)/height]}")

                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if not ret:
                    logger.error(f"Failed to encode frame {frame_count}")
                    continue

                frame_data = buffer.tobytes()
                frame_detections.append((frame_count // frame_interval, detections))
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        except Exception as e:
            logger.error(f"Error processing frame {frame_count}: {e}")
            continue

    cap.release()
    try:
        with open(tempfile.gettempdir() + '/detections.json', 'w') as f:
            json.dump(frame_detections, f)
        logger.info(f"Saved detections to {tempfile.gettempdir()}/detections.json")
    except Exception as e:
        logger.error(f"Error saving detections: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_video():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
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
        try:
            os.unlink(temp_path)
            logger.info(f"Deleted temp file {temp_path}")
        except Exception as e:
            logger.error(f"Error deleting temp file: {e}")

@app.route('/process_image', methods=['POST'])
def process_image():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ['.jpg', '.jpeg', '.png']:
        return jsonify({"error": "Unsupported file format"}), 400

    try:
        img_array = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img_array is None:
            return jsonify({"error": "Failed to decode image"}), 400

        results = model.predict(source=img_array, conf=0.3, classes=[0, 1, 2], imgsz=416)
        detections = []
        annotated_img = None
        height, width = img_array.shape[:2]
        for result in results:
            annotated_img = result.plot()
            if annotated_img is None or annotated_img.size == 0:
                logger.warning("No valid annotation generated")
                return jsonify({"error": "No valid annotation generated"}), 500
            boxes = result.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                cls = int(cls)
                class_name = get_class_names().get(cls, "unknown")
                detections.append({
                    "class": class_name,
                    "confidence": float(conf),
                    "bbox": [float(x1)/width, float(y1)/height, float(x2)/width, float(y2)/height]
                })
                logger.info(f"Image detection: {class_name} with conf {conf:.4f} at bbox {[float(x1)/width, float(y1)/height, float(x2)/width, float(y2)/height]}")

        ret, buffer = cv2.imencode('.jpg', annotated_img)
        if not ret:
            return jsonify({"error": "Failed to encode image"}), 500

        response = {
            "image": buffer.tobytes().hex(),
            "detections": detections
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({"error": f"Error processing image: {e}"}), 500

@app.route('/detections', methods=['GET'])
def get_detections():
    try:
        with open(tempfile.gettempdir() + '/detections.json', 'r') as f:
            detections = json.load(f)
        return jsonify(detections)
    except FileNotFoundError:
        return jsonify([])
    except Exception as e:
        logger.error(f"Error reading detections: {e}")
        return jsonify({"error": f"Error reading detections: {e}"}), 500

if __name__ == '__main__':
    if not model_path.exists():
        logger.error(f"Error: Model file not found at {model_path}")
        exit(1)
    app.run(debug=True, host='0.0.0.0', port=5000)