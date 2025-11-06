import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import datetime
import time
import threading
import queue
import json
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os

# Model path
MODEL_PATH = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\runs\rlhf\refined_rlhf_20251005_200232\weights\best.pt"

# Email settings
EMAIL_ADDRESS = "your_email@gmail.com"  # Replace with your Gmail
EMAIL_PASSWORD = "your_app_password"    # Gmail App Password (see below)
RECIPIENT_EMAIL = "riadh.ammari@etudiant-enit.utm.tn"

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("alerts.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (timestamp TEXT, alert_text TEXT, snapshot_path TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Load the model
@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model

model = load_model()

# Class names
CLASS_NAMES = ["person", "violence", "weapon"]

# Global variables for interval
person_alert_start = "22:00"
person_alert_end = "06:00"

def is_within_alert_interval(start_time_str, end_time_str):
    now = datetime.datetime.now().time()
    start = datetime.time.fromisoformat(start_time_str)
    end = datetime.time.fromisoformat(end_time_str)
    if start <= end:
        return start <= now <= end
    else:
        return now >= start or now <= end

def process_detections(results):
    alerts = []
    within_interval = is_within_alert_interval(person_alert_start, person_alert_end)
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            cls_name = CLASS_NAMES[cls_id]
            if cls_name in ["weapon", "violence"]:
                alerts.append(f"ALERT: {cls_name.upper()} detected (conf: {conf:.2f})")
            elif cls_name == "person":
                if within_interval:
                    alerts.append(f"ALERT: PERSON detected in restricted interval (conf: {conf:.2f})")
    return alerts

def send_email(alert_text, snapshot_path):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = "Intrusion Detection Alert"
    
    body = f"{alert_text}\n\nView the interface: http://localhost:8501\n"
    msg.attach(MIMEText(body, 'plain'))
    
    # Attach snapshot
    with open(snapshot_path, 'rb') as f:
        img = MIMEImage(f.read())
        img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(snapshot_path))
        msg.attach(img)
    
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)

alert_queue = queue.Queue()
detection_buffer = []

def detection_thread(video_source):
    global person_alert_start, person_alert_end, detection_buffer
    cap = cv2.VideoCapture(video_source)
    last_detection_time = time.time()
    last_alert_time = time.time()
    
    while cap.isOpened() and st.session_state.running:
        current_time = time.time()
        
        # Detect every 5 seconds
        if current_time - last_detection_time >= 5:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            alerts = process_detections(results)
            if alerts:
                detection_buffer.extend(alerts)
                # Save snapshot
                snapshot_path = f"snapshots/snapshot_{int(current_time)}.jpg"
                os.makedirs("snapshots", exist_ok=True)
                cv2.imwrite(snapshot_path, frame)
                # Store in database
                conn = sqlite3.connect("alerts.db")
                c = conn.cursor()
                c.execute("INSERT INTO alerts (timestamp, alert_text, snapshot_path) VALUES (?, ?, ?)",
                         (datetime.datetime.now().isoformat(), "; ".join(alerts), snapshot_path))
                conn.commit()
                conn.close()
            last_detection_time = current_time
        
        # Send alert every 15 seconds
        if current_time - last_alert_time >= 15 and detection_buffer:
            alert_text = f"ALERT at {datetime.datetime.now()}: " + "; ".join(detection_buffer)
            alert_queue.put(alert_text)
            # Send email with latest snapshot
            if detection_buffer:
                try:
                    send_email(alert_text, snapshot_path)
                except Exception as e:
                    print(f"Email sending failed: {e}")
            detection_buffer = []
            last_alert_time = current_time
        
        time.sleep(0.033)
    
    cap.release()

# Streamlit UI
st.title("Intrusion Detection Test Interface")

# Load saved settings
try:
    with open("config.json", "r") as f:
        config = json.load(f)
        person_alert_start = config.get("person_alert_start", "22:00")
        person_alert_end = config.get("person_alert_end", "06:00")
except FileNotFoundError:
    pass

# Time interval input
st.header("Set Person Alert Interval")
col1, col2 = st.columns(2)
with col1:
    start_time = st.time_input("Start Time", value=datetime.time(22, 0))
    person_alert_start = start_time.strftime("%H:%M")
with col2:
    end_time = st.time_input("End Time", value=datetime.time(6, 0))
    person_alert_end = end_time.strftime("%H:%M")

# Save settings
if st.button("Save Time Settings"):
    config = {"person_alert_start": person_alert_start, "person_alert_end": person_alert_end}
    with open("config.json", "w") as f:
        json.dump(config, f)
    st.success("Settings saved!")

# Video source selection
st.header("Select Video Source")
video_option = st.radio("Choose input:", ("Webcam", "Video File", "RTSP Stream"))
if video_option == "Webcam":
    video_source = 0
elif video_option == "Video File":
    video_source = st.text_input("Enter video file path:", value=r"C:\path\to\your\test_video.mp4")
else:
    video_source = st.text_input("Enter RTSP URL:", value="rtsp://your_camera_ip:port/stream")

# Start/Stop controls
if 'running' not in st.session_state:
    st.session_state.running = False

if st.button("Start Detection") and not st.session_state.running:
    st.session_state.running = True
    thread = threading.Thread(target=detection_thread, args=(video_source,))
    thread.start()
    st.write("Detection started...")

if st.button("Stop Detection") and st.session_state.running:
    st.session_state.running = False
    st.write("Detection stopped. (Note: Thread may take a moment to stop.)")

# Display alerts
st.header("Alerts")
alert_placeholder = st.empty()
if 'alerts' not in st.session_state:
    st.session_state.alerts = "No alerts yet."

# Update alerts
def update_alerts():
    try:
        alert = alert_queue.get_nowait()
        st.session_state.alerts = alert + "\n" + st.session_state.alerts[:1000]
        with open("alerts.log", "a") as f:
            f.write(f"{datetime.datetime.now()}: {alert}\n")
    except queue.Empty:
        pass
    if st.session_state.running:
        threading.Timer(0.1, update_alerts).start()

if st.session_state.running:
    update_alerts()

# Optional video feed
if st.checkbox("Show Video Feed (may slow down interface)"):
    frame_placeholder = st.empty()
    cap = cv2.VideoCapture(video_source)
    while st.session_state.running:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB")
        time.sleep(0.033)
    cap.release()