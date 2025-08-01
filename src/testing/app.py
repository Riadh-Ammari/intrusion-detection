import cv2
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
from pathlib import Path
import numpy as np
from PIL import Image, ImageTk

# Load the trained model
model_path = Path("C:/Users/amari/OneDrive/Desktop/EdgeAI_Project/runs/train/intrusion_detection_01/weights/best.pt")
model = YOLO(model_path)

# Function to process and display image
def process_image(image_path):
    results = model.predict(source=image_path, conf=0.5)
    for result in results:
        annotated_image = result.plot()  # Get annotated image
        # Convert to PIL Image for display
        img = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        img = ImageTk.PhotoImage(img)
        # Create new window to display result
        result_window = tk.Toplevel(root)
        result_window.title("Detection Result")
        label = tk.Label(result_window, image=img)
        label.image = img  # Keep a reference to avoid garbage collection
        label.pack()
        messagebox.showinfo("Success", f"Image processed and displayed. Saved as output_image.jpg")
        result.save(filename="output_image.jpg")  # Save for reference

# Function to process and handle video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video file.")
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=0.5)
        for result in results:
            annotated_frame = result.plot()
            out.write(annotated_frame)
    cap.release()
    out.release()
    messagebox.showinfo("Success", f"Video processed and saved as output_video.mp4")
    # Optional: Play the video (Windows Media Player or similar)
    os.startfile("output_video.mp4")

# Function to handle file selection and processing
def select_and_process():
    file_path = filedialog.askopenfilename(filetypes=[("Image/Video files", "*.jpg *.jpeg *.png *.mp4 *.avi")])
    if not file_path:
        return
    file_extension = Path(file_path).suffix.lower()
    if file_extension in [".jpg", ".jpeg", ".png"]:
        process_image(file_path)
    elif file_extension in [".mp4", ".avi"]:
        process_video(file_path)
    else:
        messagebox.showerror("Error", "Unsupported file format. Please select a .jpg, .png, .mp4, or .avi file.")

# Create the main window
root = tk.Tk()
root.title("YOLO Detection App")
root.geometry("300x200")

# Add button for file selection
select_button = tk.Button(root, text="Select Image or Video", command=select_and_process, font=("Arial", 12))
select_button.pack(pady=50)

# Start the application
root.mainloop()