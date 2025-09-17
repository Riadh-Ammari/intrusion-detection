import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

class SurveillanceValidator:
    def __init__(self):
        self.model_path = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\runs\train\intrusion_detection_v10_e30_20250815_1035\finetune\weights\best.pt"
        self.val_dir = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\data\dataset_pre\images\val"
        self.output_dir = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\outputs\yolo_detection\val"
        self.model = YOLO(self.model_path)
        self.class_names = ['person', 'violence', 'weapon']
        self.conf_threshold = 0.2
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)

    def process_image(self, img_path):
        """Process a single image and save predictions in YOLO format"""
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Failed to load image: {img_path}")
            return None

        # Run YOLO inference
        results = self.model(frame, conf=self.conf_threshold)[0]
        
        # Save predictions in YOLO format
        img_name = Path(img_path).name
        label_path = os.path.join(self.output_dir, 'labels', img_name.replace(Path(img_path).suffix, '.txt'))
        predictions = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            
            # Convert to YOLO format (normalized coordinates)
            img_height, img_width = frame.shape[:2]
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            predictions.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}")
        
        # Save predictions
        with open(label_path, 'w') as f:
            f.write("\n".join(predictions))
        
        # Save annotated image for review
        annotated_frame = self.annotate_frame(frame, results)
        cv2.imwrite(os.path.join(self.output_dir, 'images', img_name), annotated_frame)
        
        return predictions

    def annotate_frame(self, frame, results):
        """Annotate frame with bounding boxes and labels"""
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            color = (0, int(255 * confidence), int(255 * (1 - confidence)))
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label with background
            label = f'{self.class_names[class_id]}: {confidence:.2f}'
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_bg_x1, label_bg_y1 = int(x1), int(y1 - label_height - 10)
            label_bg_x2, label_bg_y2 = label_bg_x1 + label_width + 10, int(y1)
            cv2.rectangle(frame, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), (0, 0, 0), -1)
            cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame

    def run_validation(self):
        """Process all images in the validation set"""
        val_images = [f for f in os.listdir(self.val_dir) if f.endswith(('.jpg', '.png'))]
        if not val_images:
            print(f"No images found in {self.val_dir}")
            return
        for img_name in val_images:
            img_path = os.path.join(self.val_dir, img_name)
            self.process_image(img_path)
            print(f"Processed {img_name}")

if __name__ == "__main__":
    validator = SurveillanceValidator()
    validator.run_validation()