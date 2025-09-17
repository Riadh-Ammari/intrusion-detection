import os
import json
import ast

# Define paths (use raw string for Windows paths with backslashes)
labels_dir = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\outputs\yolo_detection\val\labels"
output_json_path = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\preference_pairs_test.json"  # Save in your project root
output_json = []

# Process each text file
for txt_file in os.listdir(labels_dir):
    if txt_file.endswith(".txt"):
        try:
            with open(os.path.join(labels_dir, txt_file), 'r') as f:
                lines = f.readlines()
                data = {}
                for line in lines:
                    key, value = line.strip().split(": ", 1)
                    data[key] = value
            # Parse Ground Truth and Predictions
            ground_truth = ast.literal_eval(data["Ground Truth"])
            predictions = ast.literal_eval(data["Predictions"])
            # Create preference pair
            pair = {
                "image": txt_file.replace(".txt", ".jpg"),
                "option_A": predictions,  # YOLO predictions with confidence
                "option_B": ground_truth, # Human-annotated ground truth
                "preference": 1          # Ground truth is preferred
            }
            output_json.append(pair)
        except Exception as e:
            print(f"Error processing {txt_file}: {e}")
            continue

# Save to JSON
with open(output_json_path, "w") as f:
    json.dump(output_json, f, indent=2)

# Validate
print(f"Created {len(output_json)} preference pairs in {output_json_path}")