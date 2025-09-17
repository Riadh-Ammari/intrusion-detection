import json
import numpy as np

json_path = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\rlhf_feedback\preference_pairs_v2.json"

# Load JSON
try:
    with open(json_path, "r") as f:
        pairs = json.load(f)
except Exception as e:
    print(f"Error loading JSON: {e}")
    exit()

if not pairs:
    print("JSON is empty, please rerun generate_preference_pairs.py")
    exit()

# Initialize counters
class_counts = {0: 0, 1: 0, 2: 0}
invalid_boxes = []
valid_pairs = 0

for pair in pairs:
    img_name = pair.get("image", "unknown")
    # Check option_A
    option_A = pair.get("option_A", [])
    if not option_A:
        invalid_boxes.append(f"{img_name}: Empty option_A")
        continue
    for box in option_A:
        if not isinstance(box, list) or len(box) != 2:
            invalid_boxes.append(f"{img_name}: Invalid option_A box format: {box}")
            continue
        cls, coords = box
        if not isinstance(coords, list) or len(coords) != 5:
            invalid_boxes.append(f"{img_name}: Invalid option_A coords, expected 5, got {len(coords)}: {box}")
            continue
        if not all(isinstance(c, (int, float, np.floating)) for c in coords):
            invalid_boxes.append(f"{img_name}: Non-numeric values in option_A box: {box}")
            continue
        if any(c < 0 for c in coords[:4]):
            invalid_boxes.append(f"{img_name}: Negative coords in option_A box: {box}")
            continue
        if cls not in [0, 1, 2]:
            invalid_boxes.append(f"{img_name}: Invalid class in option_A box: {cls}")
            continue
        class_counts[cls] += 1

    # Check option_B
    option_B = pair.get("option_B", [])
    if not option_B:
        invalid_boxes.append(f"{img_name}: Empty option_B")
        continue
    for box in option_B:
        if not isinstance(box, list) or len(box) != 2:
            invalid_boxes.append(f"{img_name}: Invalid option_B box format: {box}")
            continue
        cls, coords = box
        if not isinstance(coords, list) or len(coords) != 4:
            invalid_boxes.append(f"{img_name}: Invalid option_B coords, expected 4, got {len(coords)}: {box}")
            continue
        if not all(isinstance(c, (int, float, np.floating)) for c in coords):
            invalid_boxes.append(f"{img_name}: Non-numeric values in option_B box: {box}")
            continue
        if any(c < 0 for c in coords):
            invalid_boxes.append(f"{img_name}: Negative coords in option_B box: {box}")
            continue
        if cls not in [0, 1, 2]:
            invalid_boxes.append(f"{img_name}: Invalid class in option_B box: {cls}")
            continue
        class_counts[cls] += 1

    # Check preference
    if pair.get("preference") != 1:
        invalid_boxes.append(f"{img_name}: Invalid preference value: {pair.get('preference')}")
        continue

    valid_pairs += 1

# Print results
total_boxes = sum(class_counts.values())
print(f"Total pairs: {len(pairs)}")
print(f"Valid pairs: {valid_pairs}")
if total_boxes > 0:
    print(f"Total boxes: {total_boxes}")
    print(f"Class distribution: Person={class_counts[0]/total_boxes:.2%}, Weapon={class_counts[1]/total_boxes:.2%}, Violence={class_counts[2]/total_boxes:.2%}")
else:
    print("No valid boxes found")
if invalid_boxes:
    print("\nInvalid boxes or pairs found:")
    for error in invalid_boxes[:10]:  # Limit to first 10 errors for brevity
        print(error)
    if len(invalid_boxes) > 10:
        print(f"...and {len(invalid_boxes) - 10} more errors")
else:
    print("\nNo invalid boxes or pairs found")