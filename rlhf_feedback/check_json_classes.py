import json

json_path = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\rlhf_feedback\preference_pairs_v2.json"
with open(json_path, "r") as f:
    pairs = json.load(f)

class_counts = {0: 0, 1: 0, 2: 0}
for pair in pairs:
    for box in pair["option_A"]:
        class_counts[box[0]] += 1
    for box in pair["option_B"]:
        class_counts[box[0]] += 1

total = sum(class_counts.values())
print(f"Total boxes: {total}")
print(f"Class distribution: Person={class_counts[0]/total:.2%}, Weapon={class_counts[1]/total:.2%}, Violence={class_counts[2]/total:.2%}")