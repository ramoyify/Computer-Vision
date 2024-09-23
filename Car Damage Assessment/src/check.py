import os

# Class name to index mapping
class_name_to_index = {
    "front_hood_damage": 0,
    "front_engine_damage": 1,
    "front_bumper_damage": 2,
    "front_headlight_damage": 3,
    "front_fender_damage": 4,
    "front_windshield_damage": 5,
    "front_tyre_damage": 6,
    "rear_bumper_damage": 7,
    "rear_trunk_damage": 8,
    "rear_windshield_damage": 9,
    "rear_fender_damage": 10,
    "rear_backlight_damage": 11,
    "rear_panel_damage": 12,
    "rear_tyre_damage": 13,
    "side_window_damage": 14,
    "side_door_damage": 15,
    "side_mirror_damage": 16
}


label_dir = r"C:\Users\ramoy\PycharmProjects\CarDamageAssessment\data\processed\object_detection\front\labels"

def update_labels_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            if not lines:
                print(f"Skipping empty file: {file_path}")
                continue

            new_lines = []
            for line in lines:
                parts = line.split()
                if not parts:
                    continue

                class_name = parts[0]
                if class_name in class_name_to_index:
                    parts[0] = str(class_name_to_index[class_name])
                    new_lines.append(" ".join(parts))
                else:
                    print(f"Unknown class name '{class_name}' in {file_path}, skipping line.")

            if new_lines:
                with open(file_path, 'w') as file:
                    file.write("\n".join(new_lines))
                print(f"Updated labels in: {file_path}")
            else:
                print(f"No valid labels found in {file_path}, not overwriting file.")

# Update labels in the specified directory
update_labels_in_directory(label_dir)

print("Label files update process complete.")
