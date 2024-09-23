import os
import cv2

def resize_and_save_images(input_dir, output_dir, size=(256, 256)):
    """
    Resizes images in the input directory and saves them to the output directory.
    Maintains the same directory structure.
    """
    for subdir, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(subdir, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, size)
                    # Maintain the directory structure
                    relative_path = os.path.relpath(subdir, input_dir)
                    out_dir = os.path.join(output_dir, relative_path)
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, file)
                    cv2.imwrite(out_path, img_resized)
                    print(f"Processed and saved image: {out_path}")
                else:
                    print(f"Failed to load image: {img_path}")
            else:
                print(f"Skipping non-image file: {file}")

def preprocess_entire_dataset():
    base_raw_dir = r'C:\Users\ramoy\PycharmProjects\CarDamageAssessment\data\raw'
    base_processed_dir = r'C:\Users\ramoy\PycharmProjects\CarDamageAssessment\data\processed'

    tasks = ['damage_detection', 'object_detection', 'severity_estimation']

    for task in tasks:
        input_dir = os.path.join(base_raw_dir, task)
        output_dir = os.path.join(base_processed_dir, task)
        resize_and_save_images(input_dir, output_dir)

if __name__ == "__main__":
    preprocess_entire_dataset()
