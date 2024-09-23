import os
import torch
import cv2
import numpy as np
from torchvision import transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from yolov5.detect import run as detect_run
import tensorflow as tf
import glob

# Load the VGG16 damage detection model
vgg16_model_path = r'C:\Users\ramoy\PycharmProjects\CarDamageAssessment\src\models\damage_detection_vgg16.h5'
vgg16_model = tf.keras.models.load_model(vgg16_model_path)

# Load the YOLOv5 model weights
yolo_model_weights = r'C:\Users\ramoy\PycharmProjects\CarDamageAssessment\src\runs\train\exp23\weights\best.pt'

# Load the Mask R-CNN model (with custom layers if any)
mask_rcnn_model = maskrcnn_resnet50_fpn(weights=None, num_classes=4)

# Rebuild the custom layers as done during training
in_features_mask = mask_rcnn_model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
mask_rcnn_model.roi_heads.mask_predictor = torch.nn.Sequential(
    torch.nn.ConvTranspose2d(in_features_mask, hidden_layer, 2, 2),
    torch.nn.ReLU(),
    torch.nn.ConvTranspose2d(hidden_layer, 4, 1)  # 4 corresponds to the number of classes
)

# Load the model weights
mask_rcnn_model_path = r'C:\Users\ramoy\PycharmProjects\CarDamageAssessment\src\mask_rcnn_severity_estimation.pth'
state_dict = torch.load(mask_rcnn_model_path)
mask_rcnn_model.load_state_dict(state_dict)

# Set the Mask R-CNN model to evaluation mode
mask_rcnn_model.eval()

# Function to process the image and run it through the models
def run_pipeline(image_path):
    # Load the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_array = np.expand_dims(img_resized, axis=0)

    # Run VGG16 damage detection model
    damage_prediction = vgg16_model.predict(img_array)
    is_damaged = damage_prediction[0][0] > 0.5

    # Add VGG16 result to image
    label = "Damaged" if is_damaged else "Undamaged"
    color = (0, 255, 0) if is_damaged else (255, 0, 0)
    cv2.putText(img, f"Damage: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Initialize description
    damage_description = "The vehicle is undamaged."

    # If the car is damaged, continue with further analysis
    if is_damaged:
        # Run YOLOv5 object detection model
        detect_run(weights=yolo_model_weights, source=image_path, imgsz=(640, 640))

        # YOLOv5 will save the results in the 'runs/detect' folder. Load the latest result.
        yolo_results_dir = sorted(glob.glob(os.path.join('..', 'yolov5', 'runs', 'detect', 'exp*')), key=os.path.getmtime, reverse=True)

        if yolo_results_dir:
            yolo_labels_path = os.path.join(yolo_results_dir[0], 'labels')

            if os.path.exists(yolo_labels_path):
                yolo_results_files = glob.glob(os.path.join(yolo_labels_path, '*.txt'))
                if yolo_results_files:
                    with open(yolo_results_files[0], 'r') as file:
                        detections = file.readlines()

                    # Draw YOLOv5 bounding boxes on the image
                    for detection in detections:
                        parts = detection.split()
                        class_id, confidence, x_center, y_center, width, height = map(float, parts[:6])
                        x1 = int((x_center - width / 2) * img.shape[1])
                        y1 = int((y_center - height / 2) * img.shape[0])
                        x2 = int((x_center + width / 2) * img.shape[1])
                        y2 = int((y_center + height / 2) * img.shape[0])
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, f"Damage {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                else:
                    print("No detection results found.")
            else:
                print("YOLOv5 labels directory does not exist.")
        else:
            print("No YOLOv5 results directories found.")

        # Prepare image for Mask R-CNN
        img_tensor = T.ToTensor()(img_rgb).unsqueeze(0)

        # Run Mask R-CNN severity estimation model
        with torch.no_grad():
            severity_predictions = mask_rcnn_model(img_tensor)

        # Draw Mask R-CNN masks and severity labels on the image
        masks = severity_predictions[0]['masks'].cpu().numpy()
        labels = severity_predictions[0]['labels'].cpu().numpy()

        severity_map = {1: "Minor", 2: "Moderate", 3: "Severe"}
        colors = {
            "Minor": (0, 255, 0),    # Green for minor
            "Moderate": (255, 255, 0), # Yellow for moderate
            "Severe": (0, 0, 255)    # Red for severe
        }

        # Track damage counts
        severity_counts = {"Minor": 0, "Moderate": 0, "Severe": 0}

        for i, mask in enumerate(masks):
            mask = mask[0]
            severity = severity_map.get(labels[i], "Unknown")
            color = colors.get(severity, (255, 255, 255))
            severity_counts[severity] += 1

            # Create an overlay with the mask and combine it with the original image
            overlay = img.copy()
            overlay[mask > 0.5] = color
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

            # Draw the severity label
            pos = np.where(mask > 0.5)
            y, x = pos[0][0], pos[1][0]
            cv2.putText(img, severity, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Generate a description based on the severity counts
        damage_description = "The vehicle has sustained damage with the following severity:\n"
        for severity, count in severity_counts.items():
            if count > 0:
                damage_description += f"{count} {severity.lower()} damage area(s).\n"

    # Save the final image with all annotations
    output_path = image_path.replace(".jpg", "_result.jpg")
    cv2.imwrite(output_path, img)
    print(f"Final result saved to {output_path}")

    return output_path, damage_description
