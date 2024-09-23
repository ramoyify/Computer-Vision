import os
import torch
import cv2
import numpy as np
from torchvision import transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from yolov5.detect import run as detect_run
import tensorflow as tf
import glob

# VGG16 damage detection model
vgg16_model_path = r'C:\Users\ramoy\PycharmProjects\CarDamageAssessment\src\models\damage_detection_vgg16.h5'
vgg16_model = tf.keras.models.load_model(vgg16_model_path)


yolo_model_weights = r'C:\Users\ramoy\PycharmProjects\CarDamageAssessment\src\runs\train\exp23\weights\best.pt'


mask_rcnn_model = maskrcnn_resnet50_fpn(weights=None, num_classes=4)


in_features_mask = mask_rcnn_model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
mask_rcnn_model.roi_heads.mask_predictor = torch.nn.Sequential(
    torch.nn.ConvTranspose2d(in_features_mask, hidden_layer, 2, 2),
    torch.nn.ReLU(),
    torch.nn.ConvTranspose2d(hidden_layer, 4, 1)  # 4 corresponds to the number of classes
)


mask_rcnn_model_path = r'C:\Users\ramoy\PycharmProjects\CarDamageAssessment\src\mask_rcnn_severity_estimation.pth'
state_dict = torch.load(mask_rcnn_model_path)
mask_rcnn_model.load_state_dict(state_dict)


mask_rcnn_model.eval()


def run_pipeline(image_path):
    # Load the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_array = np.expand_dims(img_resized, axis=0)


    damage_prediction = vgg16_model.predict(img_array)
    is_damaged = damage_prediction[0][0] > 0.5


    label = "Damaged" if is_damaged else "Undamaged"
    color = (0, 255, 0) if is_damaged else (255, 0, 0)
    cv2.putText(img, f"Damage: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    severity_summary = []
    detected_damages = []
    damage_class = None  # Initialize damage_class

    # If the car is damaged, continue
    if is_damaged:
        # Run YOLOv5 object detection model
        detect_run(weights=yolo_model_weights, source=image_path, imgsz=(640, 640))


        yolo_results_dir = sorted(glob.glob(os.path.join('yolov5', 'runs', 'detect', 'exp*')), key=os.path.getmtime,
                                  reverse=True)

        if yolo_results_dir:
            yolo_labels_path = os.path.join(yolo_results_dir[0], 'labels')

            if os.path.exists(yolo_labels_path):
                yolo_results_files = glob.glob(os.path.join(yolo_labels_path, '*.txt'))
                if yolo_results_files:
                    with open(yolo_results_files[0], 'r') as file:
                        detections = file.readlines()

                    # YOLOv5
                    for detection in detections:
                        parts = detection.split()
                        class_id, confidence, x_center, y_center, width, height = map(float, parts[:6])
                        x1 = int((x_center - width / 2) * img.shape[1])
                        y1 = int((y_center - height / 2) * img.shape[0])
                        x2 = int((x_center + width / 2) * img.shape[1])
                        y2 = int((y_center + height / 2) * img.shape[0])
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, f"Damage {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    color, 2)

                        # Convert class_id
                        damage_classes = [
                            'front_engine_damage', 'front_hood_damage', 'front_bumper_damage',
                            'front_headlight_damage', 'front_windscreen_damage', 'rear_bumper_damage',
                            'rear_trunk_damage', 'rear_windshield_damage', 'rear_taillight_damage',
                            'side_window_damage', 'side_door_damage', 'side_panel_damage',
                            'side_mirror_damage', 'side_fender_damage'
                        ]
                        damage_class = damage_classes[int(class_id)]
                        detected_damages.append(damage_class)


        img_tensor = T.ToTensor()(img_rgb).unsqueeze(0)

        # Run Mask R-CNN
        with torch.no_grad():
            severity_predictions = mask_rcnn_model(img_tensor)


        masks = severity_predictions[0]['masks'].cpu().numpy()
        labels = severity_predictions[0]['labels'].cpu().numpy()

        severity_map = {1: "Minor", 2: "Moderate", 3: "Severe"}
        colors = {
            "Minor": (0, 255, 0),  # Green for minor
            "Moderate": (255, 255, 0),  # Yellow for moderate
            "Severe": (0, 0, 255)  # Red for severe
        }

        for i, mask in enumerate(masks):
            mask = mask[0]
            severity = severity_map.get(labels[i], "Unknown")
            color = colors.get(severity, (255, 255, 255))


            overlay = img.copy()
            overlay[mask > 0.5] = color
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)


            pos = np.where(mask > 0.5)
            y, x = pos[0][0], pos[1][0]
            cv2.putText(img, severity, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


            if damage_class is not None:
                severity_summary.append(f"{severity} {damage_class.replace('_', ' ')} detected")


    output_filename = os.path.basename(image_path).replace(".jpg", "_result.jpg")
    output_path = os.path.join('uploads', output_filename)
    cv2.imwrite(output_path, img)
    print(f"Final result saved to {output_path}")


    return output_path, ". ".join(severity_summary)

