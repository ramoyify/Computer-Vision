import os
import torch
from yolov5.train import run as train_run
from yolov5.detect import run as detect_run

def train_model(data_yaml, img_size=640, batch_size=16, epochs=50, weights='yolov5s.pt', project='runs/train',
                name='exp'):
    print(f"Starting training with dataset: {data_yaml}")
    train_run(imgsz=img_size, batch_size=batch_size, epochs=epochs, data=data_yaml, weights=weights, project=project,
              name=name)
    print("Training complete. Model saved in:", os.path.join(project, name))

def detect_objects(weights_path, source='inference/images', img_size=320, conf_thres=0.1):  # Lowered conf_thres
    print(f"Starting object detection with model: {weights_path}")

    # Ensure img_size is passed as a tuple
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    detect_run(weights=weights_path, source=source, imgsz=img_size, conf_thres=conf_thres)
    print("Detection complete. Results saved in 'runs/detect/'")

if __name__ == "__main__":
    data_yaml = r'C:\Users\ramoy\PycharmProjects\CarDamageAssessment\config\data.yaml'
    train_model(data_yaml, img_size=640, batch_size=8, epochs=25, weights='yolov5s.pt')
    trained_weights = r'runs/train/exp9/weights/best.pt'
    detect_objects(weights_path=trained_weights, source='inference/images', img_size=320, conf_thres=0.1)
