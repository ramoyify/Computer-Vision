import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import torch.nn as nn


class CarDamageDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.coco = COCO(annotation)
        self.transforms = transforms
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']

        # Load image
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        # Load annotations
        num_objs = len(anns)
        boxes = []
        labels = []
        masks = []

        for ann in anns:
            # Get bounding box coordinates
            xmin, ymin, width, height = ann['bbox']
            if width <= 0 or height <= 0:
                continue  # Skip invalid bounding boxes
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann['category_id'])
            # Get segmentation mask
            mask = self.coco.annToMask(ann)
            masks.append(mask)

        if len(boxes) == 0:
            return None

        # Convert everything to torch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([img_id])
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch))


def train_model():
    # Directories for the dataset
    train_dir = r'C:\Users\ramoy\PycharmProjects\CarDamageAssessment\data\processed\severity_estimation\train'
    valid_dir = r'C:\Users\ramoy\PycharmProjects\CarDamageAssessment\data\processed\severity_estimation\valid'

    train_annotation = os.path.join(train_dir, 'annotations.coco.json')
    valid_annotation = os.path.join(valid_dir, 'annotations.coco.json')

    # Dataset and DataLoader
    train_dataset = CarDamageDataset(root=train_dir, annotation=train_annotation, transforms=get_transform(train=True))
    valid_dataset = CarDamageDataset(root=valid_dir, annotation=valid_annotation, transforms=get_transform(train=False))

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)


    model = maskrcnn_resnet50_fpn(weights='MaskRCNN_ResNet50_FPN_Weights.DEFAULT')
    num_classes = 4


    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = nn.Linear(in_features, num_classes)


    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(in_features_mask, hidden_layer, 2, 2),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(hidden_layer, num_classes, 1)
    )


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Set up optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        i = 0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            i += 1
            if i % 10 == 0:
                print(f"Epoch: {epoch + 1}, Step: {i}, Loss: {losses.item()}")

    print("Training complete. Saving model...")
    torch.save(model.state_dict(), "mask_rcnn_severity_estimation.pth")


if __name__ == "__main__":
    train_model()
