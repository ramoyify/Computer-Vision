import os  # Make sure to import the os module
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Ensure the models directory exists
if not os.path.exists('models'):
    os.makedirs('models')


# Function to load images and labels
def load_images(base_dir):
    images = []
    labels = []
    categories = {'damaged': 1, 'undamaged': 0}  # Define your categories and labels
    for category, label in categories.items():
        category_dir = os.path.join(base_dir, category)
        if not os.path.exists(category_dir):
            print(f"Directory does not exist: {category_dir}")
            continue
        for file_name in os.listdir(category_dir):
            image_path = os.path.join(category_dir, file_name)
            image = cv2.imread(image_path)
            if image is not None:
                image_resized = cv2.resize(image, (128, 128))  # Resize images to 128x128
                images.append(image_resized)
                labels.append(label)
    return np.array(images), np.array(labels)


# VGG16 Transfer Learning Model
def transfer_learning_model(input_size=(128, 128, 3)):
    base_model = VGG16(input_shape=input_size, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Training function
def train_vgg16_classification(base_dir):
    X, y = load_images(base_dir)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the pixel values
    X_train = X_train / 255.0
    X_val = X_val / 255.0

    model = transfer_learning_model(input_size=(128, 128, 3))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)  # Adjust epochs as needed
    model.save('models/damage_detection_vgg16.keras')  # Save in .keras format


if __name__ == "__main__":
    base_dir = r'C:\Users\ramoy\PycharmProjects\CarDamageAssessment\data\processed\damage_detection'
    train_vgg16_classification(base_dir)
