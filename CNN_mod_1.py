import os
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the input shape of the images
input_shape = (28, 28, 1)  # Adjust according to your image size and channels

# Load CSV file containing image paths and labels
csv_path="/home/sourav/Desktop/CC_diag/dataset/dataset.csv"
df = pd.read_csv(csv_path)  # Assuming 'image_data.csv' contains 'path' and 'label' columns
print(df.head())

# Load and preprocess images
def load_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale image
        img = cv2.resize(img, (28, 28))  # Resize to input shape
        img = img.astype('float32') / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        images.append(img)
    return np.array(images)

image_paths = df['path'].tolist()
labels = df['label'].tolist()

# Load and preprocess images
images = load_images(image_paths)

# Prepare labels (assuming labels are already encoded as integers)
labels = np.array(labels)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create CNN model
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='softmax')  # Assuming 4 classes
    ])
    return model

# Create an instance of the CNN model
model = create_cnn_model()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_val, y_val)
print(f'Test accuracy: {test_acc}')
