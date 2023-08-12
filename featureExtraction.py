import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
import numpy as np
import csv

# Load pre-trained ResNet model + higher level layers
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

features = []
image_paths = []

# Read image paths from CSV
with open('image_paths.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        image_paths.append(row[0])

# Extract features for each image
for img_path in image_paths:
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features.append(base_model.predict(preprocessed_img))

features = np.squeeze(features)  # Remove single-dimensional entries
