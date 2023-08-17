import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
import numpy as np
import csv
from sklearn.cluster import KMeans


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
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features.append(base_model.predict(preprocessed_img))


features = np.squeeze(features)  # Remove single-dimensional entries

# Perform clustering
n_clusters = 10  # You can choose a different number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)

# Get cluster assignments
cluster_assignments = kmeans.labels_

# Save image paths and cluster assignments to CSV
with open('clusters.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for img_path, cluster in zip(image_paths, cluster_assignments):
        writer.writerow([img_path, cluster])