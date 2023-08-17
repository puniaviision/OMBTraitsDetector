import os
import csv

image_folder = '/Users/punia/Projects/ordinals/ordinals'
image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith('.jpeg')] # adjust the file extension as needed

with open('image_paths.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for image_path in image_paths:
        writer.writerow([image_path])

