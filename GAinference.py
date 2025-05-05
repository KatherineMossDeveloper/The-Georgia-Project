# The Georgia project.
# GAinference.py  Version 1.
#
# This code will pull png files from a folder and do inference on each one,
# reporting the classification and confidence to the output window.
#
# To do.
# create an inference folder on your computer.
# edit the image_folder and weights_file strings to point to your folder.
# Add some images to this folder.
# Save the weights file created during your training of the Georgia project to this folder, or
#      use the weights...h5 file provided in the GitHub Geogia Project.
# #############################################################################################

import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


# Set up the path to your image folder and weights file
# for testing only...image_folder = r'X:\MLresearch\CrystalStudy\Project_GA\data\GAtestBinaryDEBUG\0'
image_folder = r'X:\MLresearch\CrystalStudy\Project_GA\data\GAinference'
weights_file = r'X:\MLresearch\CrystalStudy\Project_GA\data\GAinference\GAweights_2025-03-22_16-43-54.h5'

# Load the pre-trained model with weights
print(f'Starting GAinference.py. ')
model = load_model(weights_file)
png_files_in_folder = [f for f in os.listdir(image_folder) if f.endswith('.png')]  # the no. of files in folder.
png_file_count = len(png_files_in_folder)


# Function to load, resize, and preprocess an image
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    # Load and resize the image to the target size (224, 224)
    img = image.load_img(img_path, target_size=target_size)

    # Convert the image to a NumPy array
    image_array = image.img_to_array(img)

    # Add batch dimension (the model expects a batch of images, not just one)
    image_array = np.expand_dims(image_array, axis=0)  # Shape: [1, 224, 224, 3]

    # Apply ResNet-specific preprocessing (mean subtraction and scaling)
    image_array = preprocess_input(image_array)

    return image_array


# Inference for each image in the folder
try:
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):  # Check if the file is a PNG file
            file_path = os.path.join(image_folder, filename)

            # Load and preprocess the image
            img_array = load_and_preprocess_image(file_path)

            # Perform inference on the image
            prediction = model.predict(img_array)

            # Print the full prediction output to see the raw values
            print(f"Prediction for {filename}: {prediction}")

            # Assuming binary classification: CEX (0) vs PG (1)
            confidence = prediction[0][0]  # This is the predicted probability for PG

            # Assign the class label based on the threshold of 0.5
            class_label = 'PG' if confidence < 0.5 else 'CEX'

            # Print the result for the image
            print(f"File: {filename} - Prediction: {class_label}, Confidence: {confidence:.4f}")


except Exception as e:
    print(f"An error occurred: {e}")

