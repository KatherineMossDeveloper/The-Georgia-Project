# The Georgia project on https://github.com/KatherineMossDeveloper/The-Georgia-Project/tree/main
# GAinference.py
#
# This code will pull png files from a folder and do inference on each one,
# reporting the classification and confidence to the output window.
# The prediction logic below assumes a binary classification, where the
# CEX images are in folder '0' and PG are in folder '1'.  By convention,
# '1' is the positive class in binary classification, meaning the class
# that the model is trained to predict.
#
# To do.
# Edit the folder_prefix variable to point to the Georgia Project code on your pc.
# Save the weights file downloaded from the Georgia Project on GitHub to the \inference
# folder, or you can use the weights file that you created after training the model.
# If you created you own weights file, its name will include a date and time stamp,
# so change the weights_file variable accordingly.
# #############################################################################################

import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# folder_prefix = r"your_drive_letter_and_folder"  # edit this before running the code.
folder_prefix = r"C:\Users\mossr\PycharmProjects\pythonProject\TestProject\work\Transformer\GAproject"  # edit this before running the code.

# Set up the path to your image folder and weights file
image_folder = folder_prefix + r"\inference"
weights_file = folder_prefix + r"\inference\GAweights.h5"


# Function to load, resize, and preprocess an image
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    # Load and resize the image to the target size (224, 224)
    img = image.load_img(img_path, target_size=target_size)

    # Convert the image to a NumPy array
    image_array = image.img_to_array(img)

    # Add batch dimension (the model expects a batch of images, not just one)
    image_array = np.expand_dims(image_array, axis=0)  # Shape: [1, 224, 224, 3]

    # Apply ResNet-specific preprocessing.
    # scaling:  rescaled from the [0, 255] range (default for 8-bit RGB images) to the range [-1, 1].
    # mean subtraction:  subtract ImageNet average color values.  red, 123.68; green, 116.779; blue, 103.939.
    # Did the same for testing.  See GAmodel.py get_test_data() for details.
    image_array = preprocess_input(image_array)

    return image_array


# Inference for each image in the folder
try:
    # Load the pre-trained model with weights
    print(f'Starting GAinference.py.')
    model = load_model(weights_file)
    png_files_in_folder = [f for f in os.listdir(image_folder) if f.endswith('.png')]  # the no. of files in folder.
    png_file_count = len(png_files_in_folder)

    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):  # Check if the file is a PNG file
            file_path = os.path.join(image_folder, filename)

            # Load and preprocess the image
            img_array = load_and_preprocess_image(file_path)

            raw_prediction = model.predict(img_array, verbose=0)  # Get the raw output

            # Perform inference on the image
            prediction = model.predict(img_array, verbose=0 )

            # Assuming binary classification folder structure CEX (0) & PG (1),
            # this will return the confidence that the image is of PG.
            confidence = prediction[0][0]

            # Convert to percentage
            confidence_percent = confidence * 100

            # Assign the class label with threshold 0.5.
            # Assign the class label with threshold 0.5 for PG.
            class_label = 'PG' if confidence >= 0.5 else 'CEX'  # PG if prob >= 0.5, else CEX

            # Print the result with confidence as a percentage
            print(f"File: {filename} - Prediction: {class_label}, Confidence: {confidence_percent:.2f}%")


except Exception as e:
    print(f"An error occurred: {e}")

