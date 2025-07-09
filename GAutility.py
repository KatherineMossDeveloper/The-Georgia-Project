# The Georgia project on https://github.com/KatherineMossDeveloper/The-Georgia-Project/tree/main
# GAutility.py
#
# This file contains various functions and classes for the project.
# load_and_preprocess_image # loads an image, converts to np array, then does resnet preprocess.
# print_elapsed_time        # report how long training the model took.
# print_model_details       # print the number of trainable and non-trainable layers.
# get_color                 # returns normalized RGB values for plots.
#
# To do.
# (nothing)
# #############################################################################################

from datetime import datetime
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input


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


# report how long training the model took.
def print_elapsed_time(start_time):
    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))


# print the number of trainable and non-trainable layers.
def print_model_details(model):
    # Count the number of trainable layers
    trainable_layers = sum(1 for layer in model.layers if layer.trainable)

    # Count the number of not trainable layers
    not_trainable_layers = sum(1 for layer in model.layers if not layer.trainable)

    # Print the results
    print(f"--->Number of trainable layers: {trainable_layers}")
    print(f"--->Number of not trainable layers: {not_trainable_layers}")


def get_color(r, g, b):
    normalized_rgb = (r / 255, g / 255, b / 255)  # Normalized RGB values
    return normalized_rgb
