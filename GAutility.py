# The Georgia project on https://github.com/KatherineMossDeveloper/The-Georgia-Project/tree/main
# GAutility.py
#
# This file contains various color objects and functions for the project.
#
# color_palette_small        hand full of colors, in order to control color scheme with
#                            small data collections.
# color_palette_large        many colors for large data collections.
# get_model                  creates a ResNet101 model for both training and CAM overlays.
# load_and_preprocess_image  loads an image, converts to np array, then does resnet preprocess.
# print_elapsed_time         report how long training the model took.
# print_model_details        print the number of trainable and non-trainable layers.
# get_color                  returns normalized RGB values for plots.
# get_plot_color_objects     returns a colormap for matplotlib & d3blocks, plus a matplotlib legend
# get_cam_color_scheme
#
# To do.
# (nothing)
# #############################################################################################

import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

color_palette_small = [
    '#008200',  # olive green
    '#800080',  # dark purple
    '#0000FF',  # blue blue
    '#FFAC00'   # marigold
]

color_palette_large = [
    '#0099FF', '#CC0099', '#00CC80', '#6643b5', '#009999',
    '#9900CC', '#0000CC', '#000099', '#00CCCC', '#CC00CC',
    '#FF6600', '#00FF66', '#FF0066', '#6600FF', '#FFCC00',
    '#3366FF', '#66FF66', '#FF9999', '#9966FF', '#66CCCC',
    '#FF3333', '#99CC00', '#00CCFF', '#CCFF00', '#003366',
    '#660066', '#CCCC00', '#FFCC99'
]


# Function to create a ResNet101 model.
def get_model(weights=None):

    base_model = tf.keras.applications.ResNet101(
        weights=weights,
        include_top=False,
        input_shape=(224, 224, 3)
    )

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu", name="dense")(x)
    x = tf.keras.layers.BatchNormalization(name="batch_normalization")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="dense_2")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=out)

    return base_model, model


# Function to load, resize, and preprocess an image
def load_and_preprocess_image(img_path, target_size=(224, 224)):

    # Load and resize the image to the target size (224, 224)
    img = image.load_img(img_path, target_size=target_size)

    # Convert the image to a NumPy array
    image_array = image.img_to_array(img)

    # Add batch dimension (the model expects a batch of images, not just one)
    image_array = np.expand_dims(image_array, axis=0)  # Shape: [1, 224, 224, 3]

    # Apply ResNet-specific preprocessing.  Scaling & mean sub. can make training faster and more stable.
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


def get_plot_color_objects(entries_to_map, clusters):

    dot_colors = []
    legend_entries = []

    if clusters > len(color_palette_small):
        print(f"Error in get_colors_and_legend:  only {len(color_palette_small)} "
              f"unique colors are defined, but {clusters} clusters were requested.")
    else:
        selected_colors = color_palette_small[:clusters]

        # map cluster labels to colors
        label_to_color = {i: selected_colors[i] for i in range(clusters)}
        dot_colors = [label_to_color[label] for label in entries_to_map]

        # color the legend entries to correspond to the dots on the plot.
        legend_entries = [
            mpatches.Patch(color=selected_colors[i], label=f'Cluster {i + 1}')
            for i in range(clusters)
        ]

    return dot_colors, legend_entries


def get_cam_color_scheme(color_scheme):

    # gnuplot color scheme goes from purple to red to yellow; it comes as values between 0 and 1.
    # so we convert that to values between 0 and 255 because we will apply them to RGB values per pixel.
    base_image = plt.get_cmap(color_scheme)(np.linspace(0, 1, 256))
    # convert these values into a custom color map.
    color_map = ListedColormap(base_image)

    return color_map

