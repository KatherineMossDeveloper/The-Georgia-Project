# The Georgia project on https://github.com/KatherineMossDeveloper/The-Georgia-Project/tree/main
# GAutility.py 
#
# This file contains various functions and classes code for the project.
#
# print_elapsed_time        # report how long training the model took.
# print_model_details       # print the number of trainable and non-trainable layers.
# get_color                 # returns normalized RGB values for plots.
#
# To do.
# (nothing)
# #############################################################################################

from datetime import datetime


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
