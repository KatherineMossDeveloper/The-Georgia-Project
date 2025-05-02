# The Georgia project.
# GAutility.py   version 1
#
# This file contains various functions and classes code for the project.
#
# print_elapsed_time        # report how long training the model took.
# setup_shared_variables    # variables shared to make the image objects.
# check_generator           # write the data object details to screen, in order to check them.
# get_training_data         # prepare the training data.
# get_validation_data       # prepare the validation data.
# get_test_data             # prepare the test data.
# print_model_details       # print the number of trainable and non-trainable layers.
# get_color                 # returns normalized RGB values for plots.
#
# To do.
# (nothing)
# #############################################################################################

from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

target_size = ""
batch_size = ""
random_seed = ""


#  functions.
# report how long training the model took.
def print_elapsed_time(start_time):
    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))


# variables shared to make the image objects.
def setup_shared_variables_utility(target, batch, seed):
    global target_size
    target_size = target
    global batch_size
    batch_size = batch
    global random_seed
    random_seed = seed


# write the data object details to screen, in order to check them.
def check_generator(generator, name="Generator"):
    """
    Prints key information about an ImageDataGenerator object.

    Args:
        generator: The ImageDataGenerator (train/val/test).
        name: The name of the generator (for identification).
    """
    if generator is None:
        print(f"ERROR: {name} is None!")
        return

    try:
        print(f"---> {name} Info:")
        print(f"     - Target size: {generator.target_size}")
        print(f"     - Batch size: {generator.batch_size}")
        print(f"     - Total samples: {generator.samples}")
        print(f"     - Class mode: {generator.class_mode}")
        print(f"     - Number of batches per epoch: {len(generator)}")

        # Fetch a batch to verify generator works
        x_batch, y_batch = next(generator)
        print(f"     - First batch shape: X {x_batch.shape}, Y {y_batch.shape}")

    except Exception as e:
        print(f"ERROR while checking {name}: {e}")


# prepare the training data.
def get_training_data(train_dir):
    # Load training data
    # Define ImageDataGenerators
    # create the image transforms for the images.
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        seed=random_seed,
        shuffle=True
    )
    check_generator(train_generator, "Train Generator")

    return train_generator


# prepare the validation data.
def get_validation_data(val_dir):
    # create the image transforms for the images.
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        seed=random_seed,
        shuffle=False
    )
    check_generator(val_generator, "Validation Generator")

    return val_generator


# prepare the test data.
def get_test_data(test_dir):
    # Test ImageDataGenerator
    print(f'test directory {test_dir}')
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=1,
        class_mode="binary",
        seed=random_seed,
        shuffle=False
    )
    check_generator(test_generator, "Test Generator")

    return test_generator


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

