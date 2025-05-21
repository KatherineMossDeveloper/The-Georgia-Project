# The Georgia project on https://github.com/KatherineMossDeveloper/The-Georgia-Project/tree/main
# GAdata.py
#
# This file contains a class that creates the data objects for the model.  It will apply ResNet-specific
# preprocessing to all three types of data object types:  training, validation, and testing.  The mods are...
# scaling:  rescaled from the [0, 255] range (default for 8-bit RGB images) to the range [-1, 1].
# mean subtraction:  subtract ImageNet average color values.  red, 123.68; green, 116.779; blue, 103.939.
# Note that the model in GAmodel is resnet101, but the preprocess_input is from resnet50 because   
# the pre-processing is identical, and keras did not create one for the resnet101.
#
# check_generator           # write the data object details to screen, in order to check them.
# get_training_data         # prepare the training data and report classes.
# get_validation_data       # prepare the validation data.
# get_test_data             # prepare the test data.
#
# To do.
# (nothing)
# #############################################################################################

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input


class DataObjectGeneration:
    def __init__(self, target=None, batch=24, random=42):
        self.target_size = target
        self.batch_size = batch
        self.random_seed = random

    # write the data object details to screen, in order to check them.
    @staticmethod
    def check_generator(generator, name="Generator"):
        # Print information about an ImageDataGenerator object.
        if generator is None:
            print(f"ERROR: {name} is undefined.")
            return

        try:
            print(f" {name} Info:")
            print(f"   Target size: {generator.target_size}")
            print(f"   Batch size: {generator.batch_size}")
            print(f"   Total samples: {generator.samples}")
            print(f"   Class mode: {generator.class_mode}")
            print(f"   Number of batches per epoch: {len(generator)}")

            # Fetch a batch to verify generator works
            x_batch, y_batch = next(generator)
            print(f"     - First batch shape: X {x_batch.shape}, Y {y_batch.shape}")

        except Exception as e:
            print(f"ERROR while checking {name}: {e}")

    # prepare the training data.
    def get_training_data(self, train_dir):
        # Load training data
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
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='binary',
            seed=self.random_seed,
            shuffle=True
        )
        self.check_generator(train_generator, "Train Generator")

        # Check the classes.
        print(f'Training classes:  {train_generator.class_indices}')

        return train_generator

    # prepare the validation data.
    def get_validation_data(self, val_dir):
        # Load the validation data.
        # There are no transforms because we want to valid with 'real data.'
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='binary',
            seed=self.random_seed,
            shuffle=False
        )
        self.check_generator(val_generator, "Validation Generator")

        return val_generator

    # prepare the test data.
    def get_test_data(self, test_dir):
        # Load the test data.
        # The only transform is because we want to test with 'real data.'
        print(f'test directory {test_dir}')
        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.target_size,
            batch_size=1,
            class_mode="binary",
            seed=self.random_seed,
            shuffle=False
        )
        self.check_generator(test_generator, "Test Generator")

        return test_generator
