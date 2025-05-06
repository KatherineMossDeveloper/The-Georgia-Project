# The Georgia project.
# GAmodel.py 
#
# This file, GAmodel.py, gathers datasets, trains, and calls for analysis.
#
# class ModelTrainer
#
# To do.
# (nothing)
# #############################################################################################
import os
import numpy as np
import psutil
import GPUtil
import tensorflow as tf

from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model

from GAcallbacks import standard_early_stopping, ClassWiseMetricsCallback, MetricsLoggerCallback, CustomEarlyStoppingF1Callback
from GAanalysis import analyze, setup_shared_variables_analysis
from GAutility import setup_shared_variables_utility, print_model_details, get_training_data, get_validation_data, get_test_data


class ModelTrainer:

    def __init__(self, initial_weights, builtin="n", use_cpu_only=True, epochs_total=10,
                 train_dir="", val_dir="", test_dir="", deliverables_dir="unspecified",
                 batch=64, images=(224, 224), seed=42, learning=1E-2,
                 study_name="StudyNameNotSpecified", prefix=""):
        self.builtin_weights = builtin
        self.use_cpu = use_cpu_only
        self.epochs = epochs_total
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.deliverables_dir = deliverables_dir
        self.batch = batch
        self.target_size = images
        self.seed = seed
        self.learning_rate = learning
        self.study_name = study_name
        self.prefixed_letters = prefix
        self.preliminaries()
        self.weights = initial_weights

    def preliminaries(self):
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        print(f'*** This is study {self.study_name} ***')
        return

    def train(self):
        # Load training data, define ImageDataGenerators, create the image transforms for the images.
        # Create the custom generator for training
        setup_shared_variables_utility(self.target_size, self.batch, self.seed)  # variables shared to make the image objects.
        train_generator = get_training_data(self.train_dir)              # prepare the training data.
        val_generator = get_validation_data(self.val_dir)                # prepare the validation data.
        test_generator = get_test_data(self.test_dir)                    # prepare the test data.

        # Load pre-built weights, if any are handed in here.
        if self.weights == "":
            base_model = ResNet101(weights=None, include_top=False, input_shape=(224, 224, 3))
            print(f"---> using no weights {self.weights}")

        elif self.builtin_weights == "y":
            base_model = ResNet101(weights=self.weights, include_top=False, input_shape=(224, 224, 3))
            print(f"---> using built-in weights {self.weights}")

        else:
            base_model = ResNet101(weights=None, include_top=False, input_shape=(224, 224, 3))
            base_model.load_weights(self.weights, by_name=True, skip_mismatch=True)
            print(f"---> using custom local weights {self.weights}")

        x = base_model.output
        x = GlobalAveragePooling2D()(x)  # Converts feature maps to a vector
        x = Dense(512, activation='relu')(x)  # Reduced size
        x = BatchNormalization()(x)  # Normalizes activations
        x = Dropout(0.4)(x)  # Prevents overfitting
        x = Dense(256, activation='relu')(x)  # Smaller Dense layer
        x = Dropout(0.3)(x)  # Another dropout (optional)
        x = Dense(1, activation='sigmoid')(x)  # Output for binary classification

        print(f"Set up the ResNet model. ")

        # Freeze the first layers; unfreeze the last % of layers
        total_layers = len(base_model.layers)
        num_trainable = total_layers // 10  # 10% of layers to be trainable
        for layer in base_model.layers[:total_layers - num_trainable]:
            layer.trainable = False
        for layer in base_model.layers[-num_trainable:]:
            layer.trainable = True
        print(f'number of model layers that are now trainable {num_trainable}')

        # set up the callbacks.
        early_stopping_callback = standard_early_stopping

        class_wise_metrics_callback = ClassWiseMetricsCallback(val_generator=val_generator, batch_size=self.batch,
                                                               report_frequency=1)
        custom_early_stopping_callback = CustomEarlyStoppingF1Callback(val_generator=val_generator, batch_size=self.batch,
                                                                       patience=0, f1_threshold=0.997)
        metrics_logger_callback = MetricsLoggerCallback()

        # create the model.
        model = Model(inputs=base_model.input, outputs=x)
        # model.summary()
        learning = self.learning_rate

        sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=learning, momentum=0.9, nesterov=False)

        # note that validation accuracy ("val_accuracy") is automatically tracked by Keras.
        model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # train the model
        print_model_details(model)
        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // self.batch,
            validation_data=val_generator,
            validation_steps=val_generator.samples // self.batch,
            epochs=self.epochs,
            callbacks=[early_stopping_callback, class_wise_metrics_callback, custom_early_stopping_callback, metrics_logger_callback]
        )

        # report.
        print(f"Ending training.  Deliverables will be saved in {self.deliverables_dir}.")
        setup_shared_variables_analysis(name=self.study_name, prefix=self.prefixed_letters,
                                        folder=self.deliverables_dir, classes=['PG', 'CEX'])
        analyze(model, test_generator, metrics_logger_callback)

        return metrics_logger_callback

    def run(self):
        if self.use_cpu:
            # Disable GPU to force CPU training
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            with tf.device('/CPU:0'):
                print("Using CPU for training")
                logger = self.train()
        else:
            # Set GPU before TensorFlow is initialized
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Adjust if using multiple GPUs
            print("Using GPU for training")
            print("CPU core count:", psutil.cpu_count(logical=False))
            print("Total memory:", psutil.virtual_memory().total)

            if GPUtil:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    print(f"GPU: {gpu.name}, RAM: {gpu.memoryTotal}MB")
            else:
                print("GPUtil not installed, skipping GPU details.")

            logger = self.train()

        return logger
