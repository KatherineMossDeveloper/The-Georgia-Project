# The Georgia project on https://github.com/KatherineMossDeveloper/The-Georgia-Project/tree/main
# GAmodel.py 
#
# This file, GAmodel.py, gathers datasets, trains, and calls for analysis.
#
# GAmain instantiates the class ModelTrainer, which calls preliminaties function.
# GAmain then calls the class's run function.
# The run function determine whether a CPU or GPU will be used.
# The run function then calls train function.
# The train function calls GAutility to set up the data objects.
# The train function calls create_model to create the model and load weights.
# The train function calls report after training to create files for the metrics.
#
# To do.
# (nothing)
# #############################################################################################
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model

from GAcallbacks import standard_early_stopping, ClassWiseMetricsCallback, MetricsLoggerCallback, CustomEarlyStoppingF1Callback
from GAanalysis import AnalysisConfig
from GAutility import print_model_details
from GAdata import DataObjectGeneration


class ModelTrainer:

    def __init__(self, initial_weights, builtin="n", use_cpu_only=True, epochs_total=10,
                 train_dir="", val_dir="", test_dir="", deliverables_dir="unspecified",
                 batch=64, images=(224, 224), seed=42, learning=1E-2,
                 study_name="StudyNameNotSpecified", prefix=""):
        self.weights = initial_weights
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

    def preliminaries(self):
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        print(f'*** This is study {self.study_name} ***')
        return

    def train(self):
        # Load train, valid, and test data, define their ImageDataGenerators, create the image transforms for the images.
        data_generator = DataObjectGeneration(self.target_size, self.batch, self.seed)  # variables shared for the image objects.
        train_generator = data_generator.get_training_data(self.train_dir)              # prepare the training data.
        val_generator = data_generator.get_validation_data(self.val_dir)                # prepare the validation data.
        test_generator = data_generator.get_test_data(self.test_dir)                    # prepare the test data.

        # Create the resnet model, load the weights, and designate trainable layers.
        base_model, x = self.create_model()

        # set up the callbacks.
        early_stopping_callback = standard_early_stopping
        class_wise_metrics_callback = ClassWiseMetricsCallback(val_generator=val_generator, batch_size=self.batch, report_frequency=1)
        custom_early_stopping_callback = CustomEarlyStoppingF1Callback(val_generator=val_generator, batch_size=self.batch, patience=0, f1_threshold=0.997)
        metrics_logger_callback = MetricsLoggerCallback()

        # create the model.
        model = Model(inputs=base_model.input, outputs=x)
        learning = self.learning_rate
        sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=learning, momentum=0.9, nesterov=False)
        # note that validation accuracy ("val_accuracy") is automatically tracked by Keras.
        model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        print_model_details(model)

        # train the model
        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // self.batch,
            validation_data=val_generator,
            validation_steps=val_generator.samples // self.batch,
            epochs=self.epochs,
            callbacks=[early_stopping_callback, class_wise_metrics_callback, custom_early_stopping_callback, metrics_logger_callback]
        )

        self.report(model, test_generator, metrics_logger_callback)

        return metrics_logger_callback

    def create_model(self):
        # Load pre-built weights, if any are requested.
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
        x = GlobalAveragePooling2D()(x)       # Converts feature maps to a vector
        x = Dense(512, activation='relu')(x)  # Reduced size
        x = BatchNormalization()(x)           # Normalizes activations
        x = Dropout(0.4)(x)                   # Prevents overfitting
        x = Dense(256, activation='relu')(x)  # Smaller Dense layer
        x = Dropout(0.3)(x)                   # Another dropout (optional)
        x = Dense(1, activation='sigmoid')(x)  # Output for binary classification
        print(f"The model is created. ")

        # Freeze the first layers; unfreeze the last % of layers
        total_layers = len(base_model.layers)
        num_trainable = total_layers // 10  # 10% of layers to be trainable
        for layer in base_model.layers[:total_layers - num_trainable]:
            layer.trainable = False
        for layer in base_model.layers[-num_trainable:]:
            layer.trainable = True
        print(f'The number of model layers that are now trainable {num_trainable}')

        return base_model, x

    def report(self, model, test_generator, metrics_logger_callback):
        print(f"Ending training.  Deliverables will be saved in {self.deliverables_dir}.")
        analysis = AnalysisConfig(name=self.study_name, prefix=self.prefixed_letters,
                                  folder=self.deliverables_dir, classes=['PG', 'CEX'])
        analysis.generate_deliverables(model, test_generator, metrics_logger_callback)

    def run(self):
        #  credit to https://www.geeksforgeeks.org/how-to-run-tensorflow-on-cpu/ for this code.
        #  scenario 1  use_cpu = true, and there is no GPU.
        #  scenario 2  use_cpu = true, even though there is a GPU.
        #  scenario 3  use_cpu = false, but there is no GPU.
        #  scenario 4  use_cpu = false, even though there is a GPU.

        # Check if a GPU is available
        gpu_devices = tf.config.list_physical_devices('GPU')

        if self.use_cpu or not gpu_devices:               # scenario 1, 2, and 3
            with tf.device('/CPU:0'):
                print("Tensorflow is using the CPU for training")
        else:                                             # scenario 4
            print(f"TensorFlow is using the following GPU(s): {gpu_devices}")

        logger = self.train()

        return logger
