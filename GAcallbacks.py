# The Georgia project.
# GAcallbacks.py 
#
# This file contains callbacks for model created in GAmain.py.
#
# def standard_early_stopping
# class ClassWiseMetricsCallback
# class CustomEarlyStoppingF1Callback
# class MetricsLoggerCallback
#
# To do.
# (nothing)
# #############################################################################################

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as keras_backend

standard_early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=50,         # Number of epochs with no improvement after which training will stop
    min_delta=0.001,     # Minimum change to qualify as an improvement
    restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored metric
    verbose=1            # Print messages when stopping
)


# classes.
class ClassWiseMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_generator, batch_size, report_frequency=10):
        super().__init__()
        # flow_from_directory returns the DirectoryIterator object.
        self.val_generator = val_generator
        self.batch_size = batch_size
        self.report_frequency = report_frequency

    def on_epoch_end(self, epoch, logs=None):
        # Clear unused memory before running class-wise metrics
        keras_backend.clear_session()

        # Only run the report every `report_frequency` epochs
        if (epoch + 1) % self.report_frequency == 0:
            # Get true labels and predictions for the validation dataset
            val_steps = self.val_generator.samples // self.batch_size
            y_true = []
            y_pred = []

            for step in range(val_steps):
                x_batch, y_batch = next(self.val_generator)
                y_true.extend(y_batch)
                predictions = self.model.predict(x_batch)
                y_pred.extend(np.round(predictions))  # Convert probabilities to binary predictions

            # Generate classification report
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            report = classification_report(
                y_true,
                y_pred,
                target_names=['PG', 'CEX'],  # Adjust class names as needed
                output_dict=True
            )

            # Log metrics for both classes
            print(f"\r\nEpoch {epoch + 1} - Class-Wise Metrics:")
            for class_name, metrics in report.items():
                if isinstance(metrics, dict):
                    print(
                        f"{class_name} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1-score']:.4f}")


class CustomEarlyStoppingF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, val_generator, batch_size, patience=3, f1_threshold=0.99):
        super().__init__()
        self.val_generator = val_generator
        self.batch_size = batch_size
        self.patience = patience
        self.f1_threshold = f1_threshold
        self.wait = 0  # Counts epochs that meet criteria
        self.best_f1 = 0.0  # Tracks best macro F1-score

    def on_epoch_end(self, epoch, logs=None):
        # Runs classification metrics and checks if F1-score exceeds the threshold.
        keras_backend.clear_session()

        # classification metric computation on CPU
        with tf.device('/CPU:0'):
            val_steps = self.val_generator.samples // self.batch_size
            y_true = []
            y_pred = []

            for step in range(val_steps):
                x_batch, y_batch = next(self.val_generator)
                y_true.extend(y_batch)
                predictions = self.model.predict(x_batch)  # Still runs on GPU
                y_pred.extend(np.round(predictions))  # Convert probabilities to binary predictions

            # Convert lists to numpy arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            report = classification_report(y_true, y_pred, target_names=['PG', 'CEX'], output_dict=True)

        # Print F1 scores
        if 'PG' in report and 'CEX' in report:
            try:
                f1_pg = float(report['PG'].get('f1-score', 0))
                f1_cex = float(report['CEX'].get('f1-score', 0))
                macro_f1 = (f1_pg + f1_cex) / 2
                print(f"Macro F1 Score: {macro_f1:.4f}")
                # Check if all F1-scores exceed the threshold
                if f1_pg >= self.f1_threshold and f1_cex >= self.f1_threshold:
                    print(
                        f"All F1-scores ≥ {self.f1_threshold * 100:.2f}% for {self.patience} epochs. Considering early stopping...")
                    # If F1-scores meet the threshold for multiple epochs, stop training
                    if macro_f1 > self.best_f1:
                        self.best_f1 = macro_f1  # Update best F1-score
                        self.wait = 0  # Reset patience counter
                    else:
                        self.wait += 1  # Increment patience counter

                    if self.wait >= self.patience:
                        print(
                            f"Early stopping triggered: F1-score ≥ {self.f1_threshold * 100:.2f}% for {self.patience} epochs.")
                        self.model.stop_training = True  # Stop training
                else:
                    self.wait = 0  # Reset patience counter if F1 dips below threshold

            except (ValueError, TypeError) as e:
                print(f"Error: {e}, PG F1: {report['PG'].get('f1-score')}, CEX F1: {report['CEX'].get('f1-score')}")
        else:
            print("PG or CEX class missing in the report")


class MetricsLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.learning_rates = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_losses = []
        self.epochs = []

    def on_epoch_end(self, epoch, logs=None):
        # Log the current learning rate
        current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        self.learning_rates.append(current_lr)

        # Log training accuracy
        if 'accuracy' in logs:
            self.train_accuracies.append(logs['accuracy'])

        # Log validation accuracy
        if 'val_accuracy' in logs:
            self.val_accuracies.append(logs['val_accuracy'])

        # Log validation loss
        if 'val_loss' in logs:
            self.val_losses.append(logs['val_loss'])

        # Log the current epoch
        self.epochs.append(epoch + 1)
