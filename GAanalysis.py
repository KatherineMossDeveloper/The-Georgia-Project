# The Georgia project on https://github.com/KatherineMossDeveloper/The-Georgia-Project/tree/main
# GAcallbacks.py 
#
# This file contains post-training analysis for GAmodel.py.  All the files generated will be in
# the deliverables folder, which is designated in the GAmain.py file.  The graphs created
# will not pop up as windows, but you can change the code to do that.
#
#  class AnalysisConfig                              shared variables, like deliverables folder.
#  def generate_deliverables                         drive the creation of the deliverables.
#  def save_model_to_disk                            save model to h5 and onnx formats.
#  def metrics_plot                                  plots accuracies, etc.
#  def test_eval                                     class wise breakdown of test metrics
#  def confusion                                     create a confusion matrix
#
# To do.
# (nothing)
# #############################################################################################

import os
import gc
import onnx
import tf2onnx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
from GAutility import get_color

# create colors for the plots. 
background_color = get_color(210, 220, 230)  # light blue-gray
plot_color = get_color(250, 250, 250)        # off-white


# This class will set hold the shared variables for all the deliverables
# which are generated after the training is completed in GAmodel.py.  The
# generate_deliverables function will call 4 other functions that save the
# model to disk, create plots and text files to capture the metrics when testing
# the model.
class AnalysisConfig:
    def __init__(self, name="unspecified", prefix="", folder="", classes=None):

        if classes is None:
            classes = ['PG', 'CEX']
        self.study_name = name
        self.prefix = prefix
        self.folder = folder
        self.classes = classes

    def generate_deliverables(self, model, test_generator, lr_logger):

        save_model_to_disk(model, self.prefix, self.folder)    # save the model.
        metrics_plot(lr_logger, self.study_name, self.folder, self.prefix)  # plot metrics.
        test_eval(model, test_generator, self.folder, self.prefix, self.study_name)  # print the test results.
        confusion(model, test_generator, self.classes, self.study_name, self.folder, self.prefix)  # draw confusion matrix.


# This function will create two files for the weights.
# one in the H5 format and the other in the onnx format.
def save_model_to_disk(model, prefix_name, deliverables_folder):
    gc.collect()  # Force garbage collection
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Create a timestamp.
    filename = f"{prefix_name}weights_{timestamp}"            # Create a unique file name.

    # Define full file path
    file_path = os.path.join(deliverables_folder, filename)

    # Save in HDF5 format; ensure the folder with its full path exists
    os.makedirs(deliverables_folder, exist_ok=True)
    model.save(file_path + ".h5", save_format='h5')

    # Convert to ONNX format and save that.
    onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)
    onnx.save(onnx_model, file_path + ".onnx")

    print(f"Saved models in formats h5 and onnx to {deliverables_folder}.")


# This function will create a plot of two metrics.  For each epoch,
# the plot will show the training and validation accuracies.
def metrics_plot(lr_logger, study_name, deliverables_folder, prefix_name):
    # Plot training and validation accuracy
    fig, ax1 = plt.subplots()

    # Set colors and add title.
    fig.patch.set_facecolor(background_color)
    ax1.set_facecolor(plot_color)
    plt.title(f"{study_name} Training and validation accuracies")

    # Training Accuracy (Primary Y-Axis)
    ax1.plot(lr_logger.epochs, lr_logger.train_accuracies, label='Training Accuracy',
             color='purple', marker='o', markersize=3)

    # Validation Accuracy (Secondary Y-Axis)
    ax1.plot(lr_logger.epochs, lr_logger.val_accuracies, label='Validation Accuracy',
             color='blue', marker='o', markersize=3)

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Set X-axis labels as integers
    ax1.set_xticks(lr_logger.epochs)  # Set tick positions at each epoch
    ax1.set_xticklabels([int(epoch) for epoch in lr_logger.epochs])  # Convert tick labels to integers

    # Save plot; ensure the folder with its full path exists
    os.makedirs(deliverables_folder, exist_ok=True)
    plt.savefig(f"{deliverables_folder}/{prefix_name}metrics_plot.png")

    # Show the plot, if needed
    # plt.show()


# This function will create a text file with metrics
# broken down by classes PG and CEX.  Here is an example...
#               precision    recall  f1-score   support
#
#           PG       1.00      1.00      1.00       343
#          CEX       1.00      1.00      1.00       341
#     accuracy                           1.00       684
#    macro avg       1.00      1.00      1.00       684
# weighted avg       1.00      1.00      1.00       684
def test_eval(model, test_gen, deliverables_folder, prefix_name, study_name):

    # Run final evaluation on test set
    print(f'--->TestEvalFinal starting.')
    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"\n Test Set Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

    # Run class-wise metrics on test set
    y_true, y_pred = [], []

    for step in range(test_gen.samples // test_gen.batch_size):
        x_batch, y_batch = next(test_gen)
        y_true.extend(y_batch)
        predictions = model.predict(x_batch)
        y_pred.extend(np.round(predictions))

    report = classification_report(y_true, y_pred, target_names=['PG', 'CEX'])

    # send it to the output window
    print("\nFinal Test Set Classification Report:\n", report)

    # send it to a file; ensure the folder with its full path exists
    os.makedirs(deliverables_folder, exist_ok=True)
    file_path = f'{deliverables_folder}/{prefix_name}FinalTestResults.txt'
    with open(file_path, 'w') as file:
        file.write(f"\n{study_name} Test set classification:\n")
        file.write(report)


# This function will create a plot showing how many correct and incorrect
# labels that the model predicted for the two classes PG and CEX.
def confusion(model, test_gen, class_names, study_name, deliverables_folder, prefix_name ):
    # Generates a confusion matrix at the end of training.
    print("\nGenerating Final Confusion Matrix...")

    # Get true labels and predictions
    y_true, y_pred = [], []

    for step in range(test_gen.samples // test_gen.batch_size):
        x_batch, y_batch = next(test_gen)

        # Convert y_batch to a flat list
        y_true_list = np.argmax(y_batch, axis=1).tolist() if y_batch.ndim > 1 else y_batch.tolist()
        y_true.extend(y_true_list)  # Ensures we are extending a list, not a ndarray

        # Get predictions and flatten them
        predictions = model.predict(x_batch, verbose=0)
        y_pred_list = np.round(predictions).astype(int).flatten().tolist()
        y_pred.extend(y_pred_list)  # Ensures we are extending a list, not a ndarray

    # Convert to numpy arrays
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    fig = plt.figure(figsize=(6, 5))
    fig.patch.set_facecolor(background_color)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{study_name} Confusion Matrix")

    # Save plot; ensure the folder with its full path exists
    os.makedirs(deliverables_folder, exist_ok=True)
    plt.savefig(f"{deliverables_folder}/{prefix_name}final_confusion_matrix.png")

    # Show the plot, if needed
    # plt.show()
