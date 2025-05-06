# The Georgia project.
# GAcallbacks.py 
#
# This file contains post-training analysis for GAmodel.py.
#
#  def setup_shared_variables_analysis               shared variables, like deliverables folder.
#  def analyze                                       drive the creation of the deliverables.
#  def save_model_to_disk                            save model to h5 and onnx formats.
#  def metrics_plot                                  plots accuracies, etc.
#  def test_eval                                     class wise breakdown of test metrics
#  def roc_auc                                       ROC, AUC graphic
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
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from GAutility import get_color

study_name = ""
prefix_name = ""
deliverables_folder = ""
class_names = ""

# create colors for the plots.
background_color = get_color(210, 220, 230)  # light blue-gray
plot_color = get_color(250, 250, 250)        # off-white


# variables shared to make the report the results.
def setup_shared_variables_analysis(name="unspecified", prefix="", folder="", classes=None):
    global study_name
    study_name = name
    global prefix_name
    prefix_name = prefix
    global deliverables_folder
    deliverables_folder = folder
    global class_names  # class_names=['PG', 'CEX']
    class_names = classes


# functions.
def analyze(model, test_generator, lr_logger):

    save_model_to_disk(model)         # save the model.
    metrics_plot(lr_logger)           # plot metrics.
    test_eval(model, test_generator)  # print the test results.
    roc_auc(model, test_generator)    # draw the ROC/AUC.
    confusion(model, test_generator)  # draw confusion matrix.


# create h5 and onnx versions of the model.
def save_model_to_disk(model):
    gc.collect()  # Force garbage collection
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Create filename using timestamp
    filename = f"{prefix_name}weights_{timestamp}"

    # Ensure the folder exists
    print(f'---> folder {deliverables_folder}')
    os.makedirs(deliverables_folder, exist_ok=True)

    # Define full file path
    file_path = os.path.join(deliverables_folder, filename)
    print(f'---> file path {file_path}')

    # Delete existing file if needed
    if os.path.exists(file_path):
        os.remove(file_path)

    # Save in HDF5 format
    model.save(file_path + ".h5", save_format='h5')

    # Convert to ONNX format and save that.
    onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)
    onnx.save(onnx_model, file_path + ".onnx")

    print(f"Saved model to {deliverables_folder}.")


def metrics_plot(lr_logger):
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

    # Save plot if a path is provided
    if deliverables_folder:
        plt.savefig(f"{deliverables_folder}/{prefix_name}metrics_plot.png")

    # Show the plot, if needed
    # plt.show()


def test_eval(model, test_gen):
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

    # send it to a file.
    file_path = f'{deliverables_folder}/{prefix_name}FinalTestResults.txt'
    with open(file_path, 'w') as file:
        file.write(f"\n{study_name} Test set classification:\n")
        file.write(report)


def roc_auc(model, test_gen):
    # Computes and plots the ROC/AUC for a given model and test data.
    # Args:
    #     model: Trained Keras model.
    #     test_image_generator: Test data generator (ImageDataGenerator or similar).
    print("\nGenerating ROC/AUC...")

    # Extract true labels from the test generator
    y_true = test_gen.classes  # True labels from generator
    # class_indices = list(self.test_gen.class_indices.values())  # Class mapping

    # Predict probabilities (not binary predictions)
    y_pred_proba = model.predict(test_gen, steps=len(test_gen), verbose=1)

    # If model output shape is (num_samples, 1), flatten the predictions
    if y_pred_proba.shape[1] == 1:
        y_pred_proba = y_pred_proba.ravel()

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    # Compute AUC score
    roc_auc_score = auc(fpr, tpr)

    # Plot the ROC curve
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor(background_color)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line (random model)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"{study_name} ROC/AUC")
    plt.legend(loc="lower right")

    # Save plot if a folder is provided
    if deliverables_folder:
        plt.savefig(f"{deliverables_folder}/{prefix_name}final_roc_auc.png")
        print(f"ROC/AUC plot saved at {deliverables_folder}/{prefix_name}final_roc_auc.png")

    # Show the plot, if needed
    # plt.show()

    # Open a text file for writing the thresholds used in the ROC creation.
    file_path = f'{deliverables_folder}/{prefix_name}ROC_thresholds.txt'
    print(f'printing to file {file_path}')

    with open(file_path, 'w') as f:
        # Write the headers
        f.write("Thresholds, FPR, TPR\n")
        # Iterate through the thresholds, FPR, and TPR and write them to the file
        for i in range(len(thresholds)):
            f.write(f"{thresholds[i]}, {fpr[i]}, {tpr[i]}\n")


def confusion(model, test_gen):
    # Generates a confusion matrix at the end of training.
    # Args:
    #                 model
    #                 test_generator (ImageDataGenerator): Test data generator.
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

    # Save plot if a path is provided
    if deliverables_folder:
        plt.savefig(f"{deliverables_folder}/{prefix_name}final_confusion_matrix.png")

    # Show the plot, if needed
    # plt.show()
