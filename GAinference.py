# The Georgia project on https://github.com/KatherineMossDeveloper/The-Georgia-Project/tree/main
# GAinference.py
#
# 1.  This code will call GApredict.py to report its confidence, as a percentage, that
#     a given image in the designated folder is PG, phenylglycine.  The confidence
#     percent for each image will be in the output window.
#
# 2.  This code will then run GAkmeans.py on the same images.  The code will then create
#     a Kmeans plot and save it to disk in the same 'kmeans' folder.  It will also pop
#     up a graph showing how the images cluster.
#
# To do.
# Edit the folder_prefix variable to point to the Georgia Project code on your pc.
# Save the weights file downloaded from the Georgia Project on GitHub to the \inference
# folder, or you can use the weights file that you created after training the model.
# If you created you own weights file, its name will include a date and time stamp,
# so change the weights_file variable accordingly.
# #############################################################################################

from GApredict import predict_driver
from GAkmeans import kmeans_driver
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet101

# 0.  Set up the path to your image folder and weights file
folder_prefix = r"X:/MLresearch/CrystalStudy/Project_GA/LastGoodStaging/"  # edit this before running the code.
image_folder = folder_prefix + r"\inference"
weights_file = folder_prefix + r"\inference\GAweights.h5"

# 1.  do prediction on the image files.
classifier_model = load_model(weights_file)
predict_driver(model=classifier_model, image_folder=image_folder)

# 2.  do kmeans on the image files.
image_folder = folder_prefix + r"\kmeans"
# Load the ResNet101 model without the top layers (include_top=False)
feature_model = ResNet101(weights=None, include_top=False, input_shape=(224, 224, 3))
# Load the weights from the weights file.
feature_model.load_weights(weights_file, by_name=True, skip_mismatch=True)
kmeans_driver(model=feature_model, folder_path=image_folder, num_clusters=4)
