# The Georgia project on https://github.com/KatherineMossDeveloper/The-Georgia-Project/tree/main
# GA_visualize.py
#
# 1.  This code will call GA_dataprocessing.py to report its confidence, as a percentage, that
#     a given image in the designated folder is PG, phenylglycine.  The confidence
#     percent for each image will be in the output window.
#
# 2.  This code will then run GA_kmeansmatplotlib.py on the same images.  The code will then create
#     a Kmeans plot and save it to disk in the same 'kmeans' folder.  It will also pop
#     up a graph showing how the images cluster.
#
# 3.  This code will then run GA_kmeansd3blocks.py using the same kmeans centroids and pca
#     coordinates.  The code will create a plot with the D3Blocks scatter plot.
#
# 4.  This code will then run GA_similarityd3blocks.py using the weaviate database and its
#     default similarity search, if available.  The code will create a plot with the D3Blocks d3graph plot.
#
# 5.  This code will then run GA_camoverlays.py using the original images. The classification activation
#     images will be created.  Finally, the cam images overlaid on the original images are created.
#
# To do.
# Edit the folder_prefix variable to point to the Georgia Project code on your pc.
# Do the same for the classification activation folders, if needed.
# Save the weights file downloaded from the Georgia Project on GitHub to the \inference
# folder, or you can use the weights file that you created after training the model.
# If you created you own weights file, its name will include a date and time stamp,
# so change the weights_file variable accordingly.
# #############################################################################################

from pathlib import Path
from GA_dataprocessing import DataProcessor
from GA_kmeansmatplotlib import kmeansmatplotlib_driver
from GA_kmeansd3blocks import kmeansd3blocks_driver
from GA_similarityd3blocks import similarityd3blocks_driver
from GA_camoverlays import camoverlays_driver

# step 0.  set up the path to your image folder and weights file
folder_prefix = Path("Y:/The-Georgia-Project-1.7.0")  # edit this before running the code.

# for the weights file...
weights_file = folder_prefix / "images_testing/GAweights.h5"

# for a curated subset of images...
image_folder = folder_prefix / "images_testing"

# for CAM, classification activation mapping.
base_dir = folder_prefix
orig_dir = base_dir / "images_testing"
orig_224_dir = base_dir / "images_orig224"
just_cam_dir = base_dir / "images_just_CAM"
cam_dir = base_dir / "images_overlay_CAM"
img_size = (224, 224)

# step 1.  instantiate the data processor class.
data_class = DataProcessor(image_folder, weights_file, mod=1)
data_class.setup_data()

# step 2.  graph the k-means, pca values with matplotlib, using the feature model and PCA.
kmeansmatplotlib_driver(data_class)

# step 3.  graph the k-means, pca values in D3blocks scatter plot.
kmeansd3blocks_driver(data_class)

# step 4.  graph similarity in D3blocks d3graph plot.
similarityd3blocks_driver(data_class, limit=10000)

# step 5.  generate classification activation map images.
camoverlays_driver(base_image_dir=base_dir,
                   original_images_dir=orig_dir,
                   original_images_224_dir=orig_224_dir,
                   cam_images_dir=just_cam_dir,
                   cam_overlay_images_dir=cam_dir,
                   image_size=img_size)
