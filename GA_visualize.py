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
# To do.
# Edit the folder_prefix variable to point to the Georgia Project code on your pc.
# Save the weights file downloaded from the Georgia Project on GitHub to the \inference
# folder, or you can use the weights file that you created after training the model.
# If you created you own weights file, its name will include a date and time stamp,
# so change the weights_file variable accordingly.
# #############################################################################################

from GA_dataprocessing import DataProcessor
from GA_kmeansmatplotlib import kmeansmatplotlib_driver
from GA_kmeansd3blocks import kmeansd3blocks_driver
from GA_similarityd3blocks import similarityd3blocks_driver

# step 0.  set up the path to your image folder and weights file
folder_prefix = r"C:\Users\mossr\PycharmProjects\pythonProject\TestProject\work\Transformer\The-Georgia-Project-1.5.6\The-Georgia-Project-1.5.6"  # edit this before running the code.
# C:\Users\mossr\PycharmProjects\pythonProject\TestProject\work\Transformer\The-Georgia-Project-1.5.6\The-Georgia-Project-1.5.6

# for the weights file...
weights_folder = folder_prefix + r"\images_testing\GAweights.h5"

# for a few images...
image_folder = folder_prefix + r"\images_testing"

# for a lot of images...
# image_folder = f"X:/MLresearch/CrystalStudy/Project_GA/data/GAvalidKmeanstest"  # 3k files

data_class = DataProcessor(image_folder, weights_folder, mod=1)
data_class.setup_data()

# step 2.  graph the k-means, pca values with matplotlib, using the feature model and PCA.
kmeansmatplotlib_driver(data_class)

# step 3.  graph the k-means, pca values in D3blocks scatter plot.
kmeansd3blocks_driver(data_class)

# step 4.  graph similarity in D3blocks d3graph plot.
similarityd3blocks_driver(data_class, limit=10000)
