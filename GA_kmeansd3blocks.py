# The Georgia project on https://github.com/KatherineMossDeveloper/The-Georgia-Project/tree/main
# GA_kmeansd3blocks.py
#
# This file contains code that graphs images in PCA space using D3Blocks scatter plot.
#
# Code flow.
#    kmeansd3blocks_driver(data_class)
#
# Navigation when the plot is in the browser.
# Hover for tooltip	   Mouse only
# Zoom	               Mouse wheel
# Reset	               Reload the page
#
# To do.
# (nothing)
# #############################################################################################
import os
import numpy as np
from d3blocks import D3Blocks
from GA_dataprocessing import add_note


def kmeansd3blocks_driver(data_class):

    try:
        # Extract features for each image in the folder
        print(f'Starting GAkmeansd3blocks.py')
        data_class.colors = np.array(data_class.colors).astype(str)

        # Initialize
        d3 = D3Blocks()
        full_file_path = os.path.join(data_class.image_folder, 'GAkmeansd3blocks.html')
        print(f'Inside kmeansd3blocks, saving plot as {full_file_path}')

        # Create scatter plot and save it to the filepath.
        d3.scatter(
            data_class.features_reduced[:, 0],
            data_class.features_reduced[:, 1],
            size=15,
            color=data_class.colors,
            stroke='#000000',
            opacity=0.4,
            tooltip=data_class.tooltips,
            filepath=full_file_path,
            title="PCA of some OpenCrystalData data with kmeans colors. "
        )

        add_note(full_file_path, "OpenCrystalData images represented in PCA space.")

    except Exception as e:
        print(f"An error occurred in GA_kmeansd3blocks.kmeansd3blocks_driver: {e}")
