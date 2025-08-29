# The Georgia project on https://github.com/KatherineMossDeveloper/The-Georgia-Project/tree/main
# GA_kmeansmatplotlib.py
#
# This file contains code to do K-means clustering and PCA on image files.
#
# Code flow.
#    kmeansmatplotlib_driver(data_class):
#
# To do.
# (nothing)
# #############################################################################################

import os
import matplotlib.pyplot as plt


# Visualize both PG and CEX clusters on one plot with different colors for 3 clusters
def kmeansmatplotlib_driver(data_class):

    try:

        # draw the PCA components
        plt.figure(figsize=(10, 8))
        plt.scatter(data_class.features_reduced[:, 0], data_class.features_reduced[:, 1], c=data_class.colors, s=50)

        # Label the plot with filenames (optional)
        for i, file_path in enumerate(data_class.file_paths):
            image_string = f'{os.path.basename(file_path)}'
            # image_string = "."
            plt.text(data_class.features_reduced[i, 0], data_class.features_reduced[i, 1], image_string,
                     fontsize=8, color='black')

        # Add a legend to explain the colors; draw the centroid X's.
        plt.legend(handles=data_class.legend_entries, loc='upper right')
        plt.scatter(data_class.centroids_kmeans[:, 0], data_class.centroids_kmeans[:, 1], c='lime', s=300, marker='X', label='Centroids')

        # Add the axes labels.
        plt.xlabel(f'PC1 ({data_class.pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
        plt.ylabel(f'PC2 ({data_class.pca.explained_variance_ratio_[1] * 100:.1f}% variance)')

        plt.title('K-Means for CEX and PG.')
        plt.savefig(os.path.join(data_class.image_folder, "GAkmeansmatplotlib.jpg"))
        plt.show()

    except Exception as e:
        print(f"An error occurred in GA_kmeansmatplotlib.kmeansmatplotlib_driver: {e}")
