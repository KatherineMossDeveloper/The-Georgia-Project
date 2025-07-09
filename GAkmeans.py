# The Georgia project on https://github.com/KatherineMossDeveloper/The-Georgia-Project/tree/main
# GAkmeans.py
#
# This file contains code to do K-means clustering and PCA on image files.
#
# Code flow.
#    kmeans_driver
#       extract_features
#       sklearn.cluster.PCA
#       sklearn.decomposition.Kmeans
#       visualize_clusters
#
# To do.
# (nothing)
# #############################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from GAutility import load_and_preprocess_image


# Function to extract features from the image using the pre-trained model
def extract_features(model, img_path):
    # Load and preprocess the image
    img_array = load_and_preprocess_image(img_path)

    # Create a vector of features (patterns, textures) using the pre-trained ResNet50 model
    features = model.predict(img_array)

    # Flatten the features (from 3D to 1D) which are no longer in pixel format.
    features_flat = features.flatten()

    return features_flat


# Visualize both PG and CEX clusters on one plot with different colors for 3 clusters
def visualize_clusters(pca, reduced_features, colors, image_files, centroids, image_folder, legend_handles):

    # draw the PCA components
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=colors, s=50)

    # Label the plot with filenames (optional)
    for i, file_path in enumerate(image_files):
        image_string = f'{os.path.basename(file_path)}'
        # image_string = "."
        plt.text(reduced_features[i, 0], reduced_features[i, 1], image_string, fontsize=8)

    # Add a legend to explain the colors; draw the centroid X's.
    plt.legend(handles=legend_handles, loc='upper right')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='lime', s=300, marker='X', label='Centroids')

    # Add the axes labels.
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')

    plt.title('K-Means for CEX and PG.')
    plt.savefig(os.path.join(image_folder, "kmeans_plot.jpg"))
    plt.show()


# Function to perform K-Means clustering
def kmeans_driver(model, num_clusters=2, file_paths="", image_folder=""):
    features = []
    colors = []
    centroids_kmeans = []
    labels_kmeans = []
    features_reduced = []

    try:
        # Extract features for each image in the folder
        print(f'Starting GAkmeans.py with {len(file_paths)} files.')

        for filename in file_paths:
            features.append(extract_features(model, filename))

        # Convert features list to numpy array
        features_array = np.array(features)

        # Reduce dimensionality
        pca: PCA = PCA(n_components=2)  # 2 components for a 2D plot.
        features_reduced = pca.fit_transform(features_array)

        # Apply K-Means clustering
        sklearn_kmeans_clustering: KMeans = KMeans(n_clusters=num_clusters, random_state=42,
                                                   n_init=10, max_iter=10000)
        sklearn_kmeans_clustering.fit(features_reduced)

        # Get cluster labels
        labels_kmeans = sklearn_kmeans_clustering.labels_
        centroids_kmeans = sklearn_kmeans_clustering.cluster_centers_

        # straighten out the forward, backward slashes.
        normalized_files = [os.path.normpath(file_path) for file_path in file_paths]

        # Set colors for each cluster
        colors = ['purple' if label == 0 else
                  'blue' if label == 1 else
                  'darkorange' if label == 2 else
                  'green' for label in labels_kmeans
                  ]

        # Create custom legend handles
        legend_handles = [
            mpatches.Patch(color='purple', label='Cluster 1'),
            mpatches.Patch(color='blue', label='Cluster 2'),
            mpatches.Patch(color='orange', label='Cluster 3'),
            mpatches.Patch(color='green', label='Cluster 4')
        ]

        # Visualize clusters on one plot
        visualize_clusters(pca, features_reduced, colors, normalized_files,
                           centroids_kmeans, image_folder, legend_handles)

        # Print clustering results
        print("images clustering results:")

    except Exception as e:
        print(f"An error occurred in GAkmeans: {e}")

    return colors, centroids_kmeans, labels_kmeans, features_reduced

