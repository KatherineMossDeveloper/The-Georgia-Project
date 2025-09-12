# The Georgia project on https://github.com/KatherineMossDeveloper/The-Georgia-Project/tree/main
# GA_dataprocessing.py
#
# class DataPreprocessor
#     classifications_processing(self, model, image_folder, file_type=".png", mod=1)
#             loop through image files, get confidence %, create tooltips.
#     def setup_data(self)
#     def classifications_processing(self)  
#     def extract_features(model, img_path)
#     def kmeans_processing(self, num_clusters=4)
#             extract features for each image in the folder, perform PCA to
#             reduce these vectors to 2D, then do kmeans on them.
#     def add_note(file_path, note)
#
# This code will pull png files from a folder and do inference on each one, reporting the
# classification and confidence to the output window.  The prediction logic below assumes
# a binary classification, where the CEX images are in folder '0' and PG are in folder '1'.
# By convention, '1' is the positive class in binary classification, meaning the class that
# the model is trained to predict.
#
# Note that the model in GAmodel is resnet101, but the preprocess_input is from resnet50
# because the pre-processing is identical, and keras did not create one for the resnet101.
#
# To do.
# (nothing)
# #############################################################################################
import os
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet101
from GAutility import load_and_preprocess_image, get_plot_color_objects
from GA_weaviatedatabase import WeaviateDatabase


class DataProcessor:
    def __init__(self, image_folder, weights_folder,
                 file_type=".png", mod=1):
        self.classifier_model = None
        self.feature_model = None
        self.image_folder = image_folder
        self.weights_folder = weights_folder
        self.file_type = file_type
        self.mod = mod
        self.file_paths = []
        self.file_names = []
        self.tooltips = []
        self.features = []
        self.colors = []
        self.legend_entries = []
        self.features_reduced = []
        self.normalized_files = []
        self.centroids_kmeans = []
        self.vectors = {}  # dictionary for vectors and their ids
        self.weaviate_instance = None
        self.weaviate_connected = False
        self.pca = None
        self.client_connection = None

    def setup_data(self):

        try:
            # The database.
            # -instantiate the weaviate class,
            # -see if it is up and running.
            self.weaviate_instance = WeaviateDatabase(class_name="CrystalImage",
                                                      class_description="A crystal image feature vector with metadata",
                                                      class_vectorizer="none")
            self.weaviate_connected = self.weaviate_instance.weaviate_connect()
            print(f'connected? {self.weaviate_instance.weaviate_connected}')
            print(f'version?   {self.weaviate_instance.weaviate_available()}')

            # The classifier model.
            # -predict labels
            # -get confidence percentages.
            self.classifier_model = load_model(self.weights_folder)
            base_model = ResNet101(weights=None, include_top=False, input_shape=(224, 224, 3))
            base_model.load_weights(self.weights_folder, by_name=True, skip_mismatch=True)
            self.classifications_processing()

            # The feature model.
            # -extract features for each image in the folder,
            # -put the vectors in the weaviate database, if it is up and running,
            # -perform PCA to reduce these vectors to 2D,
            # -then do kmeans on them.
            self.feature_model = Model(inputs=base_model.input,
                                       outputs=GlobalAveragePooling2D()(base_model.output))
            self.kmeans_processing(num_clusters=4)

        except Exception as e:
            print(f"An error occurred in DataPreprocessor.model_driver: {e}")

    def classifications_processing(self):
        counter = 0

        try:
            # Loop through the folder, load each image file, then get the
            # confidence factor.  Create a tooltip for each image.
            print(f'Starting DataPreprocessor.classifications_processing.')

            for filename in sorted(os.listdir(self.image_folder)):
                counter = counter + 1

                if filename.endswith(self.file_type) and counter % self.mod == 0:
                    file_path = os.path.join(self.image_folder, filename)
                    self.file_paths.append(file_path)
                    self.file_names.append(filename)

                    # Load and preprocess the image
                    img_array = load_and_preprocess_image(file_path)

                    # Perform inference on the image
                    prediction = self.classifier_model.predict(img_array, verbose=0 )

                    # Create confidence percentage
                    confidence = prediction[0][0]
                    confidence_percent = int(round(confidence * 100))
                    confidence_percent = confidence_percent if confidence_percent >= 50 else (100 - confidence_percent)

                    # Assign the class label with threshold 0.5 for PG.
                    class_label = 'PG' if confidence >= 0.5 else 'CEX'  # PG if prob >= 0.5, else CEX

                    # Create tooltip list
                    short_name = os.path.basename(filename)
                    tip = f'{short_name} ({confidence_percent}% that this is {class_label}.)'
                    self.tooltips.append(tip)

                    # Print the result with confidence as a percentage
                    print(f"File {filename} prediction: {class_label} with confidence {confidence_percent:.2f}%")

        except Exception as e:
            print(f"An error occurred in DataPreprocessor.classifications_processing: {e}")

        print(f'counter {counter}')

        return

    # Function to extract features from the image using the pre-trained model
    @staticmethod
    def extract_features(model, img_path):

        features_flat = []
        vector = []

        try:

            # Load and preprocess the image
            img_array = load_and_preprocess_image(img_path)

            # Create a vector of features (patterns, textures) using the pre-trained ResNet50 model
            features = model.predict(img_array)

            # Convert the features to a numpy list of float 32 values for storage in the db.
            vector = features.flatten().astype(np.float32).tolist()

            # Flatten the features (from 3D to 1D) which are no longer in pixel format.
            features_flat = features.flatten()

        except Exception as e:
            print(f"An error occurred in GA_dataprocessing.extract_features: {e}")

        return features_flat, vector

    # Function to perform K-Means clustering
    def kmeans_processing(self, num_clusters=4):

        try:

            # Extract features for each image in the folder, perform PCA to
            # reduce them to 2D, then do kmeans on them.
            # If the weaviate database is available, put vectors there.
            print(f'Starting GA_dataprocessing.py with {len(self.file_paths)} files.')

            if self.weaviate_connected:
                self.weaviate_instance.weaviate_delete_and_create_schema()

            for filename in self.file_names:
                if filename.endswith('.png'):
                    img_path = os.path.join(self.image_folder, filename)
                    features_flat, vector = self.extract_features(self.feature_model, img_path)
                    self.features_reduced.append(features_flat)
                    label = os.path.basename(filename)[:3]

                    # store the vectors in the vectors list in this class,
                    # for later use.
                    self.vectors[filename] = features_flat

                    # if the database is available, store the vectors there also,
                    # for demonstration purposes.
                    if self.weaviate_connected:
                        self.weaviate_instance.weaviate_add_record(filename=filename, image_vector=vector,
                                                                   class_label=label, confidence_factor=1)

            # check vectors stored here.
            print(type(list(self.vectors.values())[0]))
            print(len(self.vectors))
            print(len(list(self.vectors.values())[0]))

            # Convert features list to numpy array
            features_array = np.array(self.features_reduced)

            # Reduce dimensionality
            self.pca: PCA = PCA(n_components=2)  # 2 components for a 2D plot.
            self.features_reduced = self.pca.fit_transform(features_array)

            # Apply K-Means clustering
            sklearn_kmeans_clustering: KMeans = KMeans(n_clusters=num_clusters, random_state=42,
                                                       n_init=10, max_iter=10000)
            sklearn_kmeans_clustering.fit(self.features_reduced)

            # Get cluster labels ([2 0 3 0...)
            labels_kmeans = sklearn_kmeans_clustering.labels_
            self.centroids_kmeans = sklearn_kmeans_clustering.cluster_centers_
            print(f'f labels_kmeans {labels_kmeans}')

            # straighten out the forward, backward slashes.
            self.normalized_files = [os.path.normpath(file_path) for file_path in self.file_paths]

            # create a list of colors for each dot in the plot and each legend entry, in hex
            self.colors, self.legend_entries = get_plot_color_objects(labels_kmeans, num_clusters)

        except Exception as e:
            print(f"An error occurred in GA_dataprocessing.kmeans_processing: {e}")

        return


def add_note(file_path, note):

    try:
        # read the file and give it a legend.
        # note that this code assumes that the file was saved to disk already.
        with open(file_path, 'r', encoding='utf-8') as file:
            html = file.read()

        note_html = f"""
           <div style="
            position: absolute;
            top: 100px; right: 20px;
            background: #f0f0f0;
            padding: 10px;
            border: 1px solid #767676;
            border-radius: 4px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            font-family: system-ui, sans-serif;
            font-size: 13px;  /* optional: match button font size */
          ">
          {note}<br><br>                    

          For more on D3Blocks, click <a href="https://github.com/d3blocks/d3blocks">D3Blocks</a>.<br>
          For more on OpenCrystalData, click <a href="https://www.kaggle.com/datasets/opencrystaldata/cephalexin-reactive-crystallization?resource=download" target="_blank">OpenCrystalData</a>.<br>
          For more on the Georgia Project, click <a href="https://github.com/KatherineMossDeveloper/The-Georgia-Project">Georgia Project</a>.<br>
        </div>
        """

        html = html.replace('</body>', f'{note_html}\n</body>')

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(html)
        print(f' added the note :  {note} to {file_path}')

    except Exception as e:
        print(f"Error thrown in GA_dataprocessing.addnote:  {e}")
        return False
