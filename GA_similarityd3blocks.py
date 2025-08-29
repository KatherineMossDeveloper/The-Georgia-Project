# The Georgia project on https://github.com/KatherineMossDeveloper/The-Georgia-Project/tree/main
# GA_similarityd3blocks.py
#
#     def similarityd3blocks_driver(data_class, limit=100)
#     def visualize(nodes_dict, edges_dict)
#
# If there is a Weaviate database...
# This code will pull records from the Weaviate database (weaviate-client version 3.24.2.),
# using the data_class object handed in.  First, it will create nodes for the D3blocks.d3graph.
# Then, it will create edges for those nodes.
#
# Navigation when the plot is in the browser.
# Hover for tooltip	   Mouse only
# Zoom	               Mouse wheel
# Reset	               Reload the page
#
# To do.
# (nothing)
# #############################################################################################

import os as os
import numpy as np
import pandas as pd
from GAutility import color_palette_small
from GA_dataprocessing import add_note
from matplotlib.colors import ListedColormap
from d3blocks import D3Blocks  # version 1.6.1
from sklearn.metrics.pairwise import cosine_similarity

edges_dictionary = {}


def visualize(edges_dict, full_file_path):

    try:
        # Wrap it as a matplotlib colormap
        custom_cmap = ListedColormap(color_palette_small, name="GA_color_palette_small")

        # 1) Edges DF (keep weight)
        edges_df = pd.DataFrame(
            [
                {
                    'source': e['source'],
                    'target': e['target'],
                    'weight': float(e.get('weight', 1.0)),
                }
                for e in edges_dict.values()
            ]
        )
        if edges_df.empty:
            raise ValueError("No edges to plot.")

        d3 = D3Blocks()
        d3.d3graph(
            edges_df,
            cmap=custom_cmap,
            filepath=full_file_path,
            set_slider=3,
            showfig=False  # adding HTML after; show later
        )

        # add note and show
        add_note(full_file_path, "OpenCrystalData images represented in a force-directed graph.")
        d3.D3graph.show()

    except Exception as e:
        print(f"An error occurred in GA_similarityd3blocks.visualize: {e}")


def similarity3dblocks_withthedatabase(data_class, limit):

    try:
        print('Starting GA_similarityd3blocks.py')

        vectors = {}
        result = data_class.weaviate_instance.weaviate_select_records(limit)
        print(f'Creating the vector list. {len(result)}')

        # step 1: because the weaviate database is up and running, get the vectors from the database.
        for item in result['data']['Get']['CrystalImage']:
            identifier = item.get('image_id')
            class_label = item.get('class_label')
            additional = item.get('_additional', {})
            vector = additional.get('vector')

            if not identifier or vector is None:
                continue

            # show the data as it comes in, for debugging.
            print(f"image_id: {identifier}")
            print(f"class_label: {class_label}")
            print(f"vector preview: {vector[:5]}...")

            vectors[identifier] = np.array(vector)

        # step 2:  create edges.
        print(f'Creating the edges list. {len(vectors)}')
        i = 0
        for source_id, source_vector in vectors.items():
            print(f'--->source_id {source_id} ')
            try:
                target_results = data_class.weaviate_instance.weaviate_find_neighbors(source_vector, 5)
            except Exception as e:
                print(f"Error fetching neighbors: {e}")
                continue

            for neighbor in target_results:
                try:
                    print("--->Neighbor object:", neighbor)
                    target_id = neighbor.get('image_id')
                    edge_id = f"{source_id}_{target_id}"
                    edges_dictionary[edge_id] = {
                        'source': source_id,
                        'target': target_id,
                        'edge_distance': 5.0,
                        'edge_style': 0,
                        'marker_start': '',
                        'marker_end': 'arrow',
                        'label_fontsize': 8,
                        'group': i
                    }

                except Exception as e:
                    print(f"Error creating edge: {e}")
            i = i + 1

            print(f'Created the edges with the database. {len(edges_dictionary)}')

    except Exception as e:
        print(f"An error occurred in GA_similarityd3blocks.similarityd3blocks_driver: {e}")


def similarity3dblocks_withoutdatabase(data_class):

    try:
        print('Starting GA_similarityd3blocks.py')

        print("Inside GA_similarityd3blocks.similarityd3blocks_driver().  There is no connection to the Weaviate database.")
        image_ids = list(data_class.vectors.keys())
        vector_array = np.array(list(data_class.vectors.values()))
        similarity_matrix = cosine_similarity(vector_array)
        k = 4

        for i, source_id in enumerate(image_ids):
            # get the top neighbors, excluding self
            sim_row = similarity_matrix[i]
            top_indices = np.argsort(sim_row)[::-1][1:k + 1]  # skip self

            for j in top_indices:
                target_id = image_ids[j]
                edge_id = f"{source_id}_{target_id}"

                edges_dictionary[edge_id] = {
                    'source': source_id,
                    'target': target_id
                }

        print(f'Created the edges without the database. {len(edges_dictionary)}')

    except Exception as e:
        print(f"An error occurred in GA_similarityd3blocks.similarityd3blocks_driver: {e}")


def similarityd3blocks_driver(data_class, limit=10000):

    try:

        # see if the database is available.
        connected = data_class.weaviate_connected

        if not connected:
            similarity3dblocks_withoutdatabase(data_class)
        else:
            similarity3dblocks_withthedatabase(data_class, limit)

        file_path = os.path.join(data_class.image_folder, 'GAsimilarityd3blocks.html')
        print(f'Inside similarityd3blocks, saving plot as {file_path}')

        # visualize.
        visualize(edges_dictionary, file_path)

    except Exception as e:
        print(f"An error occurred in GA_similarityd3blocks.similarityd3blocks_driver: {e}")
