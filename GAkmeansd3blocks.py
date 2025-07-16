# The Georgia project on https://github.com/KatherineMossDeveloper/The-Georgia-Project/tree/main
# GAkmeansd3blocks.py
#
# This file contains code that graphs images in PCA space using D3Blocks scatter plot.
#
# Code flow.
#    pca3d_driver
#    add_note
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


def add_note(file_path):
    # read the file and patch it
    with open(file_path, 'r', encoding='utf-8') as file:
        html = file.read()

    # Inject your legend before </body>
    note_html = """
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
      This plot shows OpenCrystalData images in PCA space.<br>
      The colors represent KMeans clusters, graphed using D3Blocks.<br><br>

      For more on D3Blocks, click <a href="https://github.com/d3blocks/d3blocks">D3Blocks</a>.<br>
      For more on OpenCrystalData, click <a href="https://www.kaggle.com/datasets/opencrystaldata/cephalexin-reactive-crystallization?resource=download" target="_blank">OpenCrystalData</a>.<br>
      For more on the Georgia Project, click <a href="https://github.com/KatherineMossDeveloper/The-Georgia-Project">Georgia Project</a>.<br>
    </div>
    """

    html = html.replace('</body>', f'{note_html}\n</body>')

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(html)


def kmeansd3blocks_driver(colors, features_reduced, tooltips, folder_prefix):

    try:
        # Extract features for each image in the folder
        print(f'Starting GAkmeansd3blocks.py')
        colors = np.array(colors).astype(str)

        # Initialize
        d3 = D3Blocks()
        file_path = os.path.join(folder_prefix, 'GAkmeansd3blocks.html')

        # Scatter plot
        d3.scatter(
            features_reduced[:, 0],
            features_reduced[:, 1],
            size=5,
            color=colors,
            stroke='#000000',
            opacity=0.4,
            tooltip=tooltips,
            filepath=file_path,
            title="PCA of some OpenCrystalData data with kmeans colors. "
        )

        add_note(file_path)

    except Exception as e:
        print(f"An error occurred in GApca3d: {e}")


