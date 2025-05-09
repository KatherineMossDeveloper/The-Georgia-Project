# The Georgia project.
# GAsplitDataIntoTrainValidandTest.py
#
# This code splits this dataset,
#     https://www.kaggle.com/datasets/opencrystaldata/cephalexin-reactive-crystallization?resource=download
#
# into train (70%), validation (25%) and test (5%), per the paper,
#     https://www.sciencedirect.com/org/science/article/abs/pii/S1083616021010896
#
# The folder structure will be setup as CEX (0) vs PG (1), so that it matches the logic
# later in the GAinference.py code.

# To do.
# Edit the folder_prefix variable to point to the OpenCrystalData on your pc.
# If the folders you designate do not exist, they will be created.
# #############################################################################################

import os
import shutil

folder_prefix = r"your_drive_letter_and_folder"  # edit this before running the code.  

# Define source folders, pointing to the location of the download of the dataset mentioned above.
source_pg = folder_prefix + r"\archive\cropped\cropped\pg"
source_cex = folder_prefix + r"\archive\cropped\cropped\cex"

# Define destination folders, pointing to the locations where the data will be extracted to.
train_pg_dest = folder_prefix + r"\GAtrainBinary\1"
train_cex_dest = folder_prefix + r"\GAtrainBinary\0"
valid_pg_dest = folder_prefix + r"\GAvalidBinary\1"
valid_cex_dest = folder_prefix + r"\GAvalidBinary\0"
test_pg_dest = folder_prefix + r"\GAtestBinary\1"
test_cex_dest = folder_prefix + r"\GAtestBinary\0"

# Create directories if they don't exist
os.makedirs(train_cex_dest, exist_ok=True)
os.makedirs(train_pg_dest, exist_ok=True)
os.makedirs(valid_cex_dest, exist_ok=True)
os.makedirs(valid_pg_dest, exist_ok=True)
os.makedirs(test_cex_dest, exist_ok=True)
os.makedirs(test_pg_dest, exist_ok=True)


# Function to split and move files
def split_and_move_files(source_folder, train_dest, valid_dest, test_dest, train_ratio=0.70, valid_ratio=0.25):
    """
    Splits the dataset into training, validation, and test sets and moves files accordingly.
    """
    files = sorted(os.listdir(source_folder))  # Sort files for consistent ordering
    total_files = len(files)

    # Calculate split indices
    train_count = int(total_files * train_ratio)
    valid_count = int(total_files * valid_ratio)
    test_count = total_files - train_count - valid_count  # Remaining files go to test set

    train_files = files[:train_count]
    valid_files = files[train_count:train_count + valid_count]
    test_files = files[train_count + valid_count:]

    # Move training files
    for file in train_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(train_dest, file))
    print(f"Moved {len(train_files)} files to {train_dest}")

    # Move validation files
    for file in valid_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(valid_dest, file))
    print(f"Moved {len(valid_files)} files to {valid_dest}")

    # Move test files
    for file in test_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(test_dest, file))
    print(f"Moved {len(test_files)} files to {test_dest}")


# Process both classes (CEX & PG)
split_and_move_files(source_cex, train_cex_dest, valid_cex_dest, test_cex_dest)
split_and_move_files(source_pg, train_pg_dest, valid_pg_dest, test_pg_dest)

print("Data split into Train (70%), Validation (25%), and Test (5%)")
