# The Georgia project.
# GAsplitDataIntoTrainValidandTest.py
#
# This code splits this dataset,
#     https://www.kaggle.com/datasets/opencrystaldata/cephalexin-reactive-crystallization?resource=download
#
# into train (70%), validation (25%) and test (5%), per the paper,
#     https://www.sciencedirect.com/org/science/article/abs/pii/S1083616021010896
#
# The CEX images will be put in folders named "0".  The PG images will be put in folders named "1".
# This folder structure matches the logic later in the GAinference.py code.  If the folders you
# designate do not exist, they will be created.  When the files are divided up into train,
# validation, and testing, they will be moved, not copied, so that you will not need to erase
# the original extracted data set.
#
# To do.
# Edit the folder_prefix variable to point to the OpenCrystalData on your pc.
# #############################################################################################

import os
import shutil

folder_prefix = r"your_drive_letter_and_folder"  # edit this before running the code.  

# Define source folders, pointing to the location of the download of the dataset mentioned above.
source_pg = folder_prefix + r"\archive\cropped\cropped\pg"
source_cex = folder_prefix + r"\archive\cropped\cropped\cex"

# Define destination folders, pointing to the locations where the data will be extracted to.
dest_train_pg = folder_prefix + r"\GAtrainBinary\1"
dest_train_cex = folder_prefix + r"\GAtrainBinary\0"
dest_valid_pg = folder_prefix + r"\GAvalidBinary\1"
dest_valid_cex = folder_prefix + r"\GAvalidBinary\0"
dest_test_pg = folder_prefix + r"\GAtestBinary\1"
dest_test_cex = folder_prefix + r"\GAtestBinary\0"

# Create directories if they don't exist
os.makedirs(dest_train_cex, exist_ok=True)
os.makedirs(dest_train_pg, exist_ok=True)
os.makedirs(dest_valid_cex, exist_ok=True)
os.makedirs(dest_valid_pg, exist_ok=True)
os.makedirs(dest_test_cex, exist_ok=True)
os.makedirs(dest_test_pg, exist_ok=True)


# Function to split and move files
def split_and_move_files(source_folder, dest_train, dest_valid, dest_test, train_ratio=0.70, valid_ratio=0.25):

    # Splits the dataset into training, validation, and test sets and moves files accordingly.
    files = sorted(os.listdir(source_folder))  # Sort files for consistent ordering
    total_files = len(files)

    # Calculate split indices
    train_count = int(total_files * train_ratio)
    valid_count = int(total_files * valid_ratio)

    train_files = files[:train_count]                           # get the files up to train count.
    valid_files = files[train_count:train_count + valid_count]  # get the files from train count to valid count.
    test_files = files[train_count + valid_count:]              # get whatever is remaining.

    # Move training files
    for file in train_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(dest_train, file))
    print(f"Moved {len(train_files)} files to {dest_train}")

    # Move validation files
    for file in valid_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(dest_valid, file))
    print(f"Moved {len(valid_files)} files to {dest_valid}")

    # Move test files
    for file in test_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(dest_test, file))
    print(f"Moved {len(test_files)} files to {dest_test}")


# Process both classes (CEX & PG)
split_and_move_files(source_cex, dest_train_cex, dest_valid_cex, dest_test_cex)
split_and_move_files(source_pg, dest_train_pg, dest_valid_pg, dest_test_pg)

print("Data split into Train (70%), Validation (25%), and Test (5%)")
