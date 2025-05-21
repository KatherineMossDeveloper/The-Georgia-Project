# The Georgia project on https://github.com/KatherineMossDeveloper/The-Georgia-Project/tree/main
# GAmain.py
#
# This code trains a model on the data from the paper,
#
#    "In Situ Imaging Combined with Deep Learning for Crystallization Process Monitoring:  Application to Cephalexin
#    Production".  Hossein Salami, Matthew A. McDonald, Andreas S. Bommarius, Ronald W. Rousseau, and Martha A. Grover
#    Organic Process Research & Development 2021 25 (7), 1670-1679
#    DOI: 10.1021/acs.oprd.1c00136
#
# This OpenCrystalData dataset is here,
#     https://www.sciencedirect.com/org/science/article/abs/pii/S1083616021010896
#
# The data is here, 
#     https://www.kaggle.com/datasets/opencrystaldata/cephalexin-reactive-crystallization?resource=download
#
# This code will set up the folders for training a model.  The model will load the Keras built-in
# ImageNet weights.  It will train using the dataset discussed in the Salami, et al. paper cited above.
# If the folders you designate do not exist, they will be created.
#
# Setting up the CPU/GPU.
# The "use_cpu" variable is set to true by default, so that if you do not have a GPU, or if you do not have
# a lot of memory on your GPU, the training will still complete. If you set this variable to false, you will
# be using your GPU.
#
# Setting up debugging.
# If debugging, set really_training to False, edit the debugging folders for your computer, and create
# datasets that are small subsets of the complete dataset.
#
# Code structure.
# The following modules call each other in roughly this order.
#
#    GAmain.py              sets up the folder system and basic variables for the model.
#        GAmodel.py         creates the model, loads ImageNet weights, trains, and reports.
#        GAcallbacks.py     creates the callbacks that the model uses during training.
#        GAdata.py          create the data objects.
#        GAanalysis.py      reports on how the training went.
#        GAutility.py       odd and ends of code.
#
# To do.
# Edit the folder_prefix variable to point to the OpenCrystalData on your pc.
# #############################################################################################
from datetime import datetime
from GAutility import print_elapsed_time
from GAmodel import ModelTrainer

folder_prefix = r"your_drive_letter_and_folder"  # edit this before running the code.

prefix = "GA"                # prefixed letters for the deliverables file to identify them.
name = "GA_study"            # the identifying title of graphs, etc.
deliverables_folder = folder_prefix + r"\GAdeliverables"  # result files after training will be here. 
use_cpu = True               # Set this to False if running on the GPU
really_training = True       # Set this to False if debugging.

if really_training:
    # we are training.
    epochs = 100  # the early stopping functions will stop us before we get here.
    train_directory = folder_prefix + r"\GAtrainBinary"
    val_directory = folder_prefix + r"\GAvalidBinary"
    test_directory = folder_prefix + r"\GAtestBinary"
else:
    # we are debugging.  Create datasets that are small subset of the 'real' data.
    epochs = 3
    train_directory = folder_prefix + r"\GAtrainBinaryDEBUG"
    val_directory = folder_prefix + r"\GAvalidBinaryDEBUG"
    test_directory = folder_prefix + r"\GAtestBinaryDEBUG"

# set up parameters for the Model class that we may want to change.
batch_size = 64
image_size = (224, 224)
random_seed = 42                 # using a random seed in the hopes of creating more reproducible metrics.
learning_rate = 1E-1             # learning rate
loaded_weights = "imagenet"      # Keras built-in weights file.
builtin_weights = 'y'  # if using built-in weights, like ImageNet, set this to 'y'; otherwise, set this to 'n'.

start_time = datetime.now()

# Instantiate and run the model trainer
trainer = ModelTrainer(loaded_weights, builtin=builtin_weights, use_cpu_only=use_cpu, epochs_total=epochs,
                       train_dir=train_directory, val_dir=val_directory,
                       test_dir=test_directory, deliverables_dir=deliverables_folder,
                       batch=batch_size, images=image_size, seed=random_seed, learning=learning_rate,
                       study_name=name, prefix=prefix)

completed_logger = trainer.run()

print_elapsed_time(start_time)
