# The Georgia project.
# GAmain.  Version 1.
#
# This code trains a model on the data from the paper,
#
#    "In Situ Imaging Combined with Deep Learning for Crystallization Process Monitoring:  Application to Cephalexin
#    Production".  Hossein Salami, Matthew A. McDonald, Andreas S. Bommarius, Ronald W. Rousseau, and Martha A. Grover
#    Organic Process Research & Development 2021 25 (7), 1670-1679
#    DOI: 10.1021/acs.oprd.1c00136
#
# This code will set up the folders for training a Resnet101 model.  The model will load the Keras built-in
# ImageNet weights.  It will train using the dataset discussed in the Salami, et al. paper cited above.
#
# Code structure.
# To keep the code neat and modular, the following modules call each other in roughly this order.
#
#    GAmain.py              sets up the folder system and basic variables for the model.
#        GAmodel.py         creates the ResNet101 model, loads ImageNet weights, trains, and reports.
#        GAcallbacks.py     creates the callbacks that the model uses during training.
#        GAanalysis.py      reports on how the training went.
#        GAutility.py       odd and ends of code.
#
# To do.
# Edit the deliverables_folder for your computer.
# If debugging, set really_training to False, edit the debugging folders for your computer, and
# create datasets that are small subsets of the complete dataset.
# #############################################################################################
from datetime import datetime
from GAutility import print_elapsed_time
from GAmodel import ModelTrainer

prefix = "GA"                # prefixed letters for the deliverables file to identify them.
name = "GA_study_ver_1"      # the identifying title of graphs, etc.
deliverables_folder = r"X:\MLresearch\CrystalStudy\Project_GA\GAdeliverables"
use_cpu = False              # Set this to False if running on the GPU
really_training = True       # Set this to False if debugging.

if really_training:
    # we are training.  Edit these folder when training.
    epochs = 100  # the early stopping functions will stop us before we get here.
    train_directory = r'X:\MLresearch\CrystalStudy\Project_GA\data\GAtrainBinary'
    val_directory = r'X:\MLresearch\CrystalStudy\Project_GA\data\GAvalidBinary'
    test_directory = r'X:\MLresearch\CrystalStudy\Project_GA\data\GAtestBinary'
else:
    # we are debugging.  Edit these folders when debugging; create datasets that are small subset of the 'real' data.
    epochs = 3
    train_directory = r'X:\MLresearch\CrystalStudy\Project_GA\data\GAtrainBinaryDEBUG'
    val_directory = r'X:\MLresearch\CrystalStudy\Project_GA\data\GAvalidBinaryDEBUG'
    test_directory = r'X:\MLresearch\CrystalStudy\Project_GA\data\GAtestBinaryDEBUG'


# set up parameters for the Model_train class that we may want to change.
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
