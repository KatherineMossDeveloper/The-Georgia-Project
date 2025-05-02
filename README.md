![Hero](images/HeroPolaroids.png)  

![Crystallization](https://img.shields.io/badge/domain-Crystallization-white)
![Python](https://img.shields.io/badge/Python-3.8-lightblue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.1-blue)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-teal?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/opencrystaldata/cephalexin-reactive-crystallization)
![MIT License](https://img.shields.io/badge/License-MIT-green)
![PyCharm](https://img.shields.io/badge/PyCharm-2023.2.4-lightorange)
![Binary Classification](https://img.shields.io/badge/task-Binary_Classification-yellowgreen)
![ResNet](https://img.shields.io/badge/model-ResNet-yellow)
![Version](https://img.shields.io/github/v/release/KatherineMossDeveloper/the-georgia-project)


## Content. 
[Quick start.](#quick-start) • 
[Instructions](#Instructions) • 
[Code overview.](#code-overview) • 
[Model comparison](#model-comparison) • 
[Contributions](#contributions) • 
[Known issues](#known-issues) • 
[Contact info](#contact-info)

## Quick start. 
1. Set up the code for this project.  
   click "Code" on the repo page, then download the zipped file, or open with GitHub Desktop.  
   set up a Python environment (I used PyCharm, ver. 2023.2.4, Community Edition)  
   install dependencies (I used Python 3.8; TensorFlow 2.10.1)  

2. Get the data from Kaggle.  
   download and unzip the data  
[OpenCrystalData Crystal Impurity Detection](https://www.kaggle.com/datasets/opencrystaldata/cephalexin-reactive-crystallization?resource=download)  
   edit the directories in GAmain.py and GAmodel.py for your pc, per instructions in these files.  

3. Set up the data.  
   follow instructions in GAmain to do that data splitting.  
   run GAmain.  
   
4. Run the training.  
   follow instructions in GAmodel to do training.  
   run GAmodel.  

5. Play time.  
   follow instructions in GAinference to do inference on some files.  
   run GAinference.  

For more detail, read the Georgia Project's documentation is here.  
[Go to the main doc file](docs/maindoc.md)    

For more background, read the research paper.  
[In Situ Imaging Combined with Deep Learning for Crystallization Process Monitoring: Application to Cephalexin Production.](https://www.sciencedirect.com/science/article/abs/pii/S1083616021010896)   

## Code overview.  
<img src="images/codeoverview.png" alt="code overview" width="402" height="293">  

## Model comparison.  
In the table below are the details offered by the published paper, then on the right are the choices that I elected to work with.   
|                         |Salami et al. paper     |my work                |
|-------------------------|------------------------|-----------------------|
|model type               |ResNet-18, ResNet-50    |ResNet-101             |
|optimization method      |SGDM	                  |Keras SGD (momentum .9)|
|learning rate	      	  |1 × 10−4                |1 × 10−1	            |
|training data            |3200−3600 in each class |(same)                 |
|train/val./test %        |70/25/5%                |(same)                 |
|minibatch size           |32−64                   |64                     |
|validation frequency     |10−50                   |1                      |
|added dropout layers     |(did not comment)       |2                      |
|trainable ImageNet layers|(did not comment)       |made last 10% trainable|

## Contributions.  
If you found an issue or would like to make a suggestion for an improvement to the code or documentation, please click on the issue tab on the project page and leave me a note.  If you like this project, leave a star.  

## Known issues.  
None.  

## Contact info.                                                                     
For more details about this project, feel free to reach out to me at katherinemossdeveloper@gmail.com or [LinkedIn](https://www.linkedin.com/pub/katherine-moss/3/b49/228) .  My time zone is EST in the U.S.

[back to top](#content) 

