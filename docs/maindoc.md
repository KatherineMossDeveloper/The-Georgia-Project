![Hero](../images/HeroPolaroidsSmall.png)  

## Content.
<a href="#the-overview">
  <img src="../images/HeroSmall.png" alt="icon" style="vertical-align: middle; width: 20px; height: 20px;"/> The overview.
</a><br>
<a href="#the-paper">
  <img src="../images/HeroSmall.png" alt="icon" style="vertical-align: middle; width: 20px; height: 20px;"/> The paper.
</a><br>
<a href="#the-goals">
  <img src="../images/HeroSmall.png" alt="icon" style="vertical-align: middle; width: 20px; height: 20px;"/> The goals.
</a><br>
<a href="#the-development-environment">
  <img src="../images/HeroSmall.png" alt="icon" style="vertical-align: middle; width: 20px; height: 20px;"/> The development environment.
</a><br>
<a href="#the-model">
  <img src="../images/HeroSmall.png" alt="icon" style="vertical-align: middle; width: 20px; height: 20px;"/> The model architecture.
</a><br>
<a href="#the-data">
  <img src="../images/HeroSmall.png" alt="icon" style="vertical-align: middle; width: 20px; height: 20px;"/> The data.
</a><br>  
<a href="#the-results">
  <img src="../images/HeroSmall.png" alt="icon" style="vertical-align: middle; width: 20px; height: 20px;"/> The results.
</a><br>
<a href="#the-georgia-code-and-deliverables">
  <img src="../images/HeroSmall.png" alt="icon" style="vertical-align: middle; width: 20px; height: 20px;"/> The georgia code and deliverables.
</a><br>
<a href="#how-to-recreate-the-results">
  <img src="../images/HeroSmall.png" alt="icon" style="vertical-align: middle; width: 20px; height: 20px;"/> How to recreate the results.
</a><br>
<a href="#the-license">
  <img src="../images/HeroSmall.png" alt="icon" style="vertical-align: middle; width: 20px; height: 20px;"/> The license.
</a><br>
<a href="#contact-info">
  <img src="../images/HeroSmall.png" alt="icon" style="vertical-align: middle; width: 20px; height: 20px;"/> Contact info.  
</a><br>

## The overview
I am pleased to say that I have successfully trained an A.I. model to distinguish between two crystal types in images.  The trained model had all F1 <sup id="a1">[1](#f1)</sup> scores above 99.7% in about 50 epochs or less.

I am a software developer doing an independent study into using machine learning to identify crystallization in images.  I found an interesting dataset and a really good research paper on the topic, so I wrote code to train on the data, using the paper for guidance.  I am posting the code and results here in the hope that others will also find it interesting.  

I found the crystal image dataset on Kaggle.  I decided to work with it because there were enough images to train with, and the images are all high quality.  Here is the hyperlink to the dataset:  [OpenCrystalData Crystal Impurity Detection](https://www.kaggle.com/datasets/opencrystaldata/cephalexin-reactive-crystallization?resource=download).  

The dataset I found was collected using Mettler-Toledo, LLC, (MT) instrumentation.  While this is not a research paper, where one would typically make an affiliation statement, I should mention that I worked at MT on their vision products for years.  However, I am no longer affiliated with the Company and am not necessarily endorsing their products here, nor have I used any intellectual property owned by MT.  

I chose crystallization in images because I think it is an important area of A.I.  Other corners of the A.I. world, like LLM’s, video creation, and robotics, have grabbed headlines these days, but I would argue that the detection and categorization of crystallization in images is just as important because it is used in food processing, drug discovery, quality control in manufacturing, etc.  It would be good to have more developers and data scientists interested in this part of A.I.  

This study is about the crystallization dataset on Kaggle.  However, in this documentation, to make things easier, I will refer to the dataset as the “GA data,” and this project as the “Georgia Project,” since all of the authors were at the School of Chemical & Biomolecular Engineering, Georgia Institute of Technology, in Atlanta, GA, which happens to be my husband’s alma mater.  
[back to top](#content)   

## The paper  
The dataset references the research paper,  

Salami, H., McDonald, M. A., Bommarius, A. S., Rousseau, R. W., & Grover, M. A. (2021). [In Situ Imaging Combined with Deep Learning for Crystallization Process Monitoring: Application to Cephalexin Production](https://doi.org/10.1021/acs.oprd.1c00136). *Organic Process Research & Development, 25*(7), 1670–1679. 

This paper will be referred to here as “the paper.”  Its hyperlink points to an abstract of the paper, along with a supplemental file.   The full paper is only available behind paywalls.  I purchased a copy in order to continue my study, so I ethically cannot share my copy with others here.  

What I can say, without going against the spirit of the paywall agreement, is that the scientists who wrote the paper trained ResNet models with ImageNet weights on the GA data.  They used both ResNet-18 and ResNet-50.  They used MATLAB 2020b Deep Learning Toolbox and deepNetworkDesigner app.  It was a binary classification of images of crystals, designating them as either CEX (a.k.a., “cephalexin antibiotic,” a good thing) or PG (a.k.a. “phenylglycine,” a bad thing). One purpose of the paper was to determine if an impurity, PG, shows up during cephalexin antibiotic production.  This would allow scientists to stop the process early, saving time and resources.  This is the task that the A.I. model addresses.  
[back to top](#content)   

## The goals
I wanted to create useful code, weights, and documentation that use the paper’s guidance and its binary dataset of crystallization images.  The code and weights files that I produce might be useful to others because they are in popular technologies.  The code is in Python, TensorFlow, and Keras.  The weights files are in HDF5, the default in Tensorflow, and ONNX, the cross-platform format.  The details are posted in this project in Markdown language and PNG files.  
[back to top](#content)   

## The development environment.
The paper does not mention whether the MATLAB work done is publicly available.  Therefore, I tried to recreate their work with my code, which is written in Python with TensorFlow (Python 3.8; TensorFlow 2.10.1; Keras 2.10.0).  I used PyCharm ver. 2023.2.4, Community Edition, as the IDE.   

I have a Windows PC.  I have not tested this project in other environments.  For the GPU, I used a Quadro P1000, with 4 GB of memory, using CUDA version 12.2.  The paper said that the researchers used a Intel Core i7 3 GHz unit.

The `use_cpu` variable in GAmain.py is set to true by default, so if you do not have a lot of memory on your GPU, the training will still complete.  If you set this variable to false, you will be using your GPU, assuming you have one.  I do not have access to a variety of computers, so I could not test every scenario.  I only tested it on my own computer, which has a GPU with limited memory.  I have gotten out of memory errors occasionally during training on my GPU, so I created this variable to avoid them.  
[back to top](#content)   

## The model.  
In the table below are the details offered by the published paper, then on the right are the choices that I elected to work with.   
|                         |Salami et al. paper     |my work                |
|-------------------------|------------------------|-----------------------|
|framework                |MATLAB                  |PyCharm                |
|model type               |ResNet-18, ResNet-50    |ResNet-101             |
|optimization method      |SGDM	                   |Keras SGD (momentum .9)|
|learning rate 			      |1 × 10−4		             |1 × 10−1	             |
|training data            |3200−3600 in each class |(same)                 |
|train/val./test %        |70/25/5%                |(same)                 |
|minibatch size 		      |32−64                   |64                     |
|validation frequency     |10−50                   |1                      |
|added dropout layers     |(did not comment)       |2                      |
|trainable ImageNet layers|(did not comment)       |made last 10% trainable|

Here are the changes that made the metrics better and the training shorter. 

1.  ResNet-101.  See `GAmodel.py.`  
The paper used ResNet-18 and ResNet-50.  I thought it would be interesting if I used the more complex model architecture here.  

2.  The learning rate 1E-1.  See `GAmain.py.`  
I gradually increased the learning rate from 1E-4 to 1E-1.  The model learned faster, without losing its mind.  

3.  Two Dropout layers, with drop out rates at .4 and .3 respectively.  See `GAmodel.py.`  
I added these just to see what would happen.  That was a good idea. 

4.  Trainable ImageNet layers.  See `GAmodel.py.`    
I made 10% of the ImageNet layers trainable, not knowing if they would make a difference.  They did.  

I also wrote code to split the data into training, validation, and test sets.  See section a. in “The Georgia code and deliverables.”  I added code that would stop the training based on the F1 scores.  See `GAcallbacks.py.`  
[back to top](#content)  

## The data.  
There are ~6,800 image files for each of the two classes of crystal images.  
|             |PG                |CEX              |
|-------------|------------------|-----------------|
|Train 	      |4,762 image files |4,802 image files|
|Validation 	|1,701 image files |1,715 image files|
|Test		      |  341 image files |  343 image files|
|Totals       |6,804 image files |6,860 image files|

The paper remarks that the two image sets have “very distinct visual features.”  Here are some of the CEX images.  Below that are some PG images.  While the CEX images are distinctive, let me suggest here that the PG data varies a lot.  
<a href="#">
  <img src="../images/samples_cex.png" alt="CEX samples" style="vertical-align: middle; width: 623px; height: 134px;"/>
</a><br>

<a href="#">
  <img src="../images/samples_pg.png" alt="PG samples" style="vertical-align: middle; width: 623px; height: 151px;"/>
</a><br>

The folder structure is as follows during training, validation, and testing.  PG images are put in the “1” folders, and CEX are put in the “0” folders.  The `GAsplitDataIntoTrainValidandTest.py` code will set up these folders, and the `GAmain.py` code file will refer to these locations during training.  You will, of course, need to modify the code to match the folder structure on your computer.  See the “How to recreate the results” section for more.  

…\GAtrainBinary  
    \0  
    \1  
…\GAvalidBinary  
    \0  
    \1  
…\GAtestBinary  
    \0  
    \1  
[back to top](#content)  

## The results.  
The final code had all F1 scores above 99.7% in about 50 epochs or less.  The details of the training runs for this final version are in section D. below.  

To produce these numbers, I re-ran versions of the code so that I could understand exactly what delivered the best metrics and shortest runtimes for training.  I found that making 10% of the ImageNet layers trainable improved metrics.  I found that adding two Dropout layers improved metrics.  Finally, I found that both of them, together, produced a better model.  

This seems contradictory.  When layers are trainable, more weights are getting updated.  When dropouts are used, some weights are getting dropped.  These changes seem to be at odds with each other.  However, I can say that with this data, and this Resnet model, and these ImageNet weights, dropout layers and trainable layers together made the model better.  I want to try this idea on other data and architectures in the future.  

Below are the training results column definitions and then the results in tables.  Each line in the tables below represents a training run (`GAmodel.py`) and test run (`GAanalysis.py`).  The best runs were with version D., which is the version of the code here in the GitHub project.  If you have any questions, or would like more details, write to me.  

The training results column definitions. 

|variable name     |definition                                                                |
|------------------|--------------------------------------------------------------------------|
|run name	         | the unique name given to the run.                                        |
|run time	         | the time it took to train, then run GAanalysis.py code to get metrics.   | 
|train acc	       | the training accuracy of the last epoch, reported in the output window.  | 
|valid acc	       | the validation accuracy of the last epoch, reported in the output window.|   
|test acc	         | the test accuracy, reported in GAFinalTestResults.txt file.              |   
|acc delta	       | the validation accuracy minus the training accuracy.                     |   
|test missed	     | the no. of misclassified images, reported in the confusion matrix.       |   
|epochs		         | the number of epochs completed during the training.                      |   
|learning rate	   | the learning rate used throughout the training.                          |   
|dropout layers	   | the status of the dropout layers, shown in the GAmodel.py.               |   
|trainable layers  | the status of the top 10% ImageNet layers, shown in the GAmodel.py.      |   

Generally, in C. and D. training runs below, the dropout layers made the model run consistently longer.  In the B. training runs, where 10% of the ImageNet weights were being trained, I do not see an improvement over the A. runs.  However, in D., with both dropouts and 10% training of ImageNet weights, I see the best combination.   

The training results.  
A.	The version with no dropouts and no trainable ImageNet layers.  
 <img src="../images/results_a.png" alt="result group a." width="624" height="209">  

B.	The version with no dropouts, but with trainable ImageNet layers.  
<img src="../images/results_b.png" alt="result group b." width="624" height="209">  

C.  The version with dropouts and with no trainable ImageNet layers.  
<img src="../images/results_c.png" alt="result group c." width="624" height="209">  

D.  The final published version with dropouts and trainable ImageNet layers:  the porridge that is just right.  
<img src="../images/results_d.png" alt="result group d." width="624" height="209">  

Here is a note on the formatting of floats.  For some of the runs, for example GArun_27, the GAfinal_confusion_matrix.png showed 1 error, which is in the ‘test missed’ column.  Meanwhile, the ‘test acc’ column shows a note about the results of the GAFinalTestResults.txt.  In these runs, the final class-wise breakdown showed 100% in all categories.  This is due to the formatting of floats.  I could have changed the code perhaps, which produces the final class-wise breakdown, so that 4 decimals, or so, were used, but I did not bother to do that.  Also, seeing 100% anywhere in metrics in the machine learning world is, of course, a mark of over-fitting.  I do not believe that is a problem here.  

[back to top](#content)  

## The Georgia Project code and deliverables.  
Sections a. through c. below are notes on the Python code files.  Sections d. through h. are the notes about the deliverables.  Let's start with a high-level look... 
  
<img src="../images/codeoverview.png" alt="code overview" width="402" height="293">  

a.  The code to split up the data.  
`GAsplitDataIntoTrainValidandTest.py`  
This code splits the OpenCrystalData dataset that you downloaded and extracted into 70% training data, 25% validation data, and 5% test data, because that is the way that it is divided up in the paper.  The dataset extracts itself into a CEX folder and a PG folder.  With this code, these data will be moved to a '0' folder for CEX and a '1' folder for PG.  For example, 70% of the OpenCrystalData CEX folder images will be moved to a GAtrainBinary\0 folder. 

b.  The training and analysis code.  
Generally, this next section of code files further preprocesses the data, trains the ResNet-101 model, then reports results.  The model will load the Keras built-in ImageNet weights.  It will train with the hyperparameters in the paper, with exceptions mentioned in “The model” section.  The code is organized into files, or modules, which call each other, in roughly the order here.

`GAmain.py`        
This module sets up the folder system and basic variables for the model.  It will then call `GAmodel.py`. 

`GAmodel.py`       
This module creates a ResNet-101 model, loads ImageNet weights, trains, and then calls `GAanalyze.py` to reports results.  

`GAcallbacks.py`  
This module creates the callbacks <sup id="a2">[2](#f2)</sup>  that the model will need during training.  

`GAutility.py`    
This module organizes unrelated pieces of code in one place. 

`GAdata.py`    
This module creates the data objects for training, validation, and testing.  For training, it includes data augmentation.

`GAanalysis.py`    
Finally, this code is called after the training, at which time it creates the plots and other results from the training and testing.  

c.  The inference code.  
It is time to play.  After you trained the model, the file `GAinference.py` can then perform inference on any png file that you give it. As an example of an inference run, here is a collection of images, most of which are from the GA dataset, with a few wildcards thrown in.  (I was surprised to see that Tara, The Cat, is in fact phenylglycine.)  Note that the weights file is in the same folder, as the code expects.  

A weights file is created at the end of the training. The computer's date and time stamp are part of the name, so that previously created weights files are not overwritten.  Therefore, your weights files will have a different name than the one shown here, of course.  

You can also download the GAweights file from the GitHub Georgia Project website.  There are two formats available.  The weights files are in HDF5, the default in Tensorflow, and ONNX, the cross-platform format.  Download the [HDF5 weights file](https://github.com/KatherineMossDeveloper/The-Georgia-Project/releases/download/v1.5.2/GAweights.h5) or the [ONNX weights file](https://github.com/KatherineMossDeveloper/The-Georgia-Project/releases/download/v1.5.2/GAweights.onnx) to the \inference folder where you downloaded the Georgia Project code on your PC.  

![InferenceExamples](../images/InferenceExample.png)

|image               |prediction                           |
|--------------------|-------------------------------------|
|File: CEX (1).png   | Prediction: CEX, Confidence: 0.00%  |
|File: CEX (2).png   | Prediction: CEX, Confidence: 0.00%  |
|File: CEX (3).png   | Prediction: CEX, Confidence: 0.20%  |
|File: OIP (5).png   | Prediction: CEX, Confidence: 94.59% |
|File: PG (5282).png | Prediction: PG, Confidence: 100.00% |
|File: PG (567).png  | Prediction: PG, Confidence: 100.00% |
|File: PG (590).png  | Prediction: PG, Confidence: 100.00% |
|File: PG (5946).png | Prediction: PG, Confidence: 76.98%  |
|File: tara.png      | Prediction: PG, Confidence: 90.27%  |

The following notes are about the deliverables created at the end of the training run.  

d.  The resulting weights files in the HDF5 format, native to TensorFlow, and in the ONNX format, for developers working in other environments, like ML.NET.  

<pre style="font-family: 'Courier New', Courier, monospace;">
   GAweights_2025-05-22_23-25-12.h5  
   GAweights_2025-05-22_23-25-12.onnx  
</pre>

e.  During training, in the ouput window in PyCharm, there are class-wise text breakdowns of the test precision, recall, macro average<sup id="a4">[4](#f4)</sup>, weighted average<sup id="a4">[4](#f4)</sup>, and F1-scores.
<pre style="font-family: 'Courier New', Courier, monospace;">
   Epoch 3 - Class-Wise Metrics:
   PG -  Precision: 0.9988, Recall: 0.9982, F1-Score: 0.9985
   CEX - Precision: 0.9982, Recall: 0.9988, F1-Score: 0.9985
   macro avg - Precision: 0.9985, Recall: 0.9985, F1-Score: 0.9985
   weighted avg - Precision: 0.9985, Recall: 0.9985, F1-Score: 0.9985
</pre>

f.  After training, in the GAFinalTestResults.txt file, the model gives similar metrics during testing. The word “support” here tells us how many files the model used to determine the precision, recall, and F1 scores.   
            
|             |precision | recall | f1-score |  support|
|-------------|----------|--------|----------|---------|
|          PG |    0.99  |   1.00 |    1.00  |     341 |
|         CEX |    1.00  |   0.99 |    1.00  |     343 |
|    accuracy |          |        |    1.00  |     684 |
|   macro avg |    1.00  |   1.00 |    1.00  |     684 |
|weighted avg |    1.00  |   1.00 |    1.00  |     684 |  

g.  GAmetrics_plot.png, the plot of training accuracy and validation accuracy<sup id="a5">[5](#f5)</sup>.  
![Results](../images/results_accuracies.png)  

h.  GAfinal_confusion_matrix.png, the confusion matrix<sup id="a6">[6](#f6)</sup>, the heatmap of the errors and correct inferences.   
![Results](../images/results_confusion.png)  

[back to top](#content)  

## How to recreate the results.  
These instructions were written for my Windows PC.  The data zip file is over 1 GB, so you will need a drive with some free space on it to work with this data.  There are notes at the top of each code file about the minimum one needs to do in order to run the code.  Basically, just edit the folder strings to match your pc folder system. 

Step 1. download the code.  
   - Download the [source code ZIP file.](https://github.com/KatherineMossDeveloper/The-Georgia-Project/archive/refs/tags/v1.5.2.zip) or the [source code TAR file.](https://github.com/KatherineMossDeveloper/The-Georgia-Project/archive/refs/tags/v1.5.2.tar.gz) and extract it.  
   - Set up a Python environment, if you don't already have one.  I used PyCharm, ver. 2023.2.4, Community Edition.  
   - Install dependencies as needed.  I used Python 3.8, TensorFlow 2.10.1, and Keras 2.10.0.  

Step 2. download the data.  
   - Navigate to the Kaggle website using this hyperlink, create an account (which you can do for free), sign into the site, and press the “Download” button.
   - Download [OpenCrystalData Crystal Impurity Detection](https://www.kaggle.com/datasets/opencrystaldata/cephalexin-reactive-crystallization?resource=download).
   - Extract it.  Downloading, and especially extraction, can take a while, so go get another cup of coffee.  

The data structure created during extraction for the cropped image files is like this…
<pre style="font-family: 'Courier New', Courier, monospace;">
   your_drive_letter_and_folder/archive/cropped/cropped/cex
   your_drive_letter_and_folder/archive/cropped/cropped/pg
</pre>

Step 3. split up the data.  
   - Open the `GAsplitDataIntoTrainValidandTest.py` code file in the Python IDE.
   - Edit `folder_prefix` to the location where you extracted the data earlier.  The code will take that data and split it up in the `destination folders`.
   - Read the “to do” list at the top of the file for more information.
   - Run the `GAsplitDataIntoTrainValidandTest.py` code file.
   
At the end of the run, you should see messages that look something like this.  

<pre style="font-family: 'Courier New', Courier, monospace;">
   Moved 4802 files to X:\MLresearch\CrystalStudy\Project_GA\GitHubTestingData\GAtrainBinary\1
   Moved 1715 files to X:\MLresearch\CrystalStudy\Project_GA\GitHubTestingData\GAvalidBinary\1
   Moved 343 files to X:\MLresearch\CrystalStudy\Project_GA\GitHubTestingData\GAtestBinary\1
   Moved 4762 files to X:\MLresearch\CrystalStudy\Project_GA\GitHubTestingData\GAtrainBinary\0
   Moved 1701 files to X:\MLresearch\CrystalStudy\Project_GA\GitHubTestingData\GAvalidBinary\0
   Moved 341 files to X:\MLresearch\CrystalStudy\Project_GA\GitHubTestingData\GAtestBinary\0
   Data split into Train (70%), Validation (25%), and Test (5%)
</pre>    

Step 4. prepare for training.  
   - Open the `GAmain.py` code file.  Edit `folder_prefix` to the location where you extracted the data earlier.  

   - Setting up the study variables.  
The `prefix` and `name` variables are put on plots, etc., after training, so you can edit them to identify a given run.  

   - Setting up the deliverables folder.  The `deliverables_folder` variable is where the code will save the results and the weights files at the end of training.  This folder will be, by default, automatically created in the same folder where you saved the data.  You can, of course, change the folder location and/or name. 

   - Setting up the CPU/GPU.  The `use_cpu` variable is set to true by default, so if you do not have a lot of memory on your GPU, the training will still complete.  If you set this variable to false, you will be using your GPU, assuming you have one.  I do not have access to a variety of computers, so I could not test every scenario.  I only tested it on my own computer, which has a GPU with limited memory.  I have gotten out of memory errors occasionally during training on my GPU, so I created this variable to avoid them.  

   - Setting up debugging folders and data.  The `really_training` variable is set to true by default.  On the other hand, sometimes when debugging, it is nice to have a way to run a short “rehearsal” training.  If you want that, in the `GAmain.py` code, set the variable `really_training` to False.  Then edit the folders for debugging to point to an abbreviated version of the files.  To fill up the train, validation, and test image file folders for debugging, I copied about 10% of the files from the “real” training folders.  The epochs variable is set to 3, again so that the run is short.  Note that your training metrics will look bad when you run your debugging sub-set of the data because the model will not have time to train on the data, nor enough data to train with.  Debugging in this way is for testing things like a new result file, not model performance.  

Step 5. train.  
   - To train the model, start the `GAmain.py` file.  You can watch progress in the output window of PyCharm, if that is your IDE.  

Step 6. view results.  
   - All results files generated will be in your deliverables folder that you designated in the `GAmain.py` code file.  

Step 7. play with it.  
   - Lastly, there is a code file, `GAinference.py`, that will perform inference on any png file that you give it.  Edit `folder_prefix` to point to the location where you extracted the code earlier (not the data).  This folder contains a few images from the project, plus a few stray images.  You could add your own png files there too.  

   - Setting up the weights file.  There are two options with the weights file.  You can use the weights file that you created in Step 5, or you can use the weights file from the Georgia Project on GitHub.  Click here to download the [HDF5 weights file](https://github.com/KatherineMossDeveloper/The-Georgia-Project/releases/download/v1.5.0/GAweights.h5) or the [ONNX weights file](https://github.com/KatherineMossDeveloper/The-Georgia-Project/releases/download/v1.5.0/GAweights.onnx). Either way, save it to the existing \inference folder in the code folder on your pc.  Note that there is a `weights_file` variable in the `GAinference.py` code.  By default, it is expecting the weights file to be called "GAweights.h5".  Edit that as needed.  

   - Run `GAinference.py`  The labeling and confidence factors will appear in the output window.  

[back to top](#content)  

## The license.  
This project is licensed under the MIT License.  See the license.txt file for details [here](../LICENSE).  
[back to top](#content)  

## Contact info.                                                                     
For more details about this project, feel free to reach out to me at katherinemossdeveloper@gmail.com or my account on [LinkedIn](https://www.linkedin.com/pub/katherine-moss/3/b49/228) .  
My time zone is EST in the U.S.

[back to top](#content)  


#### Footnotes
> <sup id="f1">1</sup> The F1-score is an overall score of how well a model is learning.  It is the harmonic mean of precision and recall for each class.  A simple average, also known as an “arithmetic mean,” would be too crude a measure because it would simply “take an average” of the two numbers.  For example, if precision were 90% and recall were 20%, then the average would be 55, so you might think the model is working somewhat.
>
>                                        (90 + 20) / 2 = 55
>
> In reality, the imbalance between the two numbers is a problem, a problem not reflected if you calculated an arithmetic mean, which gives you 55 as the answer.  In contrast, using the same numbers, the harmonic mean would be ~32%, a better reflection of the trouble that the model is in.  Below is the formula for the harmonic mean.
>
>           F1=  (2 x precision x recall) / (precision + recall) = (2 x 90 x 20) / (90 + 20) ≅ 32
>
> For more about F1, visit [F1 on Wikipedia](https://en.wikipedia.org/wiki/F-score).  
> [⏎](#a1)  


> <sup id="f2">2</sup> Callbacks can be thought of as hooks into the model while it is training.  A typical callback is a class with an event function, like on_epoch_end.   Just like it sounds, when the model comes to the end of each epoch, it will call all callback classes, so that each class can do things, like collect metrics, write to the screen, make a decision about stopping the training, etc.  It gives the developer more information and control during training.
>
> Technically, in Keras, callbacks are classes because they inherit from the “Callback” mothership.  Therefore, you would instantiate a callback class, hand it to the model, which would then call the instantiation, not the class itself.
> 
> For more about callbacks, visit [callbacks on Keras.io](https://keras.io/api/callbacks/).  
> [⏎](#a2)

> <sup id="f4">4</sup>  The macro average is just what it sounds like, the average of precision, recall, or F1 scores for all classes, which for us is just PG and CEX.
>   
> Macro Avg Precision = Average of the precision values of both classes.  
> Macro Avg Recall = Average of the recall values of both classes.  
> Macro Avg F1-Score = Average of the F1-scores of both classes.  
>
> Here is an example from a training run.  
>
> PG           - Precision: 0.9591, Recall: 0.9993, F1-Score: 0.9788  
> CEX          - Precision: 0.9992, Recall: 0.9552, F1-Score: 0.9767  
> macro avg    - Precision: 0.9791, Recall: 0.9773, F1-Score: 0.9777  
> weighted avg - Precision: 0.9786, Recall: 0.9778, F1-Score: 0.9778  
>
> In contrast, the weighted average takes into account the number of samples in each class.  For us, the number of samples is about the same, but for another dataset, these numbers could be unequal, so these numbers could tell us more in that case.  
> 
> For more, visit [weighted average on Wikipedia](https://en.wikipedia.org/wiki/Weighted_arithmetic_mean).  
> [⏎](#a4)

> <sup id="f5">5</sup> Training accuracy is a measure, during the training epochs, of how well the model is learning the training data.
> 
> Validation accuracy is a measure, during the training, of how well the model can perform on validation data, meaning data that it has not been trained on.
> 
> Testing accuracy is a measure, after training, of how well the model can do predictions, or inference, on a data test set that it was not trained on, often deemed an “independent” dataset that is related to the other two sets, but not collected with it.
> 
> For more, visit [validation vs test vs training accuracy](https://www.geeksforgeeks.org/training-vs-testing-vs-validation-sets/).  
> [⏎](#a5)

> <sup id="f6">6</sup> A Confusion matrix is a commonly used deliverable in machine learning.  Rather than just looking at accuracy numbers, for example, this matrix
> can simply show you how many errors there were, and in which category.
>
> Since this project has been a binary classification effort, and the metrics were very good, the confusion matrix is simple, as seen below.  The X axis (the horizontal axis) is what the model decided, or predicted.  The Y axis (the vertical axis) is the “label,” or “truth.”  The squares are colorized.  The higher the number, the deeper the shade of blue. 
>
> ![Results](../images/results_confusion.png)  
> -The dark blue square on the upper left shows us the number of times, 343, that the model was shown PG crystals and recognized them.  
> -The dark blue square on the lower right shows us the number of times, 341, that the model was shown CEX crystals and recognized them.  
> -The other two squares show the number of errors.  
> 
> It is called a “confusion” matrix because it can show where a model was confused.  Since the numbers here are good, I could have named it the “Clarity Matrix.”
> 
> For more, visit [confusion matrix on Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix).  
> [⏎](#a6)
 

