
![Hero](images/HeroWithTitleSmall.png)

![Python](https://img.shields.io/badge/Python-3.8-blue)
![Kaggle](https://img.shields.io/badge/Kaggle-Data-teal?logo=kaggle&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-green)
![PyCharm](https://img.shields.io/badge/PyCharm-2023.2.4-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.1-gold)

### Contents:
- [The overview.](#the-overview).
- [The paper.](#the-paper).
- [The goals.](#the-goals).
- [The development environment.](#the-development-environment).
- [The model architecture.](#the-model-architecture).
- [The data.](#the-data).
- [The results.](#the-results).
- [The georgia code and deliverables.](#the-georgia-code-and-deliverables).
- [ How to recreate the results.](#how-to-recreate-the-results).
- [The license.](#the-license).
- [How to contact me](#how-to-contact-me).
- [The footnotes.](#the-footnotes).

## The overview
<img src="images/HeroSmall.png" alt="Logo" style="vertical-align: left;"> The overview.  
I am pleased to say that I have successfully trained an A.I. model to distinguish between two crystal types in images.  The trained model had all F1 scores above 99.7% in 5 epochs or less. 

I am a software developer doing an independent study into using machine learning to identify crystallization in images.  I found an interesting dataset and a really good research paper on the topic, so I wrote code to train on the data, using the paper for guidance.  I am posting the code and results here in the hope that others will also find it interesting.  

I found the crystal image dataset on Kaggle.  I decided to work with it because there were enough images to train with, and the images are all high quality.  Here is the hyperlink to the dataset.  

https://www.kaggle.com/datasets/opencrystaldata/cephalexin-reactive-crystallization?resource=download
The dataset I found was collected using Mettler-Toledo, LLC, (MT) instrumentation.  While this is not a research paper, where one would typically make an affiliation statement, I should mention that I worked at MT on their vision products for years.  However, I am no longer affiliated with the Company and am not necessarily endorsing their products here, nor have I used any intellectual property owned by MT.  

I chose crystallization in images because I think it is an important area of A.I.  Other corners of the A.I. world, like LLM’s, video creation, and robotics, are in the news more these days, but the detection and categorization of crystallization in images is important because it is used in the food processing, drug discovery, quality control in manufacturing, etc.  It would be good to have more developers and data scientists interested in this part of A.I.  

This study is about the crystallization dataset on Kaggle.  However, in this documentation, to make things easer, I will refer to the dataset as the “GA data,” and this project as the “Georgia Project,” since all of the authors were at the School of Chemical & Biomolecular Engineering, Georgia Institute of Technology in Atlanta, GA, which happens to be my husband’s alma mater.  

## The paper
<img src="images/HeroSmall.png" alt="Logo" style="vertical-align: left;"> The paper.  
## The goals
<img src="images/HeroSmall.png" alt="Logo" style="vertical-align: left;"> The goals.  
<img src="images/HeroSmall.png" alt="Logo" style="vertical-align: left;"> The development environment.  
<img src="images/HeroSmall.png" alt="Logo" style="vertical-align: left;"> The model architecture.  
<img src="images/HeroSmall.png" alt="Logo" style="vertical-align: left;"> The data.  
<img src="images/HeroSmall.png" alt="Logo" style="vertical-align: left;"> The results.  

This project trains an A.I. model to label images in a crystallization dataset.  You can train the model, or use the weights file included, to label images of your own.  
![InferenceExamples](images/InferenceExample2.png)

<img src="images/HeroSmall.png" alt="Logo" style="vertical-align: left;"> The georgia code and deliverables.  
<img src="images/HeroSmall.png" alt="Logo" style="vertical-align: left;"> How to recreate the results.  
<img src="images/HeroSmall.png" alt="Logo" style="vertical-align: left;"> The license.  
<img src="images/HeroSmall.png" alt="Logo" style="vertical-align: left;"> How to contact me.  
<img src="images/HeroSmall.png" alt="Logo" style="vertical-align: left;"> The footnotes.  

![Tara](images/BannerTara.png)


Go to another md file...
[Go to the ReadmeAux file](docs/ReadmeAux.md)

Go to another section in this md file...
[Go to Section 5](docs/ReadmeAux.md#section-5-methodology)

Example hyperlink 
- [OpenCrystalData](https://www.kaggle.com/datasets/opencrystaldata/cephalexin-reactive-crystallization?resource=download) - HTML

### Bulleted List:
- Item 1
- Item 2
- Item 3

This is some more text with a footnote[^1].

[^1]: This is the footnote content, which can provide more information or a citation.

## Minor Heading
> The overriding design goal for Markdown's
> formatting syntax is to make it as readable

This text you see here is *actually- written in Markdown! To get a feel
for Markdown's syntax, type some text into the left window and
watch the results in the right.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |

### Python Code Example
```python
def hello_world():
    print("Hello, World!")
```markdown
