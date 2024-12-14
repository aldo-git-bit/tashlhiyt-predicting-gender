# The gender system of Tashlhiyt: Using supervised learning to predict noun gender

## Introduction
This repository gives the supplementary materials for our 2025 Linguistic Analysis article (see citation below) on gender assignment in Tashlhiyt. Tashlhiyt is an Amazigh language of Morocco in which nouns are assigned to one of two gender categories: masculine or feminine. Gender assignment is the system drawing on the form and meaning of a noun to assign these categories. In this project, we treat gender assignment in Tashlhiyt as an supervised learning problem where we try to predict gender from a set of 14 linguistic features. The principal finding reported in our article is that Tashlhiyt is a mixed system that uses both morphological (word structure) and phonological (sound structure) to assign gender. The data and models in this repository can be used to validate our assumptions and also extend it to new investigations. 

## Usage - repository contents
The supplementary materials are organized into the folders below. File names contain certain strings with the following meanings: cv(cross-validation), nonhp(non hyper-parameterized model), rel_impX(relative importance of model X using importance methods). PIYUSH: you can drop this if you rename the files.

### data
Gives the data used by the model to predict gender, including data frames derived from feature engineering. Original data set, tashdata, has all the core data, and tashdata_fields gives detailed field explanations.

### code
Gives all the code for our analyses, including feature engineering, training with cross-validation using different gender systems (including the three category system we don't report in the article), correlation analysis, feature importance, hyper-paramterizing, and visualization. Each of the files describe what they are performing in the first comment.

### plots
Repository of all plots used to visualize the results, including the results of many models not discussed in the article.

### results
Data tables for all results, including the results for some models not reported in the article, like models using one-hot encoding.

### notebook
Some useful notebooks for exploring the data and establishing conclusions for feature engineering. 

### Package requirements
All packages used are listed in the requirements.txt file. `pip install -r requirements.txt` can be used to install all packages.

## Citation
If the data or models are used for another project or publication, please cite the research article below.

Alderete, John, Piyush Agarwal, Kaye Holubowsky, Abdelkrim Jebbour. 2025. The gender system of Tashlhiyt: Using supervised learning to predict noun gender. Linguistic Analysis [volume, pages pending]

## Acknowledgements
We are grateful to Donna Gerdts, Lana Leal, and Rachid Ridouane and two anonymous Linguistic Analysis reviewers for questions and comments on earlier drafts of this article. This research was supported in part by an Insight grant from the Social Science and Humanities Research Council of Canada (435-2020-0193).
