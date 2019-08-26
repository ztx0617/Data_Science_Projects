# Deep learning: Flower Image Classifier
## Overview
![overview](https://github.com/ztx0617/Data_Science_Projects/blob/master/pictures/deep_learning_overview.png)

- Used **Pytorch** and the **DenseNet-121** pre-trained model to build a flower image classifier:
	- Load and pre-proccess the data: For training data, do random scaling, cropping, and flipping and then do normalization of RGB channels. For testing data, do resizing and normalization only.
	- Used the DenseNet-121 pre-trained model and defined my own classifier to build the model framework.
	- Trained and tested the performance of the model.
	- Saved the model to checkpoint.pth.
	- Used the model to do sanity checking by visualizing images with class probability.
- The classifier can identify **102 types** of flowers with **87.6% accuracy** (on test datasets). 
- Developed two Python command line APIs for training models and making predictions.

[Open the project](https://github.com/ztx0617/Data_Science_Projects/blob/master/Machine%20leanring/Flower%20Image%20Classifier/Image%20Classifier%20Project.ipynb)

## Contents
* **Image Classifier Project.ipynb**

	The jupyter notebook file with codes and results. You can browse it online by clicking on it.
* **checkpoint.pth**

	The saved trained model.

* **Python command line API**
	* **train.py**
	
		The Python command line application for training the model.
	* **predict.py**
	
		The Python command line application for using the model for classification.