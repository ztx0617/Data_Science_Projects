# Machine Learning Project: Identify Fraud from Enron Email and Finance Dataset
## Overview
Here I built a classifier to identify fraud suspects from all Enron staffs. 
Below is how I bulit my machine learning pipeline:

1. Familiar with the data.
2. Wrangle the data.
3. Do feature selection and optimization.
4. Do model selection and evaluation.

After tried three algorithms **(Naive Bayes, SVM and Decision Tree)** and tuned some parameters,
I selected **Decision Tree** as the final algorithm for my classifier. 
After **1000-fold cross validation**, my classifier yielded results with **84.83% accuracy,
41.20% precision and 32.22% recall rate**.

[Open the report](https://github.com/ztx0617/Udacity_projects/blob/master/p5/Enron%20Machine%20Learing%20Project_Tianxing%20Zhai.ipynb)

## Contents
* **Enron Machine Learing Project_Tianxing Zhai.ipynb**

	The machine learning report written with jupyter notebook. 
	It shows the whole process of my machine learning pipeline.
	You can browse it directly online by clicking on it.
	
* **enron_final.pkl**

	The original Enron Email and Finance Dataset. It is a pkl file.
	You can load it using Python pickle.