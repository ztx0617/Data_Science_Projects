# What makes a popular mobile game?
## Overview
The problem comes from a fictitious music streaming company called Sparkify. Sparkify has millions of users who use the company's app to stream thier favorite songs everyday. Each user uses either the Free-tier with advertisement between the songs or the premium subscription plan. Users can upgrade, downgrade or cancel thier service at any time. Hence, it is important to predict users who are going to cancel subscriptions (churned users). Knowing who will be churned users, we can take actions to keep paid users before it is too late.

In this project, I will use Spark and Python to analyze the Sparkify user event data. I will also build a machine learning model to predict who are going to cancel. Because the original data is too big (12GB), a tiny subset (128MB) was extracted and will be used in this project. All the analysis and model building will be done on my PC.

[Link to my blog post](https://blog.csdn.net/Star_Zhai/article/details/104722521)

## Results
In this project, I used Spark and Python to build a model to identify churned users. Here are my process and findings:

1. I loaded the data (128MB subset of the 12GB data) and did the exploratory data analysis. The data had 286500 rows and 18 variables. I analyzed some important variables and decided to keep 6 variables. Among 6 variables, 'page' was the most important one. It recorded what a user did specifically using the APP.


2. I removed rows with missings in 'userId' and 'page'. I also found there were blanks in the 'userId'. Blanks should be considered as missing and removed. I also removed free users, because we only interested in paid users who will cancel. After removal, the data had 222433 rows and 6 variables.


3. To create features, I did some aggregation on users to make one user per row. I created 10 features, 8 of them were counts of events per interaction hour, 1 is number of interaction hours per record hour, 1 is the gender. I also created the label. It represented whether a user confirmed the cancellation (1 is yes, 0 is no). The final dataset had 164 rows (164 unique users), 10 features and 1 label. Among 164 unique users, 31 of them (~19%) confirmed the cancellation and be labeled as 1.


4. I splited the dataset into training (80%) and testing (20%). The distribution of positive cases in both datasets was checked. Both datasets had about 19% positive cases. Then a StandardScaler was fitted using the training dataset and used to transform both the training and testing data.


5. I tried three different algorithms to bulid the classifier: Logistic Regression (LogisticRegression), Linear Support Vector Machines (LinearSVC) and Gradient-Boosted Trees (GBTClassifier). The GBTClassifier had the best performance on testing data (F1: 0.763, Recall: 0.778, Precision: 0.753).


6. I tunned three hyperparameters of the GBTClassifier using the Spark CrossValidator. The bset maxDepth is 7, stepSize is 0.05 and featureSubsetStrategy is 'all'. On testing data, the performance of best model is: F1: 0.722, Recall: 0.750, Precision: 0.703. After tuning, the performance gets worse. I provided three possible reasons: (1) Small training and testing data. (2) Unbalance labels and (3) Overfitting.


In a word, I developed a classifer to predict users who are going to unsubscribe. The best model had F1 of 0.763 on the testing data.



## Contents
* **Spark Project.ipynb**

	The jupyter notebook file with full codes and results of the analysis. 

* **mini\_sparkify\_event\_data.zip**

	The zip file with the 128MB subset json data file in it. 

* **df_final.csv**

	The csv data file after feature engineering and ready for machine learning. 

* **README.md**

	The markdown file that introduce the project.

## Libraries
I used the Python v3.6 to do all the analysis. I used these Python libraries:

	Pandas v1.0
	Seaborn v0.10.0
	Datetime
	PySpark v2.4.5

