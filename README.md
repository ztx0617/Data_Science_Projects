# Udacity Data Analyst Nano Degree Projects
## P2: Investigate a Dataset
I used Python to explore the 
Titanic passengers survival data set. I found that ticket class, gender, age, companionship and 
port of embarkation were significantly associated with survival rates 
of passengers on Titanic. I had used **Pandas, Numpy, Scipy and Matplotlib** libraries to 
**transform, analyze, visualize the data and do statistical tests**.
## P3: Wrangle OpenStreetMap Data
Here I wrangled the Washington city map data and exported it to a relational database.


Firstly, I **extracted the map data from XML files** and transformed them into five separate csv files. 
The five csv files stored nodes, node-tags, ways, way-tags and ways-nodes connection data 
separately. Secondly, I loaded the csv files to Pandas dataframes and cleaned them. 
Thirdly, I uesed sqlite3-Python API to create a **relational database** which had five tables
with **appropriate primary keys and foreign keys**. Then I exported the data to it. 
Finally, I ran some **SQL queries** to explore the database.
## P4: Exploratory Data Analysis using R
I used R to explore the white wine quality data set. I found that 
among all 11 chemical parameters, alcohol and density are strongest influence factors of white wine quality.
 Better wines have higher alcohol content and lower density. 
 I had used **dplyr, tidyr, ggplot2, GGally and other four R libraries** to 
**transform, analyze, visualize the data, do statistical tests and build regression models.**
## P5: Machine Learning Project: Identify Fraud from Enron Email and Finance Dataset
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
## P6: Interactive Data Visualization on Website
Here I used **D3.js and Dimple.js** to visualize my discovery of P1 **on website.** 
I created 4 precnt bars to show the association between 'ticket classes, gender, 
age and port of embarkation' and 'death rates of Titanic passengers'. 
**All the plots were interactive**. You can put your cursor on the plot to see exact counts.