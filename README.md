# Tianxing Zhai's Data Science Projects
Welcome to my projects presentation space! Below are overviews of all my projects:
## Machine learning projects
### Deep learning: Flower Image Classifier
![overview](https://github.com/ztx0617/Data_Science_Projects/blob/master/pictures/deep_learning_overview.png)

- Used **Pytorch** and the **DenseNet-121** pre-trained model to build a flower image classifier. The classifier can identify **102 types** of flowers with **87.6% accuracy** (on test datasets). 

- Developed two Python command line APIs for training models and making predictions.

### Unsupervised learning: Identify Customer Segments

## Other data analysis projects
### Mine and Wrangle OpenStreetMap Data
Here I mined and wrangled the Washington city map data and exported it to a relational database.

1. **Extracted the map data from XML files** and transformed them into five separate csv files. The five csv files stored nodes, node-tags, ways, way-tags and ways-nodes connection data separately. 
2. Loaded the csv files to Pandas dataframes and cleaned them. 
3. Uesed sqlite3-Python API to create a empty **relational database** which had five tables with **appropriate primary keys and foreign keys**. Then exported the data to the database. 
4. Ran some **SQL queries** to explore the database.

[Project link](https://github.com/ztx0617/Udacity_projects/tree/master/p3)
### Exploratory Data Analysis using Python
I used Python to explore the 
Titanic passengers survival data set. I found that ticket class, gender, age, companionship and 
port of embarkation were significantly associated with survival rates 
of passengers on Titanic. I had used **Pandas, Numpy, Scipy and Matplotlib** libraries to 
**transform, analyze, visualize the data and do statistical tests**.


[Project link](https://github.com/ztx0617/Udacity_projects/tree/master/p2)


### Exploratory Data Analysis using R
I used R to explore the white wine quality data set. I found that 
among all 11 chemical parameters, alcohol and density were strongest influence factors of white wine quality.
 Better wines had higher alcohol content and lower density. 
 I had used **dplyr, tidyr, ggplot2, GGally and other four R libraries** to 
**transform, analyze, visualize the data, do statistical tests and build regression models.**

[Project link](https://github.com/ztx0617/Udacity_projects/tree/master/p4)

