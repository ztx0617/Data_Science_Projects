# Tianxing Zhai's Data Science Projects
Welcome to my projects presentation space! Below are overviews of all my finished projects:
## Machine learning projects
### Deep learning: Flower Image Classifier
![overview](https://github.com/ztx0617/Data_Science_Projects/blob/master/pictures/deep_learning_overview.png)

- Used **Pytorch** and the **DenseNet-121** pre-trained model to build a flower image classifier. The classifier can identify **102 types** of flowers with **87.6% accuracy** (on test datasets). 

- Developed two Python command line APIs for training models and making predictions.

[Project link](https://github.com/ztx0617/Data_Science_Projects/tree/master/Machine%20leanring/Flower%20Image%20Classifier)
### Unsupervised learning: Identify Customer Segments
![overview](https://github.com/ztx0617/Data_Science_Projects/blob/master/pictures/segments_results.png)



- Used the **k-means** algorithm to cluster the Germany general population demographics data into 13 segments. 
- Applied the same clustering model to the demographics data of customers of a mail-order company. 
- Compared the difference of segments distribution between general population and customers.
- Found that customers of the mail-order company tended to be be richest, middle-aged males with these personalities: combative, dominant and critical minded.

[Project link](https://github.com/ztx0617/Data_Science_Projects/tree/master/Machine%20leanring/Identify%20Customer%20Segments)

### NLP: Identify message for help in disasters

In this project, I built ETL and machine learning pipelines to classify messages for help in disasters. I also built a website for data visualization and a web App to identify messages from users' input.

The datasets used for training were provided by [Figure Eight](https://www.figure-eight.com), whcih contained real messages sent during disasters and their respective labels. 

The project will help aid agencies identify what disaster victims need.

[Project link](https://github.com/ztx0617/Data_Science_Projects/tree/master/Machine%20leanring/Identify%20message%20for%20help%20in%20disasters)

### Spark: Identify churned users for a music streaming APP

The problem comes from a fictitious music streaming company called Sparkify. Sparkify has millions of users who use the company's app to stream thier favorite songs everyday. Each user uses either the Free-tier with advertisement between the songs or the premium subscription plan. Users can upgrade, downgrade or cancel thier service at any time. Hence, it is important to predict users who are going to cancel subscriptions (churned users). Knowing who will be churned users, we can take actions to keep paid users before it is too late.

In this project, I will use Spark and Python to analyze the Sparkify user event data. I will also build a machine learning model to predict who are going to cancel. Because the original data is too big (12GB), a tiny subset (128MB) was extracted and will be used in this project. All the analysis and model building will be done on my PC.


[Project link](https://github.com/ztx0617/Data_Science_Projects/tree/master/Machine%20leanring/Identify%20churned%20users%20for%20a%20music%20streaming%20APP)



## Other data analysis projects
### Mine and Wrangle OpenStreetMap Data
Here I mined and wrangled the Washington city map data and exported it to a relational database.

1. **Extracted the map data from XML files** and transformed them into five separate csv files. The five csv files stored nodes, node-tags, ways, way-tags and ways-nodes connection data separately. 
2. Loaded the csv files to Pandas dataframes and cleaned them. 
3. Used sqlite3-Python API to create a empty **relational database** which had five tables with **appropriate primary keys and foreign keys**. Then exported the data to the database. 
4. Ran some **SQL queries** to explore the database.

[Project link](https://github.com/ztx0617/Data_Science_Projects/tree/master/Other%20data%20analysis%20projects/Mine%20and%20Wrangle%20OpenStreetMap%20Data)
### Exploratory Data Analysis using Python
I used Python to explore the 
Titanic passengers survival data set. I found that ticket class, gender, age, companionship and 
port of embarkation were significantly associated with survival rates 
of passengers on Titanic. I had used **Pandas, Numpy, Scipy and Matplotlib** libraries to 
**transform, analyze, visualize the data and do statistical tests**.


[Project link](https://github.com/ztx0617/Data_Science_Projects/tree/master/Other%20data%20analysis%20projects/Investigate%20a%20Dataset)


### Exploratory Data Analysis using R
I used R to explore the white wine quality data set. I found that 
among all 11 chemical parameters, alcohol and density were strongest influence factors of white wine quality.
 Better wines had higher alcohol content and lower density. 
 I had used **dplyr, tidyr, ggplot2, GGally and other four R libraries** to 
**transform, analyze, visualize the data, do statistical tests and build regression models.**

[Project link](https://github.com/ztx0617/Data_Science_Projects/tree/master/Other%20data%20analysis%20projects/Exploratory%20Data%20Analysis%20using%20R)



### What makes a popular mobile game?
The mobile games industry is developing so fast, with companies spending vast amounts of money on the development and marketing of these games to an equally large market. According to the [newzoo](https://newzoo.com/insights/articles/the-global-games-market-will-generate-152-1-billion-in-2019-as-the-u-s-overtakes-china-as-the-biggest-market/), mobile gaming (smartphone and tablet) is largest segment of gaming market in 2019, which is 68.5 billion dollars (45% of the global games market). Of this, 54.9 billion dollars will come from smartphone games.

The data used here includes 17007 strategy games on the Apple App Store. It was collected on the 3rd of August 2019, using the iTunes API and the App Store sitemap, by the kaggle user Tristan. I downloaded it from the [kaggle datasets](https://www.kaggle.com/tristan581/17k-apple-app-store-strategy-games).

Using this data set, I find what makes a popular strategy game.Firstly, I defined a parameter of popularity: User Rating Count per Day. Then I studied the association between popularity and charging mode, rating, size. I found that games with the free download and IAP (In-app purchase) charging mode, highly rated and larger size are more likely to be popular.


[Project link](https://github.com/ztx0617/Data_Science_Projects/tree/master/Other%20data%20analysis%20projects/What%20makes%20a%20popular%20mobile%20game)
