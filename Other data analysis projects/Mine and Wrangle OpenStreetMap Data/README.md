# Wrangle OpenStreetMap Data
## Overview
Here I wrangled the Washington city map data and exported it to a relational database.


Firstly, I **extracted the map data from XML files** and transformed them into five separate csv files. 
The five csv files stored nodes, node-tags, ways, way-tags and ways-nodes connection data 
separately. Secondly, I loaded the csv files to Pandas dataframes and cleaned them. 
Thirdly, I uesed sqlite3-Python API to create a empty **relational database** which had five tables
with **appropriate primary keys and foreign keys**. Then I exported the data to the database. 
Finally, I ran some **SQL queries** to explore the database.

[Open the report](https://github.com/ztx0617/Udacity_projects/blob/master/p3/Project_Tianxing%20Zhai.ipynb)

## Contents
* **Project_Tianxing Zhai.ipynb**

	The wrangling report written with jupyter notebook. 
	It has narratives and related Python codes.
	You can browse it directly online by clicking on it.
* **Other files mentioned in the report**
	
	They can't be uploaded to github because they are too big. 
	You can get them from my Google drive if you are interested.
	
	[Google drive link](https://drive.google.com/drive/folders/1XVr5d5amiZuzhLWw6JSiPOn4Qm9CeMQq?usp=sharing)