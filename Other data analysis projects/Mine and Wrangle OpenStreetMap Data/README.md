# Wrangle OpenStreetMap Data
## Overview
Here I mined and wrangled the Washington city map data and exported it to a relational database.

1. **Extracted the map data from XML files** and transformed them into five separate csv files. The five csv files stored nodes, node-tags, ways, way-tags and ways-nodes connection data separately. 
2. Loaded the csv files to Pandas dataframes and cleaned them. 
3. Uesed sqlite3-Python API to create a empty **relational database** which had five tables with **appropriate primary keys and foreign keys**. Then exported the data to the database. 
4. Ran some **SQL queries** to explore the database.

[Open the report](https://github.com/ztx0617/Data_Science_Projects/blob/master/Other%20data%20analysis%20projects/Mine%20and%20Wrangle%20OpenStreetMap%20Data/Project_Tianxing%20Zhai.ipynb)

## Contents
* **Project_Tianxing Zhai.ipynb**

	The wrangling report written with jupyter notebook. 
	It has narratives and related Python codes.
	You can browse it directly online by clicking on it.
* **Other files mentioned in the report**
	
	They can't be uploaded to github because they are too big. 
	You can get them from my Google drive if you are interested.
	
	[Google drive link](https://drive.google.com/drive/folders/1XVr5d5amiZuzhLWw6JSiPOn4Qm9CeMQq?usp=sharing)