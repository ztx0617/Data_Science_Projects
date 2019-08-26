# Unsupervised learning: Identify Customer Segments
## Overview
![overview](https://github.com/ztx0617/Data_Science_Projects/blob/master/pictures/segments_results.png)

- Used the **k-means** algorithm to cluster the Germany general population demographics data into 13 segments:
	- Cleaned the data, did the feature engineering.
	- Used PCA to collapse 218 features into 129 principal components. The 129 principal components explained 95% of the variability.
	- Used the Elbow Method to choose the best k for the k-means algorithm: k = 13.
	- Trained the clustering model on the Germany general population demographics data.
- Applied the same clustering model to the demographics data of customers of a mail-order company. 
- Compared the difference of segments distribution between general population and customers.
- Found that customers of the mail-order company tended to be richest, middle-aged males with these personalities: combative, dominant and critical minded.

[Open the project](https://github.com/ztx0617/Data_Science_Projects/blob/master/Machine%20leanring/Identify%20Customer%20Segments/Identify_Customer_Segments.ipynb)

## Contents
* **Identify\_Customer\_Segments.ipynb**

	The jupyter notebook file with codes and results. You can browse it online by clicking on it.

* **data/data.zip**
	- **Udacity\_AZDIAS\_Subset.csv**
		-  Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
	-  **Udacity\_CUSTOMERS\_Subset.csv** 
		-  Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
	- **Data\_Dictionary.md**
		- Detailed information file about the features in the provided datasets.
	- **AZDIAS\_Feature\_Summary\.csv**
		- Summary of feature attributes for demographics data; 85 features (rows) x 4 columns