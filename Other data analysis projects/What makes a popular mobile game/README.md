# What makes a popular mobile game?
## Overview
The mobile games industry is developing so fast, with companies spending vast amounts of money on the development and marketing of these games to an equally large market. According to the [newzoo](https://newzoo.com/insights/articles/the-global-games-market-will-generate-152-1-billion-in-2019-as-the-u-s-overtakes-china-as-the-biggest-market/), mobile gaming (smartphone and tablet) is largest segment of gaming market in 2019, which is 68.5 billion dollars (45% of the global games market). Of this, 54.9 billion dollars will come from smartphone games.

The data used here includes 17007 strategy games on the Apple App Store. It was collected on the 3rd of August 2019, using the iTunes API and the App Store sitemap, by the kaggle user Tristan. I downloaded it from the [kaggle datasets](https://www.kaggle.com/tristan581/17k-apple-app-store-strategy-games).

Using this data set, I find what makes a popular strategy game.

[Link to my blog post](https://blog.csdn.net/Star_Zhai/article/details/104395871)

## Results
I analyzed the Apple App Store Strategy Games data. Firstly, I defined a parameter of popularity: User Rating Count per Day. Then I studied the association between popularity and charging mode, rating, size. I found that games with the free download and IAP (In-app purchase) charging mode, highly rated and larger size are more likely to be popular.



## Contents
* **games.ipynb**

	The jupyter notebook file with full codes and results of the analysis. 

* **README.md**

	The markdown file that introduce the project.

## Libraries
I used the Python v3.6 to do all the analysis. I also used these Python libraries:

	Pandas v1.0
	Matplotlib v3.1.3
	Seaborn v0.10.0
	Datetime
	NumPy v1.19
	Scipy v1.4.1

