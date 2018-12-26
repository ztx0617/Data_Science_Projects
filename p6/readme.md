# Summary
Here I analyzed how ticket classes, gender, age and port of embarkation affected death rates of Titanic passengers. I found passengers who were males, aduclts, embarked from Queenstown and stayed in 3rd classes were more likely to die of that shipwreck.

# Design
Firstly, I cleaned the raw data using Python and Pandas: I removed missing values, transformed the age to categorical variable and only kept 5 variables in the final data set. Then I chose vertical precnt bar in dimple.js to do data visualization, because I need to focus on the comparison of **mortality rates** and **survival rates** of different groups. To let dimple.js create plots on frequency percentage, I created a dummy variable 'count' in the final data set and let it equal to 1. Finally, I created 4 precnt bars to show the association between 'ticket classes, gender, age and port of embarkation' and 'death rates of Titanic passengers'.

# Feedbacks
1. Feedback from Ming Li

	He suggested me to control the margin of words, because he thought that my composing was so ugly. I did so and also tuned the font size and type of words to make my web more beautiful (index1.html to index2.html).
2. Feedback from James

	He thought that the imformation provied was not enough. So I added the association between port of embarkation and mortality rates to make content richer (index2.html to index_final.html).
3. Feedback from Sherri

	She praised my visualization. She said that she will never buy 3rd class cruise tickets because she saw the mortality rate of 3rd class passengers in Titanic was so high! (Just joking)

# Resources
* [RMS Titanic - Wikipedia](https://en.wikipedia.org/wiki/RMS_Titanic)
* [Data Visualization Titanic](http://bl.ocks.org/ishashankverma/62ae65230f506f2d18c7606c6b00ca59)
* [Vertical 100% Bar, Dimple.js](http://dimplejs.org/examples_viewer.html?id=bars_vertical_stacked_100pct)
* [Pandas Documents](http://pandas.pydata.org/pandas-docs/stable/index.html)
