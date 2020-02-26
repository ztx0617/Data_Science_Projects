# Identify message for help in disasters


## Introduction
In this project, I built ETL and machine learning pipelines to classify messages for help in disasters. I also built a website for data visualization and a web App to identify messages from users' input.

The datasets used for training were provided by [Figure Eight](https://www.figure-eight.com), whcih contained real messages sent during disasters and their respective labels. The project will help aid agencies identify what disaster victims need. 


## Instruction
1. To process the data for machine learning, run the following commands in the project's root directory. The processed data will be saved as **DisasterResponse.db** database file in the data directory.

	`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2. To train and test the machine learning model, run the following commands in the project's root directory. The trained pipeline will be saved as **classifier.pkl** pickle file in the model directory.
	
	`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        

2. To see the data visualization and web App, run the following commands in the app directory, and go to [http://0.0.0.0:3001/](http://0.0.0.0:3001/)
    
	`python run.py`



## Contents

* **README.md**

	The markdown file that introduce the project.

* **app**
	* **templates**
		
		**go.html** and **master.html**
		
		The html web templates for data visualization and web App.
	* **run.py**
		
		The python script to build the server and make data visualization and web App.

* **data**
	* **disaster_categories.csv**
		
		The csv data file of the categories of messages.
	* **disaster_messages.csv**
		
		The csv data file of the messages and genre.
	* **DisasterResponse.db**
		
		The database file which stored the processed data. It can be re-created by running the process_data.py python script.
	* **process_data.py**
		
		The python script which processes disaster\_categories.csv and disaster_messages.csv to make the DisasterResponse.db.
* **models**
	* **classifier.pkl**
		
		The pickle which stored the trained machine learning pipeline.
	* **train_classifier.py**
		
		The python script which trains and saves the machine learning pipeline using the DisasterResponse.db data. 