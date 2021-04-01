# Disaster Response Pipeline Project

Yu Tao 2021/03/28

Udacity Data Scientist Nanodegree Project 2: Disaster Response Pipelines

### Table of Contents

- 1. Installation
- 2. Project Motivation
- 3. File Descriptions
- 4. Instructions
- 5. Licensing, Authors, Acknowledgements

### Installation:
The program is running on Python 3, the following packages are needed: numpy, pandas, matplotlib, sqlalchemy, nltk, re, pickle and sklearn.

### Project Motivation:
In the Udacity Data Scientist Nanodegree course, I've learned and built many data engineering skills. In this project, I apply these skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. The goal is to build a machine learning model to identify if certain messages are related to disaster or not, if so, we can label these messages into 36 categories. This would be of great help for some disaster relief agencies. After building and training such a model, a web service which can label new messages from users' input will be launched, which helps people to get alerted during natural disasters. 

### File Descriptions:
The file structure is as follows:

	app
    
	|- template
    
	| |- master.html # main page of web app
    
	| |- go.html # classification result page of web app
    
	|- run.py # Flask file that runs app
    
	data
    
	|- disaster_categories.csv # data to process
    
	|- disaster_messages.csv # data to process
    
	|- process_data.py
    
	|- InsertDatabaseName.db # database to save clean data to
    
    models
    
	|- train_classifier.py
    
	|- classifier.pkl # saved model
    
	README.md

In the data folder, disaster_categories.csv and disaster_messages.csv are the original disaster data from Figure Eight, process_data.py includes code to clean the data.

In the model folder, train_classifier.py includes the code to deploy a ML classification model and train on the cleaned data.

In the app folder, run.py include the code to launch a web app to visualize the classification on the data.

ETL Pipeline Preparation.ipynb and ML Pipeline Preparation.ipynb are the jupyter notebooks used in the preparation step of this project.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing, Authors, Acknowledgements
This app was completed as part of the Udacity Data Scientist Nanodegree.
The disaster response data used to train the classification model was provide by Figure Eight.