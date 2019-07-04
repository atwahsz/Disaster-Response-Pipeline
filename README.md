# Disaster Response Pipeline Project

## Installation
This repository was written in HTML and Python , and requires the following Python packages: 
 pandas, numpy, re, pickle, nltk, flask, json, plotly, sklearn, sqlalchemy, sys,  warnings.

## Project Overview
This code is designed to iniate a  web app which an emergency operators could exploit during a disaster (e.g. an earthquake or Tsunami), to classify a disaster text messages into several categories which then can be transmited to the responsible entity

The app built to have an ML model to categorize every message received
## File Description:
* **process_data.py**: This python excutuble code takes as its input csv files containing message data and message categories (labels), and then creates a SQL database
* **train_classifier.py**: This code trains the ML model with the SQL data base
* **ETL Pipeline Preparation.ipynb**:  process_data.py development procces
* **ML Pipeline Preparation.ipynb**: train_classifier.py. development procces
* **data**: This folder contains sample messages and categories datasets in csv format.
* **app**: cointains the run.py to iniate the web app.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
   Or Go to http://localhost:3001/

## Screenshots

***Screenshot 1: App analysis of the data base***
![Screenshot 1](https://github.com/atwahsz/Disaster-Response-Pipeline/blob/master/readme-pics/Capture2.JPG)

***Screenshot 2: App word search Page***
![Screenshot 2](https://github.com/atwahsz/Disaster-Response-Pipeline/blob/master/readme-pics/Capture.JPG)


## Licensing, Authors, Acknowledgements
This app was completed as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).