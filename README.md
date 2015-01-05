Titanic-Kaggle-
===============

This repository contains the code written for the Titanic Kaggle competition, in which we are asked to predict survival rates on Titanic using a host of features identifying each passenger (sex, boarding class, socioeconomic background, age, etc). My best submission score was 0.77512 (78% correct prediction). 

The files are described below. 

1) data_trends.ipynb
This ipython notebook uses matplotlib to plot certain aspects of the dataset in order to visualize trends in an effort to understand which features are relevant for prediction. To view the ipython notebook, visit the link http://nbviewer.ipython.org/github/andylee024/Titanic-Kaggle-/blob/master/data_trends.ipynb

2) titanic.py
This is the main script and executes high level objectives of our prediction scheme including reading in data, cleaning data and prediction. 

3) clean.py
This script handles cleaning the dataset using the pandas library. More specifically, we apply concepts like feature engineering as well as null-value filling to prepare the dataset for our machine learning algorithms. 

4) learn.py
We apply all our learning algorithms in this script using the library scikit-learn and output our prediction files to be entered into the competition. 
