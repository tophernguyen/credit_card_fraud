# Credit Card Fraud
In this folder it's just a full script that runs on a single .csv file to to see if it can predict CC fraud with Machine Learning

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score 
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix

# Program
Main - Runs the whole program
Configure - Reads the .csv
Exploratory Data Analysis - Just regular statistics on how many fraud and how many are true transactions
Split the data set.  I broke it up 80/20.  Using random seed of 78.  
Run it through a Random Forest Classifier and then compare my test results vs the 20 percent actual results that I split off
Run some analyses between the two.

# Real World Application
I'd probably have two files.  One to train the machine learning.  The other to analyze daily transactions.  Any flags would automatically sent to fraud department.
