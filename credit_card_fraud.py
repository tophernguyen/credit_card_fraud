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



def main():
    files = configure()
    data = eda(files)
    xTrain, xTest, yTrain, yTest = split(data)
    rcc(xTrain, xTest, yTrain, yTest)


def configure():

    cwd = os.getcwd()
    files = glob.glob('*.csv')

    print(f'Opening the following directory {cwd}')
    print(f'The files found in this directory are: {files}')

    return(files)

def eda(files):
    for file in files:
        data = pd.read_csv(file)
        shape = data.shape
        description = data.describe()
        fraud = data[data['Class'] == 1]
        valid = data[data['Class'] == 0]
        FraudPercentage = len(fraud)/float(len(data))
        fraud_description = fraud.Amount.describe()
        valid_description = valid.Amount.describe()

        with open('01_cc_eda.txt', 'a') as file:
            file.write(f'Data Shape = {shape}\n')
            file.write(f'Data Description = {description}\n\n')
            file.write(f'{FraudPercentage = :.2%}\n')
            file.write(f'Fraud Transactions = {len(fraud)}\n')
            file.write(f'Valid Transactions = {len(valid)}\n\n')
            file.write(f'Amount details of the fraudulent transaction = {fraud_description}\n\n')
            file.write(f'Amount details of the valid transaction = {valid_description}\n\n')

        corrmat = data.corr()
        fig = plt.figure(figsize = (12, 9))
        sns.heatmap(corrmat, vmax = .8, square = True)
        plt.savefig('02_cc_eda_corr_map.png', bbox_inches='tight', pad_inches=0.0)

        return(data)

def split(data):
    X = data.drop(['Class'], axis = 1)
    Y = data["Class"]
    xData = X.values
    yData = Y.values
    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size = 0.2, random_state = 78)
    return(xTrain, xTest, yTrain, yTest)

def rcc(xTrain, xTest, yTrain, yTest):
    rfc = RandomForestClassifier()
    rfc.fit(xTrain, yTrain)
    yPred = rfc.predict(xTest)
    acc = accuracy_score(yTest, yPred)
    prec = precision_score(yTest, yPred)
    rec = recall_score(yTest, yPred)
    f1 = f1_score(yTest, yPred)
    MCC = matthews_corrcoef(yTest, yPred)
    cr = classification_report(yTest, yPred)
    with open('03_cc_output.txt', 'a') as file:
        file.write(f'The model used is Random Forest classifier\n')
        file.write(f'The accuracy is {acc} \n')
        file.write(f'The precision is {prec}\n')
        file.write(f'The recall is {rec}\n')
        file.write(f'The F1-Score is {f1}\n')
        file.write(f'The Matthews correlation coefficient is {MCC}\n\n')
        file.write(f'The Classification Report is {cr}\n')

    LABELS = ['Normal', 'Fraud']
    conf_matrix = confusion_matrix(yTest, yPred)
    plt.figure(figsize =(12, 12))
    sns.heatmap(conf_matrix, xticklabels = LABELS, 
                yticklabels = LABELS, annot = True, fmt ="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig('04_cc_conf_matrix.png', bbox_inches='tight', pad_inches=0.0)


if __name__ == '__main__':
    main()