import collections
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

names = ['caseNum', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10',
         'age', 'qChat', 'sex', 'ethnicity', 'jaundice', 'familyASD',
         'whoCompleted', 'ASD']
dataset = pd.read_csv("Toddler Autism.csv", skiprows = [0], names=names)
dataset.loc[dataset['ASD'] == 'No', 'ASD'] = 0
dataset.loc[dataset['ASD'] == 'Yes', 'ASD'] = 1
dataset.drop(dataset.columns[[0]], axis = 1, inplace = True)

#Divide dataset into attributes and labels
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 17].values

labelEncoder = LabelEncoder()
#x[:, 11] = labelEncoder.fit_transform(x[:, 11])
x[:, 12] = labelEncoder.fit_transform(x[:, 12])
x[:, 13] = labelEncoder.fit_transform(x[:, 13])
x[:, 14] = labelEncoder.fit_transform(x[:, 14])
x[:, 15] = labelEncoder.fit_transform(x[:, 15])
x[:, 16] = labelEncoder.fit_transform(x[:, 16])

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.25)

#MLP model, one hidden layer.
#Input = 17 Neuron
#Hidden = 50
#Output = 2
mlp = MLPClassifier(hidden_layer_sizes = (50),
                    solver = 'sgd',
                    learning_rate_init = 0.01,
                    max_iter = 1000,
                    random_state = 13)

mlp.fit(xTrain, yTrain)

yPred = mlp.predict(xTest)
print("Confusion Matrix:")
print(confusion_matrix(yTest, yPred))
print("Classification Report:")
print(classification_report(yTest, yPred))

print("\nAccuracy: ", metrics.accuracy_score(yTest, yPred))
print("Loss: ", metrics.log_loss(yTest, yPred))
print("Precision: ", metrics.average_precision_score(yTest, yPred))
print("F1 Score: ", metrics.f1_score(yTest, yPred))
print("Recall: ", metrics.recall_score(yTest, yPred))

