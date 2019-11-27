import collections
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

names = ['caseNum', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10',
         'age', 'qChat', 'sex', 'ethnicity', 'jaundice', 'familyASD',
         'whoCompleted', 'ASD']
dataset = pd.read_csv("Toddler Autism.csv", skiprows = [0], names=names)
dataset.drop(dataset.columns[[0]], axis = 1, inplace = True)

#Divide dataset into attributes and labels
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 17].values

labelEncoder = LabelEncoder()
x[:, 12] = labelEncoder.fit_transform(x[:, 12])
x[:, 13] = labelEncoder.fit_transform(x[:, 13])
x[:, 14] = labelEncoder.fit_transform(x[:, 14])
x[:, 15] = labelEncoder.fit_transform(x[:, 15])
x[:, 16] = labelEncoder.fit_transform(x[:, 16])

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.20)

xTrainKMeans = xTrain
xTestKMeans = xTest

#K Means
print("\nKmeans:")
kmTrain = KMeans(n_clusters = 2)
kmTrain.fit(xTrainKMeans)
centers = kmTrain.cluster_centers_
print(centers)
kmTrainLabels = kmTrain.labels_
print(collections.Counter(kmTrainLabels))

kmTest = KMeans(n_clusters = 2)
kmTest.fit(xTestKMeans)
centers = kmTest.cluster_centers_
print(centers)
kmTestLabels = kmTest.labels_
print(collections.Counter(kmTestLabels))
