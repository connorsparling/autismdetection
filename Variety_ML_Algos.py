import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
scaler = StandardScaler()
scaler = StandardScaler()
scaler.fit(xTrain)

xTrainKMeans = xTrain
xTestKMeans = xTest
xTrain = scaler.transform(xTrain)
xTest = scaler.transform(xTest)


#K Nearest Neighbors
classifier = KNeighborsClassifier(n_neighbors = 13)
classifier.fit(xTrain, yTrain)
yPredKNN = classifier.predict(xTest)
print("KNN---------------------------------------------------")
print("Confusion Matrix:")
print(confusion_matrix(yTest, yPredKNN))
print("Classification Report:")
print(classification_report(yTest, yPredKNN))

#Naive Bayes
gnb = GaussianNB()
gnb.fit(xTrain, yTrain)
yPredNB = gnb.predict(xTest)
print("Naive Bayes-------------------------------------------")
print("Confusion Matrix:")
print(confusion_matrix(yTest, yPredNB))
print("Classification Report:")
print(classification_report(yTest, yPredNB))

#Gradient Boosting
gbc = GradientBoostingClassifier(n_estimators = 20, learning_rate = 0.5,
                                 max_features = 2, max_depth = 2,
                                 random_state = 0)
gbc.fit(xTrain, yTrain)
yPredGB = gbc.predict(xTest)
print("Gradient Boosting--------------------------------------")
print("Confusion Matrix:")
print(confusion_matrix(yTest, yPredGB))
print("Classification Report:")
print(classification_report(yTest, yPredGB))
featureImpGB = pd.Series(gbc.feature_importances_,
                       index=names[1:18]).sort_values(ascending = False)
print("Feature Importance: ")
print(featureImpGB)

#Random Forest
clf = RandomForestClassifier(n_estimators = 100)
clf.fit(xTrain, yTrain)
yPredRF = clf.predict(xTest)
#print("Accuracy: ", metrics.accuracy_score(yTest, yPredRF))
print("Random Forest------------------------------------------")
print("Confusion Matrix:")
print(confusion_matrix(yTest, yPredRF))
print("Classification Report:")
print(classification_report(yTest, yPredRF))
featureImpRF = pd.Series(clf.feature_importances_,
                       index=names[1:18]).sort_values(ascending = False)
print("Feature Importance: ")
print(featureImpRF)

'''
#K Means
print("\nKmeans:")
km = KMeans(n_clusters = 2, n_jobs = 18, random_state = 21)
km.fit(xTrainKMeans)
centers = km.cluster_centers_
print(centers)
'''

'''
error = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(xTrain, yTrain)
    predI = knn.predict(xTest)
    error.append(np.mean(predI != yTest))

plt.figure(figsize = (12,6))
plt.plot(range(1, 40), error, color = 'red', linestyle = 'dashed', marker = 'o',
         markerfacecolor = 'blue', markersize = 10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
'''
