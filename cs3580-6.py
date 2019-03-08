import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import seaborn as sns


import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

ogtest = pd.read_csv("A5_test.csv")
ogtrain = pd.read_csv("A5_train.csv")

test = ogtest.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1).replace('male', 1).replace('female', 0)
train = ogtrain.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1).replace('male', 1).replace('female', 0)
test = pd.get_dummies(test.dropna())
train = pd.get_dummies(train.dropna())

trainX = train.loc[:, train.columns != 'Survived']
testX = test.loc[:, train.columns != 'Survived']
trainXF = train.loc[:, train.columns != 'Fare']
testXF = test.loc[:, train.columns != 'Fare']

trainY = train['Survived'].values
testY = test['Survived'].values
trainYF = train['Fare'].values
testYF = test['Fare'].values
#testYF = np.asarray(test['Fare'].values, dtype="|S6")

#converted to ndarray
scaler = StandardScaler()
scaler.fit(trainX)
trainX = scaler.transform(trainX)
testX = scaler.transform(testX)


#linear regression
lm = linear_model.LinearRegression()
#survival threshold = 0.5
model = lm.fit(trainX, trainY)
predictions = lm.predict(testX)
#intercept = lm.score(testY, testX)
predictions = [1 if x>=0.5 else 0 for x in predictions]
lineSurv = accuracy_score(predictions, testY)

model = lm.fit(trainXF, trainYF)
predictions = lm.predict(testXF)
#Fare threshold within 1 std of testYF ~= 27
correct = 0
for i,j in zip(predictions, testYF):
    if i-j <=testYF.std() and i-j >=testYF.std():
        correct += 1

#neural net
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30), max_iter=1000)
mlp.fit(trainX, trainY.ravel())
predictions = mlp.predict(testX)


mlpSurv = accuracy_score(testY, predictions)
mlpSurv = round(mlpSurv * testY.shape[0])

mlpF = MLPRegressor(hidden_layer_sizes=(30,30,30), max_iter=1000)
mlpF.fit(trainXF, trainYF.ravel())
predictions = mlpF.predict(testXF)
mlpFare = mlpF.score(testXF, testYF)
#mlpFare = accuracy_score(testYF, predictions)

#decision tree
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)dt.fit(X, y)
