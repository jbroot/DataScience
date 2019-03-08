import pandas as pd
import re
#import warnings
#warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib.pyplot as plt
plt.rc("font", size=15)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

from imblearn.over_sampling import SMOTE

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("weatherAUS.csv")


print("Start Date: "+df["Date"].min()+" End Date: "+df["Date"].max())

df['Month'] = df.apply(lambda x:  re.search('-(.+)-', x['Date']).group(1), axis=1)

print(df.groupby('Month')['MinTemp'].mean())

#histogram
x = df.groupby('Month')['MinTemp'].mean().values
y = sorted(df['Month'].unique())
plt.figure(1)
plt.xlabel("Numeric month")
plt.ylabel("Average Minimum Temperature")
plt.bar(y, x)

print("\nNumber of unique cities: " + str(df['Location'].unique().size))

#5 cities with highest average rainfall
print(df.groupby('Location')['Rainfall'].mean().sort_values(ascending=False)[:5])

#5 different univariate values
print("\nHumidity3pm mode: " + str(df['Humidity3pm'].mode().values) + "\nEvaporation mean: "+ str(df['Evaporation'].mean())
      +"\nPressure3pm max: "+str(df['Pressure3pm'].max()) + "\nMinTemp min: "+str(df['MinTemp'].min())
      +"\nMaxTemp standard deviation: "+str(df['MaxTemp'].std()))

print("Correlation between MinTemp and Rainfall: "+str(df['MinTemp'].corr(df['Rainfall'])))
plt.figure(2)
plt.xlabel("Minimum  Temperature")
plt.ylabel("Rainfall")
plt.scatter(df['MinTemp'], df['Rainfall'])
print("There is little correlation between MinTemp and Rainfall. Correlation is about 0.1 which means the correlation is weak.")


rndnCities = ['Townsville', 'Dartmoor', 'Darwin', 'Brisbane', 'Albany']

#rainfall means for the 5 cities
#df[df['Location'].isin(rndnCities)].groupby('Location', as_index=False)['Rainfall'].mean()

#correlation matrix
df_dummies = pd.get_dummies(df[df['Location'].isin(rndnCities)]['Location'])
df_new = pd.concat([df['Rainfall'], df_dummies], axis=1)
corr = df_new.corr()

#what does the correlation matrix mean
print("\nEach cell in the correlation matrix indicates how strongly correlated its column's name and row's name are with each other.\n"
      "-1 indicates strongly negative, 0 is no correlation, and 1 is strongly positive")
print(str(rndnCities) + " all have little correlation with the rainfall.")

#Darwin had the highest correlation
#beautiful scatterplot
plt.figure(3)
plt.ylabel("Rainfall")
plt.scatter(df[df['Location']=="Darwin"]['Location'], df[df['Location']=="Darwin"]['Rainfall'])

#logistic regression model that predicts RainTomorrow.

#Columns that you used in your model.
lgDf = df[['Pressure3pm', 'Humidity3pm', 'RainToday', 'Pressure3pm', 'RainTomorrow']]
lgDf = lgDf.dropna()
X = lgDf.loc[:, lgDf.columns != 'RainTomorrow']
y = lgDf.loc[:, lgDf.columns == 'RainTomorrow']

X = X.replace("Yes", 1).replace("No", 0)
y = y.replace("Yes", 1).replace("No", 0)

lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X,y)
#accuracy of the logistic regression
#lr.score(X, y)

#confusion matrix
yPred = lr.predict(X[:])
confMat = confusion_matrix(y, yPred)
print("\nColumns I decided:\n"+str(X.columns)+"\n"
      +"Accuracy: "+str(lr.score(X,y))+
      "\nConfusion Matrix:\n"+str(confMat))

#Use Recursive Feature Elimination to determine the best columns ('features') to use in another model that predicts RainTomorrow.

lgDf = df
lgDf = lgDf.replace("Yes", 1).replace("No", 0)
lgDf = lgDf.select_dtypes(['number'])
lgDf = lgDf.dropna()
X = lgDf.loc[:, lgDf.columns != 'RainTomorrow']
y = lgDf.loc[:, lgDf.columns == 'RainTomorrow']

X = X.replace("Yes", 1).replace("No", 0)
y = y.replace("Yes", 1).replace("No", 0)

os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
ycolumns = y_train.columns

os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=ycolumns)

dfVars = lgDf.columns.values.tolist()
yCol=['RainTomorrow']
XCol=[j for j in dfVars if j not in yCol]

lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X,y)

rfe = RFE(lr, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())

#remove columns RFE deemed unfit
x = 0
for i in rfe.support_:
    if not i:
        X = X.drop(x)
    else:
        x += 1

#accuracy of the logistic regression
lr.score(X, y)

#confusion matrix
yPred = lr.predict(X)
confMat = confusion_matrix(y, yPred)

print("\nRFE decided columns:\n"+str(X.columns)+"\nAccuracy with RFE: "+str(lr.score(X, y))+"\nRFE Confusion Matrix:\n"+ str(confMat))

