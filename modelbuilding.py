'''''''''
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import *
from sklearn.linear_model import LinearRegression
path = 'rainfall.csv'
df = pd.read_csv(path)
df = df[['tavg', 'tmin', 'tmax', 'prcp']]
df.dropna(inplace=True)
print(df.head())
# Handling Missing Data

v1 = df['tavg'].mean()
v2 = df['tmin'].mean()
v3 = df['tmax'].mean()
v4 = df['prcp'].mean()
print(v2)
df['tavg'].fillna(v1,inplace=True)
df['tmin'].fillna(v2,inplace=True)
df['tmax'].fillna(v3,inplace=True)
df['prcp'].fillna(v4,inplace=True)

x = df[['tavg','tmin','tmax']]
yuyty = df['prcp']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
#classifier = SVC()
#classifier.fit(x_train,y_train)
lm = LinearRegression()
lm.fit(x_train,y_train)
print(lm.predict([[25.5,22.7,28.4]]))

import pandas as pd   #importing Libraries
import numpy as np
import matplotlib.pyplot as plt
ds=pd.read_csv("fall.csv")   #Reading Dataset
ds.head()     #Displaying first 5 records
ds=ds.drop(['year','month','day'], axis=1)   #Droping Unnecessary columns
ds.head()
x=ds.iloc[:,:6].values     #Assigning variables to the dependent and independent attiributes
y=ds.iloc[:,6].values
from sklearn.model_selection import train_test_split         #performing train-test split.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20,random_state=0)
#x_train
from sklearn.ensemble import RandomForestRegressor       #Model building using Random Forest
regressor = RandomForestRegressor(n_estimators=100, random_state = 0)
regressor.fit(x_train, y_train)
ypred =regressor.predict(x_test)     #Predicting using testing data on trained data.
from sklearn.metrics import r2_score #Testing accuracy
print(r2_score(y_test,ypred))
ypred1= regressor.predict([[18,16,65,1013,6,8]])      #Predicting new data record.
print(ypred1)     #prediction
print("Model Building Done!")
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(x_train,y_train)
print(lm.score(x_test,y_test))
#print(lm.predict([[25.5,22.7,28.4]]))

'''''''''