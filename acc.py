import pandas as pd
import numpy as np
from matplotlib.pyplot import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor       #Model building using Random Forest
df = pd.read_csv("final_data.csv")
lr = LinearRegression()
X = df.drop(['AQI'],axis = 1)
y = df[["AQI"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
lr.fit(X_train, y_train)
from sklearn.metrics import r2_score #Testing accuracy
ypred = lr.predict(X_test)
lracc = round((r2_score(y_test,ypred))*100,2)
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
from sklearn.metrics import r2_score #Testing accuracy
ypred = dt.predict(X_test)
dtacc = round((r2_score(y_test,ypred))*100,2)
df = pd.read_csv("final_data.csv")
print(df.head())
print(df.isnull().sum())
X = df.drop(['AQI'],axis = 1)
y = df["AQI"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=12)
regressor = RandomForestRegressor(n_estimators=100, random_state = 0)
regressor.fit(X_train, y_train)
ypred =regressor.predict(X_test)
#print(regressor.score(X_test,y_test))
from sklearn.metrics import r2_score #Testing accuracy
rfacc = round((r2_score(y_test,ypred))*100,2)
v = [lracc,dtacc,rfacc]
y1 = [v[0],v[1],v[2]]
x = ["Linear Regression","Decision Tree","Random Forest"]
f = figure()
width = 0.3
fig = f.add_axes([0.1,0.1,0.8,0.8])
bars = fig.bar([0,1,2],y1,label="Accuracy",width=width)
fig.legend()
fig.bar_label(bars,y1)
xticks([0,1,2],x)
show()