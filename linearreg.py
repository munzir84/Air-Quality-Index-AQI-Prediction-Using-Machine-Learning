import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
df = pd.read_csv("final_data.csv")
print(df.head())
print(df.isnull().sum())
lr = LinearRegression()
X = df.drop(['AQI'],axis = 1)
y = df[["AQI"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
lr.fit(X_train, y_train)
#print(lr.predict([[94.52,32.66,24.39,67.39,111.33]]))
#print(lr.predict([[79.84, 28.68, 13.85, 48.49, 97.07]]))
print(lr.score(X_test,y_test))
from sklearn.metrics import r2_score #Testing accuracy
ypred = lr.predict(X_test)
print("Accuracy of Linear Regression Model is:",round((r2_score(y_test,ypred))*100,2),"%")