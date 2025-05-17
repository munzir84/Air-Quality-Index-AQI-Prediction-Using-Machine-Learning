import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor       #Model building using Random Forest
regressor = RandomForestRegressor(n_estimators=100, random_state = 0)
df = pd.read_csv("final_data.csv")
print(df.head())
print(df.isnull().sum())
X = df.drop(['AQI'],axis = 1)
y = df["AQI"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=12)
regressor.fit(X_train, y_train)
print("Model Building Done!")
ypred =regressor.predict(X_test)
#print(regressor.score(X_test,y_test))
from sklearn.metrics import r2_score #Testing accuracy
print("Accuracy of Random Forest Model is:",round((r2_score(y_test,ypred))*100,2),"%")
import pickle
import joblib
joblib.dump(regressor,"modelpred.pkl")
model = joblib.load("modelpred.pkl")
