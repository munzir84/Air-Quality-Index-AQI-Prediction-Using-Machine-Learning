import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
df = pd.read_csv("final_data.csv")
scaler = StandardScaler()
scaler.fit(df)
scaled_features = scaler.transform(df)
df1 = pd.DataFrame(scaled_features)
print(df1.head())
print(df1.isnull().sum())
X = df1[[0,1,2,3,4]]
y = df1[[5]]
print(X.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=12)
print("Data Pre-Processing Done!")