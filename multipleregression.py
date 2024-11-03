import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

data=pd.read_csv("diabetes.csv")
print(data.head())
X=data.iloc[:,:-1]
print(X.head())
y=data.iloc[:,-1]
print(y.head())
X=np.array(X).reshape(-1,1)
print(X)
y=np.array(y).reshape(-1,1)
print(y)
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.3, random_state=0)
ml=LinearRegression()
ml.fit(X_train,y_train)
y_pred=ml.predict(X_test)
print(y_pred)
print(y_test)




