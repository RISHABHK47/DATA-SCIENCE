import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
data=pd.read_csv("diabetes.csv")
print(data.head())
x=data.iloc[:,7]
print(x.head())
y=data.iloc[:,6]
print(y.head())
x=np.array(x).reshape(-1,1)
print(x)
y=np.array(y).reshape(-1,1)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
classifier=LinearRegression()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print(y_pred)
print(y_test)
r=r2_score(y_pred,y_test)
ms=mean_squared_error(y_pred,y_test)
print(r)
print(ms)
print(classifier.intercept_)
print(classifier.coef_)
plt.scatter(x_test,y_test,c="b")
plt.plot(x_test,y_pred,c="k")
plt.show()