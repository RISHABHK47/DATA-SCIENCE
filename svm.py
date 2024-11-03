import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
data=pd.read_csv("iris.csv")
print(data.head())
x=data.iloc[:,:4]
y=data.iloc[:,-1]
print(x.head())
print(y.head())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20)
classifier=SVC(kernel="linear")
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print(y_pred)
print(y_test)
cm=confusion_matrix(y_test,y_pred)
ac=accuracy_score(y_test,y_pred)
print(cm)
print(ac)
