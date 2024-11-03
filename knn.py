import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score
data=pd.read_csv("iris.csv")
print(data.head())
x=data.iloc[:,:4]
print(x.head())
y=data.iloc[:,-1]
print(y.head())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20)
print(x_train.head())
print(x_test.head())
sc=StandardScaler()
sc.fit(x_train)
x_train=sc.transform(x_train)
x_test=sc.transform(x_test)
classifier=KNeighborsClassifier()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print(y_pred)
print(y_test)
ac=accuracy_score(y_pred,y_test)
print(ac)
cm=confusion_matrix(y_pred,y_test)
print(cm)
