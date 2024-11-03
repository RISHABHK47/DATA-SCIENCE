import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
data=pd.read_csv("iris.csv")
print(data.head())
x=data.iloc[:,:4]
print(x.head())
y=data.iloc[:,-1]
print(y.head())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20)
classifier=DecisionTreeClassifier()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print(y_pred)
print(y_test)
ac=accuracy_score(y_pred,y_test)
print(ac)
tree.plot_tree(classifier)
plt.show()