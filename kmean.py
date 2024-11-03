import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data=pd.read_csv("iris.csv")
print(data.head())
x=data.iloc[:,:4]
print(x.head())
km=KMeans(n_clusters=3)
km.fit(x)
y_pred=km.predict(x)
print(y_pred)
centroid=km.cluster_centers_
print(centroid)


