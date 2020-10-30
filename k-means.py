import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs

#Creiamo un custer con 200 samples, 2 colonne, 4 centroidi, 1.8 di dev std
clust = make_blobs(n_samples=200, n_features =2, centers=4, cluster_std=1.8, random_state=12345)
fig = plt.figure()
plt.scatter(clust[0][:,0], clust[0][:,1], c = clust[1])
#Impostiamo il modello
kmeans = KMeans(n_clusters = 4)
#Adattiamo il modello, si usa clust[0] perchè clust da solo non funziona
kmeans.fit(clust[0])
#Stampiamo le labels che sono uscite fuori dal processo di clustering
print(kmeans.labels_)
#Ottimizziamo la scelta del valore K <elbow method>
cl1 = []
for i in range(1,10):
    kmeans = KMeans(n_clusters = i, init='k-means++', random_state=12345)
    kmeans.fit(clust[0])
    cl1.append(kmeans.inertia_)
fig2 = plt.figure()
plt.plot(range(1,10), cl1)
plt.title("Elbow Method")
plt.xlabel("Numero di Clusters")
plt.ylabel("Var")
plt.show()
