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

#LEggiamo il dataset
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header = 0)
#Diamo un nome alle colonne
df.columns = ['sepal_length', 'sepal_width', 'petal_length' , 'petal_width', 'class' ]
print(df.head(2))
#Carichiamo il cluster
cl = df.iloc[:, :4]
print(cl.head(2))
#Creiamo un modello
kmeans = KMeans(n_clusters = 3, random_state=12345)
#Effettuiamo il clustering
kmeans.fit(cl)
#Stampiamo le labels
print(kmeans.labels_)
#Confrontiamo i dati derivanti dal clustering con i dati reali del dataset
label_dict = {'Iris-virginica': 2, 'Iris-setosa':0, 'Iris-versicolor': 1}
original_class = df['class'].map(label_dict)
print(original_class.head(5))
print(classification_report(kmeans.labels_, original_class))
#Ottimizziamo la scelta del valore K <elbow method>
cl1 = []
for i in range(1,10):
    kmeans = KMeans(n_clusters = i, init='k-means++', random_state=12345)
    kmeans.fit(cl)
    cl1.append(kmeans.inertia_)
fig2 = plt.figure()
plt.plot(range(1,10), cl1)
plt.title("Elbow Method")
plt.xlabel("Numero di Clusters")
plt.ylabel("Var")
plt.show()
