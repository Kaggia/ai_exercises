import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#Definiamo KNN classifier
knn = KNeighborsClassifier(n_neighbors=3, )
#Creiamo due array di dati random
x = np.array(np.random.randint(0,5,size=12))
y = np.array(np.random.randint(6,12, size=12))

#Facciamo il reshaping in modo tale che diventino coordinate
x = x.reshape(6,2)
y = y.reshape(6,2)

#Impiliamo i due array appena creati
z = np.vstack((x, y))
#Definiamo le classi dei 12 punti nello spazio (matrice z composta da x e y impilati)
cl = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

#Training del modello
knn.fit(z, cl)

#Predict di alcuni punti
knn.predict([[8, 12]])
knn.predict([[1, 3]])

#Plotting
plt.plot(x, 'ro', color='red')
plt.plot(y, 'ro', color='green')
plt.plot(8, 12, 'ro', color='blue', markersize=12)
plt.plot(1, 3, 'ro', color='blue', markersize=12)
plt.show()




