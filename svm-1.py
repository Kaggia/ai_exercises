import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

#Definiamo due array di features
x = np.array(np.random.randint(0, 5, size=12))
y = np.array(np.random.randint(7, 14, size=12))
#Effettuiamo il reshaping in modo tale da farli diventare coordinate 
#Stackiamo le due amtrici di punti x ed y in un unica matrice
x = x.reshape(6, 2)
y = y.reshape(6, 2)
z = np.vstack((x, y))
#Definiamo la label di classificazione
cl = np.array([0,0,0,0,0,0,1,1,1,1,1,1])

#Grafichiamo i punti
fig1 = plt.figure()
plt.plot(x, 'ro', color='blue')
plt.plot(y, 'ro', color='green')
#Creiamo il modello ed effettuiamo il training
svc = SVC()
svc.fit(z, cl)
#Effettuiamo due prediction
print("Il punto (8,13) appartiene alla classe: ", svc.predict([[8, 13]]))
print("Il punto (0,3) appartiene alla classe: ", svc.predict([[0, 3]]))

#Grafichiamo grazie alla libreria mlxtends le regioni che si sono
#formate grazie all'iperpiano trovato
fig2 = plt.figure()
plot_decision_regions(z, cl, clf=svc, res=0.2, legend=2)
plt.show()





