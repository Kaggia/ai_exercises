import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn import datasets

#Il RandomForest utilizza una n-upla di Alberi decisionali per il suo miglioramento, 
#Effettua delle iterazioni affinchè emerga il miglior albero

#Importiamo il dataset
iris = datasets.load_iris()
#Creiamo il test e train data
X = iris.data[:, 0:4]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
#Creiamo il classificatore  tramite Boosting
forest= RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1234, n_jobs=2)
#Otteniamo il valore media dei risultati dei classificatori deboli
scores = cross_val_score(forest, X_train, y_train)
print("La media dei classificatori weak è di: ", scores.mean())
#Adattiamo il modello ai dati di training
forest.fit(X_train, y_train)
#Prediciamo un valore
pred = forest.predict(X_test)
#Otteniamo i risultati
print(classification_report(y_test, pred))

