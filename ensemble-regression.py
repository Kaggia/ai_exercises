import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import datasets
from mlxtend.plotting import plot_decision_regions


#Il Bagging utilizza un classificatore per il suo miglioramento, 
#Quindi fornisce alla fine una media dei risultati di classificatori

#Importiamo il dataset
iris = datasets.load_iris()
#Creiamo il test e train data
X = iris.data[:, 0:4]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
#Definiamo l'ensemble model
bagging = BaggingRegressor(LinearRegression(), n_jobs=1, n_estimators=1000, max_features=0.8)
#Otteniamo lo score di validazione
scores = cross_val_score( bagging, X_train, y_train)
print(scores.mean())
#Eseguiamo il training
bagging.fit(X_train, y_train)
#Effettuiamo una predizione
pred = bagging.predict(X_test)
#Otteniamo i risultati
print("R2 SCORE: ", r2_score(y_test, pred))