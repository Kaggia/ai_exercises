#id,diagnosis_result,radius,texture,perimeter,area,smoothness,compactness,symmetry,fractal_dimension
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_decision_regions

#Carichiamo i dati su un dataframe
df = pd.read_csv('data/Prostate_Cancer.csv')
df.columns = ['id','diagnosis_result','radius','texture','perimeter','area','smoothness','compactness','symmetry','fractal_dimension']
print(df.head(5))
#Convertiamo la colonna dei risultati in numerici
cl = {'B' : 0, 'M' : 1}
df['diagnosis_result'] = df['diagnosis_result'].map(cl)
print(df.head(5))
print("Shape del dataframe: ", df.shape)
#Definiamo il modello
svm = SVC()
#Definiamo i dati di train e di test
X = df.iloc[:, 1:10].values
y = df.iloc[:, 1].values
print(y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
print("Shapes of train and test data:")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
#Adattiamo il modello agli oggetti di training
svm.fit(x_train, y_train)
#Effettuiamo una predizione
pred = svm.predict(x_test)
#Otteniamo il report sui risultati del modello
print(classification_report(y_test, pred))
#Cerchiamo di migliorare i parametri
parameters = {'C': [0.1, 1, 10, 100, 1000], 'kernel' : ['rbf'], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), parameters, refit=True, verbose=1)
grid.fit(x_train, y_train)
print(grid.best_params_)
pred_grid = grid.predict(x_test)
print(classification_report(y_test, pred_grid))