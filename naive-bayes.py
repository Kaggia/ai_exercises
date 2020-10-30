import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#LEggiamo il dataset
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header = 0)
#Diamo un nome alle colonne
df.columns = ['sepal_length', 'sepal_width', 'petal_length' , 'petal_width', 'class' ]
print(df.head(2))
#Vediamo quanti valori può assumere la label
print(set(df['class']))
#Mappiamo i valori presenti nella colonna label
cl = {'Iris-virginica' : 0, 'Iris-versicolor' : 1, 'Iris-setosa' : 2}
df['class'] = df['class'].map(cl)
print(df.head(2))
#Definiamo il modello
nb = GaussianNB()
#Definiamo i dati di train e di test
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values
print(X)
print(y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
print("Shapes of train and test data:")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
#Adattiamo il modello agli oggetti di training
nb.fit(x_train, y_train)
#Effettuiamo una predizione
pred = nb.predict(x_test)
#Otteniamo il report sui risultati del modello
print(classification_report(y_test, pred))