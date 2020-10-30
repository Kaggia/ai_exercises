import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#Definiamo KNN come modello, in particolare il K
knn = KNeighborsClassifier(n_neighbors = 3)
#LEggiamo il dataset
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header = 0)
#Diamo un nome alle colonne
df.columns = ['sepal_length', 'sepal_width', 'petal_length' , 'petal_width', 'class' ]
print(df.head(2))

#Estraiamo le colonne relative alle features
X = df.iloc[:, 0:4]
#Scaliamo i valori
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled_X = pd.DataFrame(scaler.fit_transform(X))
print(scaled_X.head(2))

#Creiamo un set di train e test
x_train, x_test, y_train, y_test = train_test_split(scaled_X, df['class'], test_size=0.3, random_state=12345)

#Creiamo il modello, allenandolo con i dati di training
knn.fit(x_train, y_train)

#Effettuiamo una predizione con il modello appena creato
pred = knn.predict(x_test)
#Otteniamo il report relativo alla prediction
print(classification_report(y_test, pred))

#Tramite il Metodo Elbow cerchiamo il miglior numero di Neighbors in un range 1%6
err = []
for i in range (1, 6):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    err.append(np.mean(pred_i != y_test))
    
#Grafichiamo il risultato del metodo elbow vedendo quello con il minor errore
plt.figure(figsize = (10,6))
plt.plot(range(1, 6), err, color='green', linestyle='dotted', marker= 'o', markerfacecolor='red', markersize=8)
plt.xlabel("Numero di k")
plt.ylabel('Tasso di errore')
plt.show()

#Lanciamo una prediction con il tasso di errore PIU' elevato
knn_max_error = KNeighborsClassifier(n_neighbors = 2)
knn_max_error.fit(x_train, y_train)
pred_max_error = knn_max_error.predict(x_test)

print(classification_report(y_test, pred_max_error))

#Lanciamo una prediction con il tasso di errore MENO elevato
knn_min_error = KNeighborsClassifier(n_neighbors = 5)
knn_min_error.fit(x_train, y_train)
pred_min_error = knn_min_error.predict(x_test)

print(classification_report(y_test, pred_min_error))



