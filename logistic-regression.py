import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix


#Carichiamo il dataset
dataset = pd.read_csv('data/diabetes.csv')
dataset.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
print(dataset.head(5))
#Estraiamo le features e la label
X = dataset.iloc[:, 0:7].values
y = dataset.iloc[:, 8].values

#Splittiamo i dati in modo tale da avere train e test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Scaliamo i dati in modo tale da avere numeri omogenei
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
#Si effettua la predizione
prediction = classifier.predict(X_test)
#Si listano i risultati del modello
print("Results: ")
print("MAE: ", metrics.mean_absolute_error(y_test, prediction))
print("MSE: ", metrics.mean_squared_error(y_test, prediction))
print("RMSE: ", np.sqrt(metrics.mean_absolute_error(y_test, prediction)))
print("Accuracy(Confusion matrix method): ", str((confusion_matrix(y_test, prediction)[0][0]+confusion_matrix(y_test, prediction)[1][1]) / X_test.size))
print("R2 SCORE: ", r2_score(y_test, prediction))
#Stampiamo i dati a grafico
#plt.scatter(y_test, prediction)
plt.plot(y_test, "ro")
plt.plot(prediction, 'ro', color='blue')
plt.show()