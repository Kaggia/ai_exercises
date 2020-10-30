import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from warnings import filterwarnings

#importiamo i dati
boston_ds = load_boston()

#ispezioniamo i nostri dati
#type(boston_ds) #otteniamo il tipo di dato(oggetto)
boston_ds.data.shape #otteniamo le dimensioni della matrice associata ai dati
print(boston_ds.DESCR) #Otteniamo alcune informazioni riguardo il dataset
boston_ds.data #Otteniamo la matrice del dataset
boston_ds.target #Label del dataframe

#convertiamo il dataset in pandas_dataframe
boston_df = pd.DataFrame(boston_ds.data) #I dati sono senza etichetta - NO LABEL, solo features
print(boston_df.head(5))
boston_df.columns = boston_ds.feature_names #Settiamo i nomi delle colonne del dataframe, come quelli del dataset
price = boston_ds.target #Settiamo la l'oggetto frame come colonna delle labels, estratte dal dataset

#Otteniamo un Dataframe con features e label
df = pd.concat([boston_df, pd.DataFrame(price)], axis = 1) #Effettuaiamo un JOIN sull'asse X dei dati

#Stampiamo le dimensionalità
print("Il dataframe ha dimensione totale: (" + str(df.shape[0]) + ", " + str(df.shape[1]) + " )")
print("Le features hanno dimensione: (" + str(boston_df.shape[0]) + ", " + str(boston_df.shape[1]) + " )")
print("La label ha dimensione: " + str(price.shape) )

#Lo scopo è predire il prezzo delle case in base alle features
df = df.rename(columns = {0: 'price'})#Rinominiamo la label
print(df.head(5))
lr = LinearRegression()
#Splittiamo i dati per il training ed il testing
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :13], df['price'], test_size = 0.3)
#x_train, x_test, y_train, y_test = train_test_split(FEATURES, LABEL, DIMENSIONE DEL TEST in double)
print("Shaping...")
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#Creiamo il modello di regressione, attraverso il training
lr.fit(x_train, y_train)
#Effettuiamo la predizione
prediction = lr.predict(x_test)
#Otteniamo le metriche
coefficients = pd.DataFrame(lr.coef_, boston_df.columns, columns=['coefficienti'])
print("Coefficents: ")
print(coefficients)
print("Results: ")
print("MAE: ", metrics.mean_absolute_error(y_test, prediction))
print("MSE: ", metrics.mean_squared_error(y_test, prediction))
print("RMSE: ", np.sqrt(metrics.mean_absolute_error(y_test, prediction)))
print("R2 SCORE: ", r2_score(y_test, prediction))

#Stampiamo i dati a grafico
#plt.scatter(y_test, prediction)
plt.plot(y_test, "ro")
plt.plot(prediction, 'ro', color='blue')
plt.show()

