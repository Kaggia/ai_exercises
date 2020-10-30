#"Outlook", "Temperature", "Humidity", "Wind", "Play"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_decision_regions

df = pd.read_csv('data/golf.csv')
df.columns = ['Outlook','Temperature','Humidity','Wind','Play']

#Effettuare il mapping dei valori presenti nel df in numeri 
outlook_dict = {'sunny' : 1, 'overcast' : 2, 'rain' : 3}
play_dict = {'yes' : 1, 'no' : 0}

df['Outlook'] = df['Outlook'].map(outlook_dict)
df['Play'] = df['Play'].map(play_dict)
print(df.head(5))

#Definiamo il modello
dtc = DecisionTreeClassifier()
#Definiamo il set di dati
X = df.iloc[:, 0 : 4]
y = df.iloc[:, 4]
print("Shapes: ")
print(X.shape)
print(y.shape)
#Adattiamo il modello ai dati
dtc.fit(X, y)
#Effettuiamo una predizione < giorno in cui non si gioca >
print("Predizione( Giorno in cui non si dovrebbe giocare): ", dtc.predict([[1, 85.0, 89.0, False]]))
#Effettuiamo una predizione < giorno in cui si gioca >
print("Predizione( Giorno in cui si dovrebbe giocare): ", dtc.predict([[3, 68.0, 80.0, False]]))
#Grafichiamo i dati
features = list(df.columns[1:])
dot_data = tree.export_graphviz(dtc, out_file=None, feature_names=features, filled=True, rounded=True,
                                special_characters=True)    
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("results/decision-tree.png")
print("Tree has been graphicated: results/decision-tree.png")
   