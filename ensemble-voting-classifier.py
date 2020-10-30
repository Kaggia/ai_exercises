import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#importiamo i modelli da confrontare
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
#Importiamo il metodo di ensemble di Voting
from sklearn.ensemble import VotingClassifier
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
#Creiamo il classificatore  tramite Voting
forest= RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1234, n_jobs=2)
cl1 = LogisticRegression(random_state=1)
cl2 = GaussianNB()
Voting_classifier = VotingClassifier(estimators=[('lr', cl1), ('rf', forest), ('gauss', cl2)], voting='hard')
#Otteniamo il valore media dei risultati dei classificatori deboli
for clf, label in zip([cl1, forest, cl2, Voting_classifier], ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print("Accuracy di ", label ," è di: ", scores.mean())