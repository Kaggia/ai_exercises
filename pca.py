import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

#Carichiamo il dataset e le colonne
df = pd.read_csv('data/wine.csv')
df.columns = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
print(df.head(5))
#Definiamo un oggetto Principal Component Analysis
pca = PCA(n_components=4)
#Standardizziamo il dataset
df_scaled = StandardScaler().fit_transform(df)
#Otteniamo le principal components
pmod = pca.fit_transform(df_scaled)
df_final = pd.DataFrame(pca.components_, columns=df.columns)
#Stampiamo a video il risultato
plt.figure(figsize=(14, 8))
sns.heatmap(df_final)
plt.show()
