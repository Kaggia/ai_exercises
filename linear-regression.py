import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from warnings import filterwarnings

#Loading a simple Dataframe
df = pd.DataFrame(
    {
        'stud' : [1, 2, 3, 4, 5, 6, 4, 1, 2, 1, 3], 
        'red' : [12000, 23000, 35000, 47000, 55000, 67000, 43000, 15000, 25000, 15000, 31500]
    }
)
print(df)
#Load data on a simple chart
plt.plot(df['stud'], df['red'], "ro")
#Loading data on matrix
mat = np.matrix(df)
x = mat[:, 0]
y = mat[:, 1]
#Loading Linear Regression Obj and fitting the data
lr = LinearRegression()
lr.fit(x, y)
print(lr.intercept_)
print(lr.coef_)
#Prediction
print(lr.predict([[4]]))
plt.plot(4, lr.predict([[4]]), "s")
plt.legend(['Training data', 'Predicted'])
plt.show()