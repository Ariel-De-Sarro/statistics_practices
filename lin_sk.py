# importamos paquetes que vamos a usar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression

# cargamos la data

data = pd.read_csv('1.01. Simple linear regression.csv')
print(data.head())  # pido que me pasen las primeras 5 lineas, para saber los nombres de las variables

x = data['SAT']
y = data['GPA']


print(x.shape) # aca me pasa la forma del vector
print(y.shape)

x_matrix = x.values.reshape(-1, 1) # aca convertimos los vectores en matrices de dos dimensiones para trabajarlos en sklearn

reg = LinearRegression() # aca instancio a reg como miembro de la clase LinearRegression
reg.fit(x_matrix, y) # aca hago la regresion, primero pongo a la variable independiente y luego la dependiente

y_pred = reg.predict(x_matrix)

print(reg)
print(reg.score(x_matrix, y)) # aca pido el R2
print(reg.coef_) # aca pido los coeficientes de la funcion lineal, en este caso nos da uno solo (x)
print(reg.intercept_) # aca nos da la ordenada al origen de la funcion. nos da siempre uno solo



predictions = reg.predict([[1740]]) # aca se pide un valor de GPA estimado para el valor SAT de 1750

new_data = pd.DataFrame(data=[1740, 1760], columns=['SAT']) #aca creo una nueba tabla con dos valores de SAT para predecir
new_data['Predicted GPA'] = reg.predict(new_data) # le agrego a la tabla new data una columna con los valores predecidos

print(float(predictions))
print(new_data)

plt.scatter(x, y)
yhat = reg.coef_*x_matrix + reg.intercept_
fig = plt.plot(x, yhat, lw=4, c='orange')
plt.show()

