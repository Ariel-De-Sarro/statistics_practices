import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

data = pd.read_csv('1.02. Multiple linear regression.csv')

x = data[['SAT', 'Rand 1,2,3']]
y = data['GPA']

reg = LinearRegression()
reg.fit(x, y)
print(reg.coef_)
print(reg.intercept_)
print(reg.score(x, y)) #aca me tira el r2

# para calcular el r2 ajustado:
r2 = reg.score(x, y)

print(x.shape) # aca vemos la forma de la variable x, en este caso es 84,2
n = x.shape[0] #numero de observaciones, 84
p = x.shape[1] #numero de predictores, o features, 2

adj_r2 = 1 - (1 - r2)*(n - 1)/(n - p - 1) #formula para calcular r cuadrado ajustado
print(adj_r2)

print(f_regression(x, y)) # usamos este metodo importado para calcular los estadisticos F y los p valores
# en este caso como son dos variables independientes a analizar, nos va a dar 2 listas, la primera con
# los F estadisticos y la segundas con los p valores

p_values = f_regression(x, y)[1] #los p valores los escribe en cientifico
p_values.round(3) #redondeo los p valores a 3 cifras despues de la coma

# aca hacemos una tabla con los valores calculados
reg_summary = pd.DataFrame(data = x.columns.values, columns=['Features'])
reg_summary['Coefficients'] = reg.coef_
reg_summary['p-values'] = p_values.round(3)
print(reg_summary)

# para estandarizar la data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # aca creamos al escalador vacio
scaler.fit(x) # aca lo hacemos calcular la media y el desvio estandar y lo guarda en scaler
x_scaled = scaler.transform(x) # aca transformamos a x en estandarizado


