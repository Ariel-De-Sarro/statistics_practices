import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler #se usa para estandarizar

data = pd.read_csv('1.02. Multiple linear regression.csv')

x = data[['SAT', 'Rand 1,2,3']]
y = data['GPA']
scaler = StandardScaler() # aca creamos al escalador vacio
scaler.fit(x) # aca lo hacemos calcular la media y el desvio estandar y lo guarda en scaler
x_scaled = scaler.transform(x) # aca transformamos a x en estandarizado

reg = LinearRegression()
reg.fit(x_scaled, y)

reg_summary = pd.DataFrame([['Intercept'], ['SAT'], ['Rand 1,2,3']], columns=['Features'])
reg_summary['Weights'] = reg.intercept_, reg.coef_[0], reg.coef_[1]
print(reg_summary)

#aca hago predicciones con la data estandarizada
new_data = pd.DataFrame(data=[[1700, 2], [1800, 1]], columns=['SAT', 'Rand 1,2,3'])
new_data_scaled = scaler.transform(new_data)
print(reg.predict(new_data_scaled))

# si queremos sacar la variable Rand 1,2,3
reg_simple = LinearRegression()
x_simple_matrix = x_scaled[:, 0].reshape(-1, 1) # aca selecciono solo la primer columna y le doy forma de matriz
reg_simple.fit(x_simple_matrix, y)
reg_simple.predict(new_data_scaled[:, 0].reshape(-1, 1)) # aca buscamos que haga predicciones con la tabla de nueva data pero sin usar la columna de Rand
# aca nos da valores muy similares a los predecidos usando las dos columnas, porque la variable rand 1,2,3 tiene muy poco peso, casi nulo