import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

data = pd.read_csv('real_estate_price_size_year.csv')
print(data.head())

x = data[['size', 'year']]
y = data['price']

reg = LinearRegression()
reg.fit(x, y)

print(f'los coeficientes son {reg.coef_}')
print(f'la ordenada al origen es {reg.intercept_}')

# para calcular el r2 ajustado:
r2 = reg.score(x, y)

n = x.shape[0] #numero de observaciones, 84
p = x.shape[1] #numero de predictores, o features, 2

adj_r2 = 1 - (1 - r2)*(n - 1)/(n - p - 1) #formula para calcular r cuadrado ajustado
print(f'El r2 es {r2}')
print(f'El R2 ajustado es {adj_r2}')

p_values = f_regression(x, y)[1] #los p valores los escribe en cientifico


# aca hacemos una tabla con los valores calculados
reg_summary = pd.DataFrame(data = x.columns.values, columns=['Features'])
reg_summary['Coefficients'] = reg.coef_
reg_summary['p-values'] = p_values.round(3)
print(reg_summary)

