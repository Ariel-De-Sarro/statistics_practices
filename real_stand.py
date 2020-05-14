import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('real_estate_price_size_year.csv')

x = data[['size', 'year']]
y = data['price']

scaler = StandardScaler() # aca creamos al escalador vacio
scaler.fit(x) # aca lo hacemos calcular la media y el desvio estandar y lo guarda en scaler
x_scaled = scaler.transform(x) # aca transformamos a x en estandarizado

reg = LinearRegression()
reg.fit(x_scaled, y)

reg_summary = pd.DataFrame([['Intercept'], ['size'], ['year']], columns=['Features'])
reg_summary['Weights'] = reg.intercept_, reg.coef_[0], reg.coef_[1]
print(reg_summary)

new_data = [[750,2009]]
new_data_scaled = scaler.transform(new_data)
print(reg.predict(new_data_scaled))