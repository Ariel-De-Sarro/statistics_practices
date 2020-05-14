import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()
"""
data = pd.read_csv('1.02. Multiple linear regression.csv')
y = data['GPA']
x1 = data[['SAT', 'Rand 1,2,3']]
x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
print(results.summary())


data =pd.read_csv('real_estate_price_size_year.csv')
print(data)
y = data['price']
x1 = data[['size', 'year']]
x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
print(results.summary())

yhat = 227.7009*'size' + '2916.7853'*'year' - 5.772e-06

"""
raw_data =pd.read_csv('1.03. Dummies.csv')
data = raw_data.copy()
data['Attendance'] = data['Attendance'].map({'Yes': 1, 'No': 0}) #esto se hace para cambiar la variable categorica a numerica
# print(data.describe())

y = data['GPA']
x1 = data[['SAT', 'Attendance']]
x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
print(results.summary())
# aca creamos un data frame para predecir valores
new_data = pd.DataFrame({'const': 1, 'SAT': [1700, 1670], 'Attendance': [0, 1]})
new_data = new_data[['const', 'SAT', 'Attendance']]
new_data.rename(index={0: 'Bob', 1: 'Alice'})

predictions = results.predict(new_data)
print(predictions)

predictionsdf= pd.DataFrame({'Predictions': predictions})
joined = new_data.join(predictionsdf)
joined_renamed = joined.rename(index={0: 'Bob', 1: 'Alice'})
print(joined_renamed)


plt.scatter(data['SAT'], y, c= data['Attendance'], cmap= 'RdYlGn_r')
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
fig = plt.plot(data['SAT'], yhat_no, lw= 2, c='red')
fig = plt.plot(data['SAT'], yhat_yes, lw= 2, c='blue')
plt.show()



"""

raw_data =pd.read_csv('real_estate_price_size_year_view.csv')
data = raw_data.copy()
data['view'] = data['view'].map({'Sea view': 1, 'No sea view': 0}) #esto se hace para cambiar la variable categorica a numerica
# print(data.describe())

y = data['price']
x1 = data[['size', 'year', 'view']]
x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
print(results.summary())

"""
