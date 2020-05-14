import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() # sobreescribe los graficos de matplotlib x los de seaborn

raw_data = pd.read_csv('2.01. Admittance.csv')
data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes': 1, 'No': 0}) # cambiamos de Yes/No to 1/0

y = data['Admitted']
x1 = data['SAT']
x = sm.add_constant(x1)
reg_log = sm.Logit(y, x)
results_log = reg_log.fit()

np.set_printoptions(formatter={'float': lambda x: '{0:0.2f}'.format(x)})
print(results_log.predict()) # aca pedimos una lista de los valores estimados

np.array(data['Admitted']) # esta es una lista de los valores reales de la columna Admitted

# vamos a comparar ambas listas
results_log.pred_table() # esto nos arroja la Confusion Matrix

cm_df = pd.DataFrame(results_log.pred_table()) # aca convierto la matriz en un dataframe
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0', 1: 'Actual 1'})
print(cm_df)

# para calcular la presicion:
cm = np.array(cm_df)
accuracy_train = (cm[0, 0] + cm[1, 1])/cm.sum()
print(accuracy_train)

# en este caso: para 67 de las observaciones, el modelo predijo 0 cuando el verdadero valor era 0
# para 87 observaciones, el modelo predijo 1 cuando el valor real era 1
# 7 veces el modelo predijo 0 cuando el verdadero valor era 1, y 7 veces predijo 1 cuando el valor era 0
# el modelo nos dio 154/168 veces bien, es una precision del 91.6%




plt.scatter(x1, y, color='C0')
plt.show()
