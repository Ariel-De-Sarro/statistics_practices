import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
sns.set()

raw_data = pd.read_csv('Bank_data.csv')
data = raw_data.copy()

data = data.drop(['Unnamed: 0'], axis=1)
data['y'] = data['y'].map({'yes': 1, 'no': 0})
print(data.head())

test = pd.read_csv('Bank_data_testing.csv')
test = test.drop(['Unnamed: 0'], axis=1)
test['y'] = test['y'].map({'yes': 1, 'no': 0})
print(test.head())

y = data['y']
x1 = data['duration']
x = sm.add_constant(x1)
reg_log = sm.Logit(y, x)
results_log = reg_log.fit()
print(results_log.summary())

y_hat = test['y']
x1_test = test['duration']
x_test= sm.add_constant(x1_test)
reg_log_test = sm.Logit(y_hat, x_test)
results_log_test = reg_log_test.fit()
print(results_log_test.summary())

plt.scatter(x1, y, color='C0')
plt.xlabel('Duration', fontsize = 20)
plt.ylabel('Subscription', fontsize = 20)
plt.show()

np.set_printoptions(formatter={'float': lambda x: '{0:0.2f}'.format(x)})
print(results_log.predict()) # aca pedimos una lista de los valores estimados

np.array(test['Admitted']) # aca hacemos una lista con los valores de Admitted de la data test

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