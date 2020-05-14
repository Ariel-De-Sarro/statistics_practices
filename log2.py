import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv('Example_bank_data.csv')
data = raw_data.copy()

data['y'] = data['y'].map({'yes': 1, 'no': 0})
data = data.drop(['Unnamed: 0'], axis=1)

y = data['y']
x1 = data['duration']
x = sm.add_constant(x1)
reg_log = sm.Logit(y, x)
results_log = reg_log.fit()
print(results_log.summary())

plt.scatter(x1, y, color='C0')
plt.xlabel('Duration', fontsize = 20)
plt.ylabel('Subscription', fontsize = 20)
plt.show()
