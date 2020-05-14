import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_csv('Country clusters standardized.csv', index_col='Country') #especificamos el nombre de una columna como
# indice del data frame

x_scaled = data.copy()
x_scaled = x_scaled.drop(['Language'], axis= 1)

sns.clustermap(x_scaled, cmap= 'mako')
plt.show()

# aca vemos el grafico y los colores. analizamos por separado latitud y longitud y ahi analizamos cuantos clusters
# tomar, en este caso lo mejor seria separarlos en 3.

