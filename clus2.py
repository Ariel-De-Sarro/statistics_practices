import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('Countries_exercise.csv')

x = data.iloc[:, 1:3]
kmeans = KMeans(7)
kmeans.fit(x)

identified_clusters = kmeans.fit_predict(x)
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = identified_clusters

#plt.scatter(data['Longitude'], data['Latitude'], c= data_with_clusters['Cluster'], cmap= 'rainbow')
#plt.xlim(-180, 180)
#plt.ylim(-90, 90)
#plt.show()

# para calcular el grafico de WCSS vs Nº de clusters hacemos:

wcss = []
for i in range(1,15):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

number_clusters = range(1, 15)
plt.plot(number_clusters, wcss)
plt.show()
# aca en el grafico buscamos el codo, y elegimos ese numero de clusters, en este caso seria 3 o 4

