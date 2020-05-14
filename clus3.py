import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn import preprocessing

data = pd.read_csv('3.12. Example.csv')
plt.scatter(data['Satisfaction'], data['Loyalty'])
plt.show()

# si calculamos los clusters antes, toma a la variable Satisfaction como la mas importante, xq los valores son mucho mas
# altos que los de Loyalty, por lo tanto hay q estandarizar Satisfaction

x = data.copy()
x_scaled = preprocessing.scale(x)

# ahora que ya estan ambas variables estandarizadas, hacemos el metodo del codo

wcss = []
for i in range(1,10):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

number_clusters = range(1, 10)
plt.plot(number_clusters, wcss)
plt.show()

# segun el metodo del codo, elijo dividir la data en 4 clusters.

kmeans_new = KMeans(4)
kmeans_new.fit(x_scaled)
clusters_new = x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)  # aca le agrego una columna a la tabla de x
# (la q es sin estandarizar Satisfaction) para ver los resultados reales

plt.scatter(clusters_new['Satisfaction'], clusters_new['Loyalty'], c= clusters_new['cluster_pred'], cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()




