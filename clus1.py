import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('3.01. Country clusters.csv')
plt.scatter(data['Longitude'], data['Latitude'])
plt.xlim(-180, 180)
plt.ylim(-90, 90)

#print(data)
x = data.iloc[:, 1:3] # esto se usa para cortar la data segun mi interes, en el primer termino se aclaran las filas,
# en este caso pongo : porque las quiero todas, y en el segundo las columnas (la primera incluida, y la segunda sin
# incluir) en este caso quiero la columna 1 y 2.
#print(x)

kmeans = KMeans(2) # creamos la variable logica, invocando al metodo que importamos y pasamos x parametro cuantos
# clusters queremos hacer, en este caso 2.
kmeans.fit(x) # aplica el metodo KMeans con 2 clusters a la variable x

identified_clusters = kmeans.fit_predict(x) #calculamos los clusters y asigna los datos a cada uno
#print(identified_clusters)

data_with_clusters = data.copy()
data_with_clusters['Cluster'] = identified_clusters # le agregamos una columna a la tabla con los clusters de cada punto
#print(data_with_clusters)

#plt.scatter(data['Longitude'], data['Latitude'], c= data_with_clusters['Cluster'], cmap= 'rainbow') # hacemos q en el grafico aparezcan
# tantos colores como clusters diferentes hayan y pinta cada punto segun su cluster
#plt.xlim(-180, 180)
#plt.ylim(-90, 90)
#plt.show()

#ahora queremos clusterizar segun lenguaje

data_mapped = data.copy()
data_mapped['Language'] = data_mapped['Language'].map({'English': 0, 'French': 1, 'German': 2})
x1 = data_mapped.iloc[:, 3:4]

kmeans1 = KMeans(3)
kmeans1.fit(x1)
identified_clusters1 = kmeans1.fit_predict(x1)

data_with_clusters1 = data_mapped.copy()
data_with_clusters1['Cluster'] = identified_clusters1
print(data_with_clusters1)

plt.scatter(data_with_clusters1['Longitude'], data_with_clusters1['Latitude'], data_with_clusters1['Language'], c= data_with_clusters1['Cluster'], cmap= 'rainbow')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.show()


