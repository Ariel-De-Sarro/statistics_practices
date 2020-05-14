import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn import preprocessing

data = pd.read_csv('iris_with_answers.csv')
print(data.head())


x = data.copy()
x = x.iloc[:, 2:4]
x_scaled = preprocessing.scale(x)


wcss = []
for i in range(1,10):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

number_clusters = range(1, 10)
plt.plot(number_clusters, wcss)
plt.show()


kmeans = KMeans(3)
kmeans.fit(x_scaled)

identified_clusters = kmeans.fit_predict(x_scaled)
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = identified_clusters
pd.options.display.max_rows=999
print(data_with_clusters)

plt.scatter(data_with_clusters['petal_length'], data_with_clusters['petal_width'],
            c= data_with_clusters['Cluster'], cmap='rainbow')
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.show()


