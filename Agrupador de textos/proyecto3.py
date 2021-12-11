import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import data
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
#from jqmcvi import base
from sklearn.metrics import davies_bouldin_score
import scikit_posthocs as sp



#import data
inFileKmeans=pd.read_csv('matrizfreq.csv')
inFileHier=inFileKmeans


#quitar la columna del identificador
inFileKMeans_var=inFileKmeans.drop(['document'],axis=1)
inFileHier_var=inFileKmeans_var

#Mostrar una descripción sobre los datos
print(inFileKMeans.info())
#Mostrar todos los datos
print(inFileKMeans.describe())

#Normalizar los valores de las caracteristicas
inFileKMeans_norm=(inFileKMeans_var-inFileKMeans_var.min())/(inFileKMeans_var.max()-inFileKMeans_var.min())
inFileHier_norm=inFileKmeans_norm

#Busqueda de la cantidad óptima de clusters mediante La gráfica codo de jambú
wcss = []
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, max_iter=300)
    kmeans.fit(inFileKMeans_norm)
    wcss.append(kmeans.inertia_)

#Graficar el códo de jambú
plt.plot(range(1, 11), wcss)
plt.title("Códo de jambú")
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.show()

#Aplicar el método k-means
clustering=KMeans(n_clusters=3, max_iter=300, algorithm='elkan')
clustering.fit(inFileKMeans_norm)

#Dunn


#Davis boulding
print("El índice Davies Boulding para KMeans es: ",davies_bouldin_score(inFileKMeans_norm, clustering.labels_))

#Agregar la clasificación al archivo original
inFileKMeans['KMeans_Clusters']=clustering.labels_
print(clustering.labels_)



##Aplicamos el método de Clustering jerárquico
Clustering_Jerarquico=linkage(inFileHier_norm,'ward')

#Graficamos el dendrograma
plt.plot(dendrogram=sch.dendrogram(Clustering_Jerarquico))
plt.title("Clustering jerárquico")
plt.xlabel('')
plt.ylabel('')
plt.show()

#Obtenemos los clusters para todos los datos (comienzan en 1)
clusters= fcluster(Clustering_Jerarquico, t=3.3, criterion='distance')
print(clusters)
#Le restamos 1 al arreglo clusters para que comiencen desde 0
clusters=clusters-1
print(clusters)

#Asignamos guardamos los clusters en el archivo original
inFileHier['Clustering_Jerarquico']=clusters

#Davies Boulding index
print("El índice Davies Boulding es: ",davies_bouldin_score(inFileHier_norm, clusters))

##Visualizamos los clusters
#Se aplica el análisis de componentes principales
#KMeans
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca_inFileKMeans=pca.fit_transform(inFileKMeans_norm)
pca_inFileKMeans_df=pd.DataFrame(data=pca_inFileKMeans, columns=['Componente_1', 'Componente_2'])
pca_nombres_inFileKMeans=pd.concat([pca_inFileKMeans_df, inFileKMeans[['KMeans_Clusters']]],axis=1)

#Clustering Jerarquico
pca_inFileHier=pca.fit_transform(inFileHier_norm)
pca_inFileHier_df=pd.DataFrame(data=pca_inFileHier, columns=['Componente_1', 'Componente_2'])
pca_nombres_inFileHier=pd.concat([pca_inFileHier_df, inFileHier[['Clustering_Jerarquico']]], axis=1)


fig=plt.figure(figsize=(6,6))
ax=fig.add_subplot(1,2,1)
ax.set_xlabel('Componente 1', fontsize=12)
ax.set_ylabel('Componente 2', fontsize=12)
ax.set_title('Agrupamiento KMeans', fontsize=17)

color_theme=np.array(["blue", "green", "red"])
ax.scatter(x=pca_nombres_inFileKMeans.Componente_1, y=pca_nombres_inFileKMeans.Componente_2, c=color_theme[pca_nombres_inFileKMeans.KMeans_Clusters], s=30)

bx=fig.add_subplot(1,2,2)
bx.set_xlabel('Componente 1', fontsize=12)
bx.set_ylabel('Componente 2', fontsize=12)
bx.set_title('Agrupamiento Clusterin Jerárquico', fontsize=17)
color_theme=np.array(["blue", "violet", "grey"])
bx.scatter(x=pca_nombres_inFileHier.Componente_1, y=pca_nombres_inFileHier.Componente_2, c=color_theme[pca_nombres_inFileHier.Clustering_Jerarquico], s=30)
plt.show()


