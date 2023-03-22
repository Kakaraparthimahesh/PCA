# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:30:41 2022

@author: MAHESH
"""

# REQUIRED LIBRARIES

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# DATA PROCESSING

import pandas as pd

wine_data = pd.read_csv("wine.csv")
wine_data

wine_data.shape
wine_data.info()
wine_data.describe()
list(wine_data)
wine_data.head()

wine_data.isnull().sum()

wine_data['Type'].value_counts()

wine_data1 = wine_data.iloc[:,1:]
wine_data1
list(wine_data1)

# Converting into numpy array

wine_data1_array = wine_data.values
wine_data1_array

# Normalizing the  numerical data

from sklearn.preprocessing import scale
wine_data1_norm = scale(wine_data1)
wine_data1_norm

# Applying PCA fit transform to data set

from sklearn.decomposition import PCA

PCA = PCA()
PCA_values = PCA.fit_transform(wine_data1_norm)
PCA_values

PCA.components_

# The amount of variance that each PCA explains is 

var = PCA.explained_variance_ratio_
var

# Variance plot for PCA components obtained 

import matplotlib.pyplot as plt
plt.plot(var,color = "red");

PCA1 = PCA_values[:,0:1]
PCA2 = PCA_values[:,1:2]
PCA1
PCA2

# Final data frame 

final_df_wine = pd.concat([wine_data['Type'],pd.DataFrame(PCA_values[:,0:3],columns=['PC1','PC2','PC3'])],axis=1)
final_df_wine

# EXPLORATORY DATA ANALYSIS 

import seaborn as sns
fig=plt.figure(figsize=(16,12))
sns.scatterplot(data=final_df_wine);

sns.scatterplot(data=final_df_wine, x='PC1', y='PC2', hue='Type');

# plot between PCA1 and PCA2 

x = PCA_values[:,0:1]
y = PCA_values[:,1:2]

plt.scatter(x,y)

# FOR TOTAL 13 PCA 

pc_win = pd.DataFrame(data = PCA_values , columns = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11','pc12','pc13'])
pc_win.head()
pc_win.shape
type(pc_win)
pc_win.values

# BAR GRAPH

import seaborn as sns
win_1 = pd.DataFrame({'var':PCA.explained_variance_ratio_,'PC':['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11','pc12','pc13']})
sns.barplot(x='PC',y="var", data=win_1, color="PINK");

# FOR FIRST 3 PCA 

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pc = pca.fit_transform(wine_data1_norm)
pc.shape

pc_win1 = pd.DataFrame(data = pc , columns = ['pc1','pc2','pc3'])
pc_win1
pc_win1.head()
pc_win1.shape
type(pc_win1)

pc_win1.values

import seaborn as sns
win_2 = pd.DataFrame({'var':pca.explained_variance_ratio_,'PC':['pc1','pc2','pc3']})
sns.barplot(x='PC',y="var", data=win_2, color="orange");

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(pc_win1.iloc[:, 0], pc_win1.iloc[:, 1], pc_win1.iloc[:, 2])
plt.show()

# HIERACHIAL CLUSTURING

xc_wi = pc_win1.iloc[:,:3]
xc_wi

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

# as we have already normalized the data so creating dendrogram

# construction of Dendogram

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title(" Dendograms")  
dend = sch.dendrogram(sch.linkage(xc_wi, method='complete')) 

# creating clustures

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
Z= cluster.fit_predict(xc_wi)
Z

# scatter plot

plt.figure(figsize=(10, 7))  
plt.scatter(xc_wi.iloc[:,0], xc_wi.iloc[:,1],xc_wi.iloc[:,2], c=cluster.labels_, cmap='rainbow')  

# group using clusters

y = pd.DataFrame(cluster.fit_predict(wine_data1_norm),columns=['clustersid'])
y['clustersid'].value_counts()

wine_data['cluster']=cluster.labels_
wine_data

# KMEANS CLUSTURING

# ELBOW CURVE
# by plotting elbow method we can decide which k is best

from sklearn.cluster import KMeans

wcss = []
for i in range(1,6):
    kmeans = KMeans(n_clusters=i,random_state=30)
    kmeans.fit(wine_data1_norm)
    wcss.append(kmeans.inertia_)
       
from matplotlib import pyplot as plt
plt.plot(range(1,6),wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS');

model = KMeans(n_clusters=3,random_state=30)
model.fit(wine_data1_norm)
model.labels_
kmeans.inertia_

wine_data['cluster']=model.labels_
wine_data
