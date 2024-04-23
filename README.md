# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import all necessary packages.

2.Upload the appropiate dataset to perform K-Means Clustering.

3.Perform K-Means Clustering on the requried dataset.

4.Plot graph and display the clusters.

## Program:
```python

Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Syed Mokthiyar S.M
RegisterNumber: 212222230156

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

# Load data from CSV
data=pd.read_csv('/content/Mall_Customers.csv')
data

# Extract features
X data[['Annual Income (k$)', 'Spending Score (1-100)']]
X

plt.figure(figsize=(4, 4))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)']) 
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Number of clusters
k = 5
# Initialize KMeans
Kmeans = KMeans(n_clusters=k)
# Fit the dats
Kmeans.fit(X)

```

## Output:
# DATASET:
![Screenshot 2024-04-23 134153](https://github.com/syedmokthiyar/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118787294/480e268c-a6c7-497a-9543-78239c4b8250)

# GRAPH:
![Screenshot 2024-04-23 134428](https://github.com/syedmokthiyar/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118787294/90d839d4-8a4e-4068-bcfc-7d93189c2e57)

# CENTROID VALUE:
![Screenshot 2024-04-23 134516](https://github.com/syedmokthiyar/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118787294/26e7bacc-5b82-43fd-b011-59fe2da03a52)

# K-MEANS CLUSTER:



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
