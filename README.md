# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

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
![K Means Clustering for Customer Segmentation](sam.png)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
