# https://www.datacamp.com/community/tutorials/introduction-machine-learning-python

from sklearn.datasets import load_iris
data = load_iris().data
# print(data.shape)

labels = load_iris().target
# print(labels.shape)

import numpy as np
labels = np.reshape(labels,(150,1))
data = np.concatenate([data,labels],axis=-1)

# print(data.shape)

import pandas as pd
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
dataset = pd.DataFrame(data,columns=names)

dataset['species'].replace(0, 'Iris-setosa',inplace=True)
dataset['species'].replace(1, 'Iris-versicolor',inplace=True)
dataset['species'].replace(2, 'Iris-virginica',inplace=True)

print(dataset.head(5))