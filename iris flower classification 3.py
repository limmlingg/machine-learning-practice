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

# print(dataset.head(5))

import matplotlib.pyplot as plt

# plt.figure(4, figsize=(10, 8))

# plt.scatter(data[:50, 0], data[:50, 1], c='r', label='Iris-setosa')

# plt.scatter(data[50:100, 0], data[50:100, 1], c='g',label='Iris-versicolor')

# plt.scatter(data[100:, 0], data[100:, 1], c='b',label='Iris-virginica')

# plt.xlabel('Sepal length',fontsize=20)
# plt.ylabel('Sepal width',fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.title('Sepal length vs. Sepal width',fontsize=20)
# plt.legend(prop={'size': 18})
# # plt.show()

# plt.figure(4, figsize=(8, 8))

# plt.scatter(data[:50, 2], data[:50, 3], c='r', label='Iris-setosa')

# plt.scatter(data[50:100, 2], data[50:100, 3], c='g',label='Iris-versicolor')

# plt.scatter(data[100:, 2], data[100:, 3], c='b',label='Iris-virginica')
# plt.xlabel('Petal length',fontsize=15)
# plt.ylabel('Petal width',fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.title('Petal length vs. Petal width',fontsize=15)
# plt.legend(prop={'size': 20})
# # plt.show()

print(dataset.iloc[:,2:].corr())
print(dataset.iloc[:50,:].corr()) #setosa
print(dataset.iloc[50:100,:].corr()) #versicolor
print(dataset.iloc[100:,:].corr()) #virginica

fig = plt.figure(figsize = (8,8))
ax = fig.gca()
dataset.hist(ax=ax)
plt.show()