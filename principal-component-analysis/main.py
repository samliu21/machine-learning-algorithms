import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
data.drop(columns=data.columns.difference(['Length', 'Birthweight']), inplace=True)

"""
Feature normalize
"""
mu = data.mean()
std = data.std()
data = (data - mu) / std
m = len(data)
K = 1

plt.scatter(data['Length'], data['Birthweight'], color='teal')

"""
Derive the covariance matrix
"""
sigma = 1 / m * np.dot(data.T, data)
eigen_values, eigen_vectors = np.linalg.eig(sigma)
sorted_index = np.argsort(eigen_values[:: -1])
sorted_eigen_values = eigen_values[sorted_index]
sorted_eigen_vectors = eigen_vectors[: K, sorted_index]

x_axis = np.arange(-3, 3, 0.1)
plt.plot(x_axis, x_axis * sorted_eigen_vectors[0][1] / sorted_eigen_vectors[0][0])

"""
Transform the data 
"""
X_reduced = np.dot(data, sorted_eigen_vectors.T) # [m * 2] * [2 * 1]

plt.figure()
plt.scatter(X_reduced, np.zeros((m, 1)))

"""
Untransform the data
"""
plt.figure()
X_approx = np.dot(X_reduced, sorted_eigen_vectors)
plt.scatter(X_approx[:, 0], X_approx[:, 1])
plt.show()

variation_lost = np.sum(np.sum((X_approx - data) ** 2))
total_variation = np.sum(np.sum((data - data.mean(axis=0)) ** 2))
print('Variation retained: {}'.format(1 - variation_lost / total_variation))
