from PIL import Image 

import numpy as np
import matplotlib.pyplot as plt 

import random

"""
An image can take up a lot of space. It is my goal in this project to reduce the memory of a picture
More specifically, each pixel can take up a total of 255 * 255 * 255 >> 1e7 numbers. We want to reduce this number to 16
Concretely, we will use the K-means algorithm to map each existing pixel colour to its closest colour centroid
"""

"""
Image produces a n * n * 3 matrix
Reshape to (n * n) * 3 matrix
"""
image = Image.open('bird.png')
a = np.array(image)
pixel_row, pixel_col = a.shape[: 2]
a = a.reshape(-1, a.shape[-1])
# Restricting number of training examples is useful for visualizing the result
# a = a[: 500]

fig = plt.figure()
ax = plt.axes(projection='3d')
x_axis = a[:, 0]
y_axis = a[:, 1]
z_axis = a[:, 2]
# ax.scatter(x_axis, y_axis, z_axis)
# plt.show()

K = 16
m = len(a)
centroids = np.array([a[i] for i in random.sample(range(m), K)])
c = [0 for _ in range(m)]
iterations = 5

def compute_distance(a, b):
	return np.sum((a - b) ** 2)

"""
K-means algorithm
Assign each training example to its closest centroid, then assign each centroid to be the mean of its corresponding points
"""
for _ in range(iterations):
	for a_idx, ex in enumerate(a):
		cur_dis = 10 ** 9
		for centroid_idx, centroid in enumerate(centroids):
			dis = compute_distance(ex, centroid)
			if dis < cur_dis:
				cur_dis = dis 
				c[a_idx] = centroid_idx

	centroids = np.zeros(centroids.shape)
	l = np.zeros((K, 1))
	for idx, ex in enumerate(a):
		centroids[c[idx]] += ex
		l[c[idx]] += 1
	for idx, i in enumerate(centroids):
		i /= l[idx]

centroids = centroids.astype(int)

# Useful for visualizing the result
for i in range(K):
	points = np.array([a[j] for j in range(m) if c[j] == i])
	ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=(random.random(), random.random(), random.random()))
# plt.show()

compressed_image = Image.new('RGB', (pixel_row, pixel_col))
pix = compressed_image.load()
for i in range(pixel_row):
	for j in range(pixel_col):
		el = i * pixel_row + j
		pix[i, j] = tuple(centroids[c[el]])

compressed_image.show()
image.show()