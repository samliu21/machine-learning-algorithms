import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

"""
Scrambled the data rows
Replaced 10 with 0
"""
X = pd.read_csv('X.csv', header=None)
y = pd.read_csv('Y.csv', header=None)
idx = np.random.permutation(X.index)
X = X.reindex(idx)
y = y.reindex(idx)
y.replace(10, 0, inplace=True)

train = X[0 : 3000].values
test = X[3000 : 5000].values
original_y = y.values
train_y = (y[0 : 3000].values == np.arange(0, 10, 1)).astype(int)
test_y = (y[3000 : 5000].values == np.arange(0, 10, 1)).astype(int)

"""
Initialize theta, # of iterations in gradient descent, learning rate
"""
input_layer_size = train.shape[1]
output_layer_size = 10
hidden_layer_size = 25

epsilon = 0.5
theta_2 = np.random.rand(hidden_layer_size, input_layer_size + 1) * 2 * epsilon - epsilon # [25 * 401]
theta_3 = np.random.rand(output_layer_size, hidden_layer_size + 1) * 2 * epsilon - epsilon # [10 * 26]

iterations = 50
alpha = 0.5
_lambda = 0.01
m = len(train)
j_history = []

"""
Initialize a grid of 100 sample images—that is, a [10 * 10] grid of [20 * 20] images
"""
image_pixels = np.zeros((200, 200))
for i in range(100):
	data = train[i].reshape((20, 20))
	x = i % 10
	y = i // 10
	start_x = 20 * x
	end_x = start_x + 20
	start_y = 20 * y
	end_y = start_y + 20

	image_pixels[start_x : end_x, start_y : end_y] = data
img = Image.fromarray(image_pixels * 200)
# img.show()

"""
The goal is to compute a gradient used to update all thetas in our gradient descent

for i = 1 to m
	1. Use forward propogation to determine a_1, a_2, and a_3
	2. Compute delta_2 and delta_3
	3. Update triangle_2 and triangle_3
	4. Verify gradients using the gradient checking algorithm when i = 1
Update theta_2 and theta_3
"""
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def add_bias(x):
	return np.insert(x, 0, [1], axis=0)

def compute_j(theta_2, theta_3):
	y_hat = np.zeros(train_y.shape)
	for i in range(m):
		_, _, a_3 = forward_propogation(train[None, i], theta_2, theta_3)
		y_hat[i] = np.transpose(a_3)
	j = -1 / m * np.sum(train_y * np.log(y_hat) + (1 - train_y) * (np.log(1 - y_hat)))
	return j

def gradient_check(delta_2, delta_3):
	epsilon_delta = 0.0001
	max_difference = 0

	# diff_matrix = np.zeros(theta_2.shape)
	# for i in range(theta_2.shape[0]):
	# 	for j in range(theta_2.shape[1]):
	# 		diff_matrix[i][j] += epsilon_delta 

	# 		y2 = compute_j(theta_2 + diff_matrix, theta_3)
	# 		y1 = compute_j(theta_2 - diff_matrix, theta_3)
	# 		derivative = (y2 - y1) / (2 * epsilon_delta)

	# 		dif = abs(derivative - delta_2[i][j])
	# 		max_difference = max(max_difference, dif)

	# 		diff_matrix[i][j] -= epsilon_delta

	diff_matrix = np.zeros(theta_3.shape)
	for i in range(theta_3.shape[0]):
		print(i)
		for j in range(theta_3.shape[1]):
			diff_matrix[i][j] += epsilon_delta 

			y2 = compute_j(theta_2, theta_3 + diff_matrix)
			y1 = compute_j(theta_2, theta_3 - diff_matrix)
			derivative = (y2 - y1) / (2 * epsilon_delta)

			dif = abs(derivative - delta_3[i][j])
			max_difference = max(max_difference, dif)

			diff_matrix[i][j] -= epsilon_delta

	print('{:.12f}'.format(max_difference))

def forward_propogation(initial, theta_2, theta_3):
	a_1 = np.transpose(initial) # [401 * 1]
	a_1 = add_bias(a_1)
	a_2 = sigmoid(theta_2.dot(a_1)) # [26 * 1]
	a_2 = add_bias(a_2)
	a_3 = sigmoid(theta_3.dot(a_2)) # [10 * 1]
	
	return a_1, a_2, a_3

for t in range(iterations):
	if t % 10 == 0:
		print('Iteration:', t)
	triangle_2 = np.zeros(theta_2.shape) # [25 * 401]
	triangle_3 = np.zeros(theta_3.shape) # [10 * 26]
	for i in range(m):
		a_1, a_2, a_3 = forward_propogation(train[None, i], theta_2, theta_3)

		delta_3 = a_3 - np.transpose(train_y[None, i]) # [10 * 1]

		delta_2 = np.dot(np.transpose(theta_3), delta_3) * a_2 * (1 - a_2) # [26, 1]

		triangle_3 += (np.dot(delta_3, np.transpose(a_2)))
		triangle_2 += (np.dot(delta_2[1 :], np.transpose(a_1)))

	D_2 = 1 / m * triangle_2
	D_3 = 1 / m * triangle_3 

	theta_2 -= alpha * D_2
	theta_3 -= alpha * D_3
	j_history.append(compute_j(theta_2, theta_3))

x_axis = np.arange(1, iterations + 1, 1)
plt.plot(x_axis, j_history)
plt.show()
print(j_history)

correct = 0

# print(theta_3)

test_m = len(test)
for i in range(m):
	_, _, a_3 = forward_propogation(train[None, i], theta_2, theta_3)
	idx = a_3.argmax()
	correct += original_y[i][0] == idx

print(correct / m)