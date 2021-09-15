import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import time

"""
A neural network created from scratch that takes in 20 x 20 greyscale images of handwritten numbers and predicts the number
Averages an AC rate of ~92%
"""
# np.random.seed(0)

"""
Scrambled and split the data into 3000 training and 2000 testing examples
Inserted column of 1's into train and test data 
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
Increase m to 50 * 3000 to test speed of mini batches
"""
train = np.tile(train, (300, 1))
train_y = np.tile(train_y, (300, 1))

train = np.insert(train, 0, [1] * len(train), axis=1) # [3000 * 401]
test = np.insert(test, 0, [1] * len(test), axis=1) # [2000 * 401]

"""
# of units in input, hidden, and output layers
Initialize theta matrices from -epsilon to epsilon
Declare other useful gradient descent variables
"""
input_layer_size = train.shape[1] 
output_layer_size = 10
hidden_layer_size = 25

epsilon = 0.5
theta_2 = np.random.rand(hidden_layer_size, input_layer_size) * 2 * epsilon - epsilon # [25 * 401]
theta_3 = np.random.rand(output_layer_size, hidden_layer_size + 1) * 2 * epsilon - epsilon # [10 * 26]

iterations = 30
alpha = 1
_lambda = 3
m = len(train)
j_history = []

"""
Initialize a a [200 * 200] grid of [20 * 20] images, each image being [10 * 10]
"""
image_pixels = np.zeros((200, 200))
for i in range(100):
	data = train[i, 1 :].reshape((20, 20))
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
The goal is to compute a vectorized gradient used to update theta in our gradient descent

1. Use forward propogation to determine a_1, a_2, and a_3
2. Compute delta_2 and delta_3
3. Update triangle_2 and triangle_3
4. Verify gradients using gradient checking 
5. Update theta_2 and theta_3
"""
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def compute_j(theta_2, theta_3):
	y_hat = np.zeros(train_y.shape)
	_, y_hat = forward_propogation(train, theta_2, theta_3)
	j = -1 / m * np.sum(train_y * np.log(y_hat) + (1 - train_y) * (np.log(1 - y_hat)))
	j += _lambda / (2 * m) * (np.sum(theta_2[:, 1 :] ** 2) + np.sum(theta_3[:, 1 :] ** 2)) # Regularization term
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
		# print(i)
		for j in range(theta_3.shape[1]):
			diff_matrix[i][j] += epsilon_delta 

			y2 = compute_j(theta_2, theta_3 + diff_matrix)
			y1 = compute_j(theta_2, theta_3 - diff_matrix)
			derivative = (y2 - y1) / (2 * epsilon_delta)

			dif = abs(derivative - delta_3[i][j])
			# print(derivative, delta_3[i][j])
			max_difference = max(max_difference, dif)

			diff_matrix[i][j] -= epsilon_delta

	if max_difference > 10 ** -9:
		print('Failed gradient check with max difference {}.\nQuiting...'.format(max_difference))
		sys.exit()
	print('Passed gradient check with max difference {}.'.format(max_difference))

def forward_propogation(data, theta_2, theta_3):
	sz = len(data)
	a_2 = sigmoid(data.dot(np.transpose(theta_2))) # [sz * 26]
	a_2 = np.insert(a_2, 0, [0] * sz, axis=1)
	a_3 = sigmoid(a_2.dot(np.transpose(theta_3))) # [sz * 10]
	
	return a_2, a_3

start = time.time()
for t in range(iterations):
	if t % 5 == 0:
		print('Starting gradient descent...' if t == 0 else 'Done {} iterations of gradient descent.'.format(t))

	triangle_2 = np.zeros(theta_2.shape) # [25 * 401]
	triangle_3 = np.zeros(theta_3.shape) # [10 * 26]

	a_2, a_3 = forward_propogation(train, theta_2, theta_3)

	delta_3 = a_3 - train_y # [3000 * 10]
	delta_2 = np.dot(delta_3, theta_3) * a_2 * (1 - a_2) # [3000 * 25]
	delta_2 = np.delete(delta_2, 0, axis=1)

	triangle_3 += np.transpose(delta_3).dot(a_2)
	triangle_2 += np.transpose(delta_2).dot(train)

	D_2 = 1 / m * triangle_2
	D_3 = 1 / m * triangle_3 
	D_2[:, 1 :] += _lambda / m * theta_2[:, 1 :]
	D_3[:, 1 :] += _lambda / m * theta_3[:, 1 :]

	# if t == 0:
	# 	gradient_check(D_2, D_3)

	theta_2 -= alpha * D_2
	theta_3 -= alpha * D_3

	j_history.append(compute_j(theta_2, theta_3))

time_took = time.time() - start
print(j_history[-1])
"""
Plot iterations against cost function
"""
x_axis = np.arange(1, iterations + 1, 1)
plt.plot(x_axis, j_history)
# plt.show()

"""
Run neural network against sample data and print AC rate
"""
correct = 0

test_m = len(test)

_, a_3 = forward_propogation(test, theta_2, theta_3)
# print(a_3.max(axis=1))
for i in range(test_m):
	idx = a_3[i].argmax()
	correct += original_y[3000 + i][0] == idx

print('AC rate:', correct / test_m, 'Took', time_took, 'seconds.')