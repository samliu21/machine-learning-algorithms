import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

m = len(train)

"""
The most intuitive first step is to handle the categorical variables Ex, Gd, TA, Fa, Po
We will map these out from 10 -> 2
"""
states = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
states_to_nums = [10, 8, 6, 4, 2]
train.replace(states, states_to_nums, inplace=True)
test.replace(states, states_to_nums, inplace=True)

"""
We start by deleting data that has more than 15% null values
"""
missing_data = (train.isnull().sum()[train.isnull().sum() > 0].sort_values(ascending=False) / m).to_frame()
missing_data = missing_data.rename(columns={ 0: 'Percentage' })
train.drop((missing_data[missing_data.Percentage > 0.15]).index, inplace=True, axis=1)
test.drop((missing_data[missing_data.Percentage > 0.15]).index, inplace=True, axis=1)

"""
Let's take a look at columns like basement
We have BsmtExposure, BsmtQual, BsmtCond, etc. that are all pretty much described by a single BsmtSF
The same goes for garage
As such, we want to remove these columns so as not to place extra emphasis on them
"""
extras = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond', 'BsmtFinType2']
train.drop(columns=extras, inplace=True)
test.drop(columns=extras, inplace=True)

"""
MasVnrArea
Skewed, so we will use the median
"""
# train['MasNvrArea'].hist(color='teal', alpha=0.6)
# plt.show()

train.fillna({ x: train[x].mode()[0] for x in ['MasVnrArea'] }, inplace=True)
test.fillna({ x: train[x].mode()[0] for x in ['MasVnrArea'] }, inplace=True)
# print(train.isnull().sum().sum())

train.drop(columns=['Id'], inplace=True)
ids = test['Id']
test.drop(columns=['Id'], inplace=True)

train = pd.get_dummies(train)
test = pd.get_dummies(test)

"""
Remove outliers 
"""
max_threshold = train['SalePrice'].mean() + 3 * train['SalePrice'].std()
min_threshold = train['SalePrice'].mean() - 3 * train['SalePrice'].std() 

train = train[(train.SalePrice < max_threshold) & (train.SalePrice > min_threshold)]
m = len(train)

y = train['SalePrice'].to_frame()
mu_y = y.mean()
std_y = y.std()
X = train.drop(columns=['SalePrice'])
mu_x = X.mean()
std_x = X.std()
y = (y - mu_y) / std_y
X = (X - mu_x) / std_x 
X.insert(0, 0, [1] * m)
n = X.shape[1]

# print(X['MiscVal'].sort_values(ascending=False))
# X['MiscVal'].hist()
# plt.show()

"""
Some categories appear in the testing set, but not the training set
"""
missing_cols = set(X.columns) - set(test.columns)
for c in missing_cols:
	if c != 0:
		test[c] = 0

alpha = 0.1
iterations = 300

theta = pd.DataFrame(np.zeros((n, 1)))
j_history = []

def cost_function(theta):
	error = X.values.dot(theta.values) - y 
	j = 1 / (2 * m) * error.T.values.dot(error.values)
	return j[0][0]

for _ in range(iterations):
	error = X.values.dot(theta.values) - y 
	theta = theta - alpha / m * X.T.values.dot(error.values)
	j_history.append(cost_function(theta))

# print(theta)

x = np.arange(1, iterations + 1, 1)
plt.plot(x, j_history)
# plt.show()

normal_theta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
normal_theta = pd.DataFrame(theta)

x_axis = []
y_axis = []
for i in range(m):
	actual = y.iloc[i][0]
	actual = (actual * std_y + mu_y)[0]
	expected = X.iloc[i].values.dot(theta.values[:, 0])
	expected = (expected * std_y + mu_y)[0]
	
	x_axis.append(expected)
	y_axis.append(actual)
plt.scatter(x_axis, y_axis)
plt.plot([0, 400_000],[0, 400_000],c='red')
# plt.show()

root_mean_squared_error = math.sqrt(sum((pd.Series(x_axis) - pd.Series(y_axis)) ** 2) / m)
print('Root mean squared error: ', root_mean_squared_error)

# print(test.isnull().sum().sort_values(ascending=False))

test = (test - mu_x) / std_x
test.insert(0, 0, [1] * len(ids))

theta_vector = theta.values[:, 0]

with open('output.csv', 'w') as out:
	out.write('Id,SalePrice\n')
	for i in range(len(ids)):
		val = test.iloc[i].values.dot(theta_vector)
		val = (val * std_y + mu_y)[0]
		out.write('{},{}\n'.format(ids[i], val))
