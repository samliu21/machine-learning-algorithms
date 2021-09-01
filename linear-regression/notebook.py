import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
m = len(train)

"""
We first drop passengerId, name, and ticket, which are evidently irrelevant to the data set
"""
train.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

# print(train.isnull().sum() / m)
"""
We see that age, cabin, and embarked have null values that we need to address
Since a whopping 77% of cabin values are null, it is logical to drop the column entirely
Plotting age, we see that the distribution is right skewed, so we'll fill in null values with the median
Plotting embarked, we see that a vast majority of values are S, so we'll fill in the null values with S
"""
train.drop(columns='Cabin', inplace=True)

# ax = train['Age'].hist(bins=10, color='teal')
# ax.set(xlabel='Age')
# plt.xlim(0, 85)

age = train['Age'].median()
train['Age'].fillna(age, inplace=True)

# ax = train['Embarked'].hist(color='teal')

train['Embarked'].fillna('S', inplace=True)

# print(train.head())
"""
Looking at the remaining data set, the categorical nature of the sex and embarked properties need to be addressed
To address sex, we simply turn female into 1 and male into 0
To address embarked, we create binary columns for each of the three categories
"""
train['Sex'] = (train['Sex'] == 'female').astype(int)
train = pd.get_dummies(train, columns=['Embarked'])

# a = train['Age'][train['Survived'] == 1]
# a.plot(kind='density')
# plt.show()

"""
Now, let's standardize
Also, insert column of 1's for x_0
"""
y = train['Survived']
X = train.drop(columns=['Survived'])
mu_x = X.mean()
std_x = X.std()

X = (X - mu_x) / std_x
X.insert(0, 0, [1] * m)
n = X.shape[1]

y = y.to_frame()

iterations = 1_000
alpha = 0.3
theta = pd.DataFrame(np.zeros((n, 1)))

j_history = []

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def compute_cost(theta):
	h_theta = sigmoid(X.values.dot(theta.values))
	j = -1 / m * np.sum(y * np.log(h_theta) + (1 - y) * np.log(1 - h_theta))
	return j[0]

for _ in range(iterations):
	error = sigmoid(X.values.dot(theta.values)) - y
	theta = theta - alpha / m * (X.T).values.dot(error)
	j_history.append(compute_cost(theta))

# print(theta)
"""
Show values of cost function
"""
# x_axis = np.arange(1, iterations + 1, 1)
# plt.plot(x_axis, j_history)
# plt.show()

test = pd.read_csv('test.csv')
test_count = len(test)
"""
Perform similar cleaning of data set
"""
ids = test['PassengerId']
test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
test['Age'].fillna(age, inplace=True)
test['Embarked'].fillna('S', inplace=True)
test['Sex'] = (test['Sex'] == 'female').astype(int)
test = pd.get_dummies(test, columns=['Embarked'])

test = (test - mu_x) / std_x 
test.insert(0, 0, [1] * test_count)

with open('output.csv', 'w') as out:
	out.write('PassengerId,Survived\n')
	for i in range(test_count):
		x = test.iloc[i]
		val = sigmoid(x.values.dot(theta[0].values))
		out.write('{},{}\n'.format(ids[i], 1 if val > 0.5 else 0))
