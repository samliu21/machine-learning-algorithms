# predict-handwritten-number-neural-network

## Overview

A trained neural network that can predict handwritten numbers, all from scratch! 🌱

The neural network receives $400$ pixels vectors, representing greyscale $20 \times 20$ images containing handwritten numbers. It then uses stochaistic gradient descent, backward propogation, and regularization to train the neural network. I used `numpy`, `pandas`, `matplotlib`, and `PIL` dependencies.

## Gradient Descent Algorithm

In this explanation, I'm going to assume a fundamental understanding of neural networks, gradient descent, linear algebra, and matrix calculus.

### Notation

$m =$ # of training examples,  
$L =$ # of layers in the neural network,  
$K =$ # of output categories in the regression (e.g. 10 for our example),  
$s_l =$ # of input neurons in layer $l, 1 \le l \le L$,  
$a_i^{(j)}=$ value of unit $i$ in layer $j$, $1 \le i \le s_j$ and $1 \le j \le L$,  
$h_\theta(x^{(j)})_i = a^{(L)}_i$, for the $j^\text{th}$ training example,    
$\theta_{ij}^{(l)} =$ weight that maps neuron $j$ in layer $l - 1$ to neuron $i$ in layer $l$,  
$y_i^{(j)} =$ correct output on the $j^\text{th}$ training example, for the $i^\text{th}$ digit,  
$\lambda = $ regularization constant.

The goal of any gradient descent algorithm is to minimize our cost function, $J(\theta)$, which can be understood (although somewhat inaccurately for our specific cost function) as the sum squared error between the output of the neural network and the correct answer. More concretely, we will use the following cost function: $$J(\theta) = -\frac 1 m[\sum_{i=1}^m\sum_{k=1}^Ky_k^{(i)}log(h_\theta(x^{(i)}))_k + (1 - y_k^{(i)})log(1 - h_\theta(x^{(i)}))_k] + \frac \lambda {2m}\sum_{l=1}^{L - 1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_l + 1}(\theta_{ji}^{(l)})^2$$

### Forward propogation
We use a <a href="https://towardsdatascience.com/forward-propagation-in-neural-networks-simplified-math-and-code-version-bbcfef6f9250">forward propogation algorithm</a> to compute the units of the neural network. Concretely, for a specific training example, $a^{(i)} = \sigma(\theta^{(i)}a^{(i - 1)})$, where $\sigma$ is the sigmoid function and $a^{(i)}$ is an $s_i$ dimensional vector. 

For convenience, we will define $z^{(i)} = \theta^{(i)}a^{(i - 1)}$.

### Backward propogation
The goal of backward propogation is to determine a gradient, $\triangle$, for $\theta_{ij}^{(l)}$ by which we update $\theta$ and strengthen our neural network. 

Let us define as $\triangle_{ij}^{(l)}$ be the gradient for $\theta_{ij}^{(l)}$. More formally, $\triangle_{ij}^{(l)} = \frac \partial {\partial \theta_{ij}^{(l)}} J$. The proof for the algorithm is quite extensive and will not be covered here, but can be found <a href="https://stats.stackexchange.com/questions/94387/how-to-derive-errors-in-neural-network-with-the-backpropagation-algorithm">here</a> if you're interested. The steps to the algorithm can be found in the code itself.

### Updating $\theta$
Let us define an intermediary term, $D$, such that
$$D_{ij}^{(l)} = \begin{cases}
\frac 1 m \triangle_{ij}^{(l)} + \frac \lambda m \theta_{ij}^{(l)} & \text{if } j \ne 0 \\
\frac 1 m \triangle_{ij}^{(l)} & \text{if } j = 0
\end{cases} $$

Note that we divide by $m$ to take the average gradient among all test examples and add the $\lambda$ term as part of the regularization technique to prevent overfitting.

Since $D$ is merely a regularized, averaged gradient for $\theta$, we start with the general gradient update formula and substitute in $D$:
$$
\theta_{ij}^{(l)} := _{ij}^{(l)} - \alpha \frac \partial {\partial \theta_{ij}^{(l)}} J \\
\theta_{ij}^{(l)} := _{ij}^{(l)} - \alpha D_{ij}^{(l)}
$$

We perform this gradient update over a sufficiently large number of iterations (e.g. 1000) to train our neural network.
