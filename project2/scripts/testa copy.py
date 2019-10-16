# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import util_mnist_reader

print("Part 1: Neural Network")
fashion_path="C:\\Users\peter\Documents\Peter\\7-College\\7-Fall-2019\CSE474\Projects\project2\data\\fashion"

n0 = 785
n1 = 392
n2 = 10

x_train, y_train = util_mnist_reader.load_mnist(fashion_path , kind='train')
x_test, y_test = util_mnist_reader.load_mnist(fashion_path, kind='t10k')

ns = x_train.shape[0] # number of samples in input

x_train = (x_train.astype("float32") / 255.0).reshape((60000, 784))
x_test = (x_test.astype("float32") / 255.0).reshape((10000, 784))

# Create an array for loss calculation with 1 for correct label, 0 otherwise
# y = np.zeros((ns, 10))
# for i in range(0, ns):
#     y[i][int(y_train[i])] = 1
# y = y.T

ones_arr = np.ones((x_train.shape[0],1))
a0 = np.hstack((x_train, ones_arr)).T

# N is batch size(sample size); D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 60000, 784, 392, 10

# Create random input and output data
x = x_train.reshape(60000, 784) #np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = y_train.reshape(60000, 1) #np.array([[0], [1], [1], [0]])

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)  

learning_rate = 0.002
loss_col = []
for t in range(200):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)  # using ReLU as activate function
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum() # loss function
    loss_col.append(loss)
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y) # the last layer's error
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T) # the second laye's error 
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0  # the derivate of ReLU
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

plt.plot(loss_col)
plt.show()