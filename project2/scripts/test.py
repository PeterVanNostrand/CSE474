# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import util_mnist_reader

def relu(z):
    return np.maximum(z, 0)

if __name__ == '__main__':
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
    y = np.zeros((ns, 10))
    for i in range(0, ns):
        y[i][int(y_train[i])] = 1
    y = y.T

    ones_arr = np.ones((x_train.shape[0],1))
    a0 = np.hstack((x_train, ones_arr)).T

    w1 = np.random.randn(n1, n0)*0.01
    w2 = np.random.randn(n2, n1)*0.01

    lr = 0.002
    loss_track = []
    for t in range(200):
        # Forward pass: compute predicted y
        z1 = np.dot(w1, a0)
        a1 = relu(z1)

        z2 = np.dot(w2, a1)
        a2 = z2

        # Compute and print loss
        loss = np.square(a2 - y).sum(axis=0).mean() # loss function
        loss_track.append(loss)
        print(t, loss)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_a2 = 2.0 * (a2 - y)
        grad_w2 = np.dot(a1, grad_a2.T).T
        grad_a1 = np.dot(grad_a2.T, w2).T
        grad_z1 = grad_a1.copy()
        grad_z1[z1 < 0] = 0
        grad_w1 = np.dot(a0, grad_z1.T).T

        # grad_y_pred = 2.0 * (y_pred - y) # the last layer's error
        # grad_w2 = h_relu.T.dot(grad_y_pred)
        # grad_h_relu = grad_y_pred.dot(w2.T) # the second laye's error 
        # grad_h = grad_h_relu.copy()
        # grad_h[h < 0] = 0  # the derivate of ReLU
        # grad_w1 = x.T.dot(grad_h)

        # Update weights
        w1 -= lr * grad_w1
        w2 -= lr * grad_w2

    plt.figure()
    plt.plot(loss_track)
    plt.show()