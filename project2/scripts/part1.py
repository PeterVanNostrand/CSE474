import numpy as np
import matplotlib.pyplot as plt
import util_mnist_reader

def normalize(matrix):
    '''
    scales the values in a matrix to be in the range [0,1]
    ## Parameters
        - matrix: the matrix to be normalized
    ## Returns
        - out: the normalized matrix
    '''
    col_max = matrix.max(axis=0)
    col_min = matrix.min(axis=0)
    return (matrix - col_min)/(col_max-col_min)

def sigmoid(z):
    z = normalize(z)
    return 1 / (1 + np.exp(-z))

def softmax(z): # computes the softmax of set of activations exp(ak) / sum( exp(aj) )
    z = normalize(z)
    exp_z = np.exp(z)
    sum_exp_z = np.sum(exp_z, axis=0) # assumes neurons for one sample are in column
    return exp_z / sum_exp_z

def cost(a, y):
    loss = np.multiply(y, np.log(a)) # ylog(a)
    return -np.mean(loss, axis=1) # determine the loss for each of the 10 output neurons

class layer_input:
    def __init__(self, x_train):
        # Layer 0: Input Layer - Activations are pixel values
        self.n = x_train.shape[1] + 1 # number of neurons in 0th layer
        ones_arr = np.ones((x_train.shape[0],1))
        self.a = np.hstack((x_train, ones_arr)).T


class layer_hidden:  # Layer 1: Hidden Layer - Fully connected
    def __init__(self, num_nodes, prev_layer):
        # store reference to previous layer
        self.prev_layer = prev_layer
        # Number of neurons in 1st layer
        self.n = num_nodes
        # create a 392 row x 784 col random matrix of weights
        self.w = np.random.randn(self.n, self.prev_layer.n)*0.01
        # create a n row x 1 col random matrix of biases (1 bias/neuron)
        self.b = np.random.randn(self.n, 1)*0.01
        
    def forward(self):
        # z = wa0 + b
        # a = sigmoid(z)

        # resulting n x ns (each col the output neurons for a sample)
        self.z = np.dot(self.w, self.prev_layer.a) + self.b
        # computes final activation of every neuron in layer 1 same dims as z
        self.a = sigmoid(self.z)

    def backward(self, y, next_layer, lr):
        # compute gradient descient for hidden layer
        del_a = np.dot((next_layer.a - y).T, next_layer.w).T
        del_z = np.multiply(del_a, np.multiply(self.a, (1-self.a)))
        del_w = np.dot(del_z, self.prev_layer.a.T)
        self.w += lr * del_w


class layer_output: # Layer 2: Output Layer - Softmax
    def __init__(self, num_nodes, prev_layer):
        self.prev_layer = prev_layer
        self.n = num_nodes # 10 output neurons, one per class
        self.w = np.random.randn(self.n, self.prev_layer.n)*0.01
        self.b = np.random.randn(self.n, 1)*0.01
        
    def forward(self):
        # z = 2a1 + b
        # a = softmax(z)
        self.z = np.dot(self.w, self.prev_layer.a) + self.b
        self.a = softmax(self.z)

    def backward(self, y, lr):
        # compute gradient descent for output layer
        del_z = self.a - y
        del_w = np.dot(del_z, self.prev_layer.a.T)
        self.w += lr * del_w


if __name__ == '__main__':
    print("Part 1: Neural Network")
    fashion_path="C:\\Users\peter\Documents\Peter\\7-College\\7-Fall-2019\CSE474\Projects\project2\data\\fashion"

    # y-train is 60,000 col x 1 row matrix of uint8 labels
    # y-test  is 10,000 col x 1 row matrix of uint8 labels
    x_train, y_train = util_mnist_reader.load_mnist(fashion_path , kind='train')
    # x-train is 784 col x 60,000 row matrix of pixels, one row per sample
    # x-train is 784 col x 10,000 row matrix of pixels, one row per sample
    # pixel values are 0-255
    x_test, y_test = util_mnist_reader.load_mnist(fashion_path, kind='t10k')
    
    ns = x_train.shape[0] # number of samples in input

    # Create an array for loss calculation with 1 for correct label, 0 otherwise
    y = np.zeros((10, ns)) # each column represents the output neurons for one sample
    for i in range(0, ns): # set desired output to 1 for correct class, 0 for rest
        y[y_train[i]-1][i] = 1
  
    l0 = layer_input(x_train)
    l1 = layer_hidden(392, l0)
    l2 = layer_output(10, l1)

    loss = []

    for i in range(0, 15):
        # forwared propagate
        l1.forward()
        l2.forward()

        # compute cost
        cost_val = np.sum(cost(l2.a, y))
        loss.append(cost_val)

        # back propagate and update weights
        l2.backward(y, 0.03)
        l1.backward(y, l2, 0.03)

    plt.figure()
    plt.plot(loss/max(loss))
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training Accuracy")
    plt.title("Training Accuracy vs Epochs")

    print("Done!")

    plt.show()
