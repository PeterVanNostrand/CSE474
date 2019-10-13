import numpy as np
import util_mnist_reader

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z): # computes the softmax of set of activations exp(ak) / sum( exp(aj) )
    exp_z = np.exp(z)
    sum_exp_z = np.sum(exp_z, axis=0) # assumes neurons for one sample are in column
    return exp_z / sum_exp_z

def loss(a, y):
    cost = np.multiply(y, np.log(a)) # ylog(a)
    return -np.mean(cost, axis=1) # determine the loss for each of the 10 output neurons

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

    # Layer 0: Input Layer - Activations are pixel values
    n0 = x_train.shape[1] # number of neurons in 0th layer
    a0 = np.transpose(x_train) # Every column is one sample, every row is a pixel within that sample
    a0 = a0 / 255 # normalizingg inputs from [0-255] to [0-1] for simplicity
    
    # Layer 1: Hidden Layer - Fully connected
    # z1 = w1a0 + b1
    # a1 = sigmoid(z1)
    n1 = 392 # Number of neurons in 1st layer
    w1 = np.random.randn(n1, n0)*0.01 # create a 392 row x 784 col random matrix of weights
    b1 = np.random.randn(n1, 1)*0.01 # create a n1 row x 1 col random matrix of biases (1 bias per neuron)
    z1 = np.dot(w1, a0) + b1 # resulting n1 x ns (each col is the output neurons for a different sample)
    a1 = sigmoid(z1) # computes final activation of every neuron in layer 1 same dims as z1    

    # Layer 2: Output Layer - Softmax
    # z2 = w2a1 + b2
    # a2 = softmax(z2)
    n2 = 10 # 10 output neurons, one per class
    w2 = np.random.randn(n2, n1)*0.01
    b2 = np.random.randn(n2, 1)*0.01
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    
    # compute gradient descent for output layer
    del_z2 = a2 - y
    del_w2 = np.dot(del_z2, a1.T)

    # compute gradient descient for hidden layer
    del_a1 = np.dot((a2 - y).T, w2).T
    del_z1 = np.multiply(del_a1, np.multiply(a1, (1-a1)))
    del_w1 = np.dot(del_z1, a0.T)

    # update weights
    w1 += del_w1
    w2 += del_w2


    print("a2\n",a2[:,0])
    print("y\n",y[:,0])

    print("Done!")

    # a_max = np.max(a0, axis=1)
    # a_min = np.min(a0, axis=1)
    # x_max = np.max(x_train, axis=1)
    # x_min = np.min(x_train, axis=1)
    # print("z1[0]\n",z1[:,0])
    # print("z1[1]\n",z1[:,1])