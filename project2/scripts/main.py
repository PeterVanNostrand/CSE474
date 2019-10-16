import numpy as np
import matplotlib.pyplot as plt
import util_mnist_reader

# def plusMinOne(matrix):
#     return normalize(matrix)*2 - 1

def relu(z):
    return np.maximum(z, 0)

# def normalize(matrix):
#     '''
#     scales the values in a matrix to be in the range [0,1]
#     ## Parameters
#         - matrix: the matrix to be normalized
#     ## Returns
#         - out: the normalized matrix
#     '''
#     col_max = matrix.max(axis=0)
#     col_min = matrix.min(axis=0)
#     return (matrix - col_min)/(col_max-col_min)

# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))

def softmax(z):
    """Compute softmax values for each sets of scores in x."""
    ez = np.exp(z - np.max(z, axis=0))
    return ez / np.sum(ez, axis=0)

# def softmax(z): # computes the softmax of set of activations exp(ak) / sum( exp(aj) )
#     exp_z = np.exp(z)
#     sum_exp_z = np.sum(exp_z, axis=0) # assumes neurons for one sample are in column
#     return exp_z / sum_exp_z

def cost(a, y):
    diff = (a-y)
    loss = np.square(diff)
    # loss = np.multiply(y, np.log(a)) # ylog(a)
    return np.sum(loss, axis=0) # determine the loss for each of the 10 output neurons

class layer_input:
    def __init__(self, num_nodes):
        # Layer 0: Input Layer - Activations are pixel values
        self.n = num_nodes # number of neurons in 0th layer

    def set_inputs(self, x_train):
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
        # self.b = np.random.randn(self.n, 1)*0.01
        
    def forward(self):
        # z = wa0 + b
        # a = sigmoid(z)

        # resulting n x ns (each col the output neurons for a sample)
        self.z = np.dot(self.w, self.prev_layer.a) #+ self.b
        # self.z = plusMinOne(self.z.T).T # normalize(z)
        # computes final activation of every neuron in layer 1 same dims as z
        self.a = relu(self.z)

    def backward(self, y, next_layer):
        # compute gradient descient for hidden layer
        # del_a = np.dot((next_layer.a - y).T, next_layer.w).T
        # del_z = np.multiply(np.multiply(del_a, self.a), (1-self.a))
        # self.del_w = np.dot(del_z, self.prev_layer.a.T)

        del_a = np.dot((next_layer.a - y).T, next_layer.w).T # 392 x 60k
        del_z = np.max(del_a.copy(), 0) # 392 x 60k
        # del_z1[h < 0] = 0 # set all neg value to 0
        self.del_w = np.dot(self.prev_layer.a, del_z.T).T

        # da1 = np.dot((next_layer.a - y).T, next_layer.w)
        # daz1 = np.dot(self.a.T, (1-self.a))
        # dz1 = np.dot(da1, daz1).reshape((ns, self.n))
        # dw1 = np.dot(dz1.T, self.prev_layer.a.reshape(self.prev_layer.n, ns).T)
        # self.del_w = dw1

        # da1 = np.dot((a2 - y).T, w2)
        # daz1 = np.dot(a1.T, (1-a1))
        # dz1 = np.dot(da1, daz1).reshape((1,n1))
        # dw1 = np.dot(dz1.T, a0.reshape((n0,1)).T)

    def update(self, lr):
        self.w -= lr * self.del_w


class layer_output: # Layer 2: Output Layer - Softmax
    def __init__(self, num_nodes, prev_layer):
        self.prev_layer = prev_layer
        self.n = num_nodes # 10 output neurons, one per class
        self.w = np.random.randn(self.n, self.prev_layer.n)*0.01
        # self.b = np.random.randn(self.n, 1)*0.01
        
    def forward(self):
        # z = 2a1 + b
        # a = softmax(z)
        self.z = np.dot(self.w, self.prev_layer.a) #+ self.b
        # self.z = normalize(self.z) #normalize(self.z)
        self.a = softmax(self.z)

    def backward(self, y):
        # compute gradient descent for output layer
        del_z = self.a - y
        self.del_w = np.dot(del_z, self.prev_layer.a.T)

    def update(self, lr):
        self.w -= lr * self.del_w


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

    x_train = (x_train.astype("float32") / 255.0).reshape((60000, 784))
    x_test = (x_test.astype("float32") / 255.0).reshape((10000, 784))

    # Create an array for loss calculation with 1 for correct label, 0 otherwise
    y = np.zeros((ns, 10))
    for i in range(0, ns):
        y[i][int(y_train[i])] = 1
    y = y.T
  
    l0 = layer_input(785)
    # l0.set_inputs(x_train)
    l1 = layer_hidden(392, l0)
    l2 = layer_output(10, l1)

    loss = []

    batch_size = 24
    num_epochs = 25
    for j in range (0, num_epochs):
        for i in range(0, int(ns/batch_size)):
            batch_ins = x_train[i*batch_size:(i+1)*batch_size,:]
            batch_labels = y.T[i*batch_size:(i+1)*batch_size,:].T
            l0.set_inputs(batch_ins)

            # forwared propagate
            l1.forward()
            l2.forward()

            # compute cost
            cost_val = np.mean(cost(l2.a, batch_labels))
            print(cost_val)
            loss.append(cost_val)

            # back propagate and update weights
            l2.backward(batch_labels)
            l1.backward(batch_labels, l2)

            l2.update(0.5)
            l1.update(0.5)

            # randomly reorder the training data (rebuilding labels to correspond)
            # data = np.hstack((x_train, y_train.reshape(60000, 1)))
            # np.random.shuffle(data)
            # x_train = data[:,:-1]
            # y_train = data[:,-1:].reshape(60000)
            # for i in range(0, ns): # set desired output to 1 for correct class, 0 for rest
            #     y[int(y_train[i])][i] = 1

    plt.figure()
    plt.plot(loss)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Cost")
    plt.title("Training Accuracy vs Epochs")

    print("Done!")

    plt.show()