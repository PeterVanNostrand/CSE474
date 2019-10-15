import numpy as np
import matplotlib.pyplot as plt
import util_mnist_reader

def plusMinOne(matrix):
    return normalize(matrix)*2 - 1

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
    return 1 / (1 + np.exp(-z))

def softmax(z): # computes the softmax of set of activations exp(ak) / sum( exp(aj) )
    exp_z = np.exp(z)
    sum_exp_z = np.sum(exp_z, axis=0) # assumes neurons for one sample are in column
    return exp_z / sum_exp_z

def cost(a, y):
    loss = np.square((a - y))
    # loss = np.multiply(y, np.log(a)) # ylog(a)
    return np.mean(loss, axis=0) # determine the loss for each of the 10 output neurons

class layer_input:
    def __init__(self, num_nodes):
        # Layer 0: Input Layer - Activations are pixel values
        self.n = num_nodes # number of neurons in 0th layer

    def set_inputs(self, x_train):
        self.a = np.transpose(x_train) # Every column is one sample, every row is a pixel within that sample
        self.a = normalize(self.a) # normalizing inputs from [0-255] to [0-1] for simplicity
        # append extra feature of 1 to every sample for biases
        self.a = np.vstack((self.a, np.ones((1,self.a.shape[1]))))


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
        self.z = plusMinOne(self.z) # normalize(z)
        # computes final activation of every neuron in layer 1 same dims as z
        self.a = sigmoid(self.z)

    def backward(self, y, next_layer):
        # compute gradient descient for hidden layer
        del_a = np.dot((next_layer.a - y).T, next_layer.w).T
        del_z = np.multiply(del_a, np.multiply(self.a, (1-self.a)))
        self.del_w = np.dot(del_z, self.prev_layer.a.T)

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
        self.z = plusMinOne(self.z) #normalize(self.z)
        self.a = softmax(self.z)

    def backward(self, y):
        # compute gradient descent for output layer
        del_z = self.a - y
        self.del_w = np.dot(del_z, self.prev_layer.a.T)

    def update(self, lr):
        self.w -= lr * self.del_w

def scale(z):
    return 2.*(z - np.min(z))/np.ptp(z)-1

def scale01(z):
    return (z - np.min(z))/np.ptp(z)

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

    x_train = x_train / 255

    # randomly reorder the training data (rebuilding labels to correspond)
    # data = np.hstack((x_train, y_train.reshape(60000, 1)))
    # np.random.shuffle(data)
    # x_train = data[:,:-1]
    # y_train = data[:,-1:].reshape(60000)

    ys = np.zeros((ns, 10)) # each column represents the output neurons for one sample

    for i in range(0, ns): # set desired output to 1 for correct class, 0 for rest
        ys[i][int(y_train[i])] = 1

    n0 = 785
    n1 = 150
    n2 = 10
    
    w1 = np.random.randn(n1, n0)*0.01
    w2 = np.random.randn(n2, n1)*0.01
    lr = 0.01
    loss = []

    for i in range (0, ns):
        a0 = np.append(x_train[i], 1)#.reshape(785,1)
        y = ys[i]#.reshape((10,1))
        
        z1 = np.dot(w1, a0)#.reshape((392,1))
        # z1 = scale(z1)
        a1 = sigmoid(z1)#.reshape((392,1))

        z2 = np.dot(w2, a1)#.reshape((10,1))
        # z2 = scale(z2)
        a2 = softmax(z2)#.reshape((10,1))

        loss.append(np.sum(np.square(a2 - y)))

        dz2 = (a2 - y).reshape((n2,1))
        # a1 = a1.reshape((392,1))
        dw2 = np.dot(dz2, a1.reshape(n1,1).T)#.reshape((10,392))

        da1 = np.dot((a2 - y).T, w2)
        daz1 = np.dot(a1.T, (1-a1))
        dz1 = np.dot(da1, daz1).reshape((1,n1))
        dw1 = np.dot(dz1.T, a0.reshape((n0,1)).T)

        w1 -= lr * dw1
        w2 -= lr * dw2

        print(i," ", loss[i])


    plt.figure()
    plt.plot(loss)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Cost")
    plt.title("Training Accuracy vs Epochs")



    print("Done!")

    plt.show()

    # a_max = np.max(a0, axis=1)
    # a_min = np.min(a0, axis=1)
    # x_max = np.max(x_train, axis=1)
    # x_min = np.min(x_train, axis=1)
    # print("z1[0]\n",z1[:,0])
    # print("z1[1]\n",z1[:,1])