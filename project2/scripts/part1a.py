import numpy as np
import matplotlib.pyplot as plt
import util_mnist_reader
import keras as k

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

    data = np.hstack((x_train, y_train.reshape(60000, 1)))
    np.random.shuffle(data)
    x_train = data[:,:-1]
    y_train = data[:,-1:].reshape(60000)

    x_train = x_train / 255
    x_test = x_test / 255

    # Create a 3 layer model using Keras
    model = k.Sequential()
    model.add(k.layers.Dense(392, activation='relu', input_dim=784)) # Layer 1 is dense relu layer
    model.add(k.layers.Dense(10, activation='softmax')) # Layer 2 is a softmax output

    # Comple the model using SGD
    sgd = k.optimizers.SGD(learning_rate=0.05, momentum=0.0, nesterov=False)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Convert labels to one hot
    y_trainOH = k.utils.to_categorical(y_train, num_classes=10)
    y_testOH = k.utils.to_categorical(y_test, num_classes=10)

    # Train the model
    model.fit(x_train, y_trainOH, epochs=1, batch_size=1)

    # Evaluate the model
    score = model.evaluate(x_test, y_testOH, batch_size=100)
    print(score)