import numpy as np
import matplotlib.pyplot as plt
import util_mnist_reader
import keras as k
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt

if __name__ == '__main__':
    print("Part 1: Neural Network")
    fashion_path="C:\\Users\peter\Documents\Peter\\7-College\\7-Fall-2019\CSE474\Projects\project2\data\\fashion"

    num_epochs = 20
    labels = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

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
    model.add(k.layers.Dense(196, activation='relu')) # Layer 1 is dense relu layer
    model.add(k.layers.Dense(98, activation='relu')) # Layer 1 is dense relu layer
    model.add(k.layers.Dense(10, activation='softmax')) # Layer 2 is a softmax output

    # Comple the model using SGD
    sgd = k.optimizers.SGD(lr=0.05, momentum=0.0, nesterov=False)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Convert labels to one hot
    y_train = k.utils.to_categorical(y_train, num_classes=10)
    y_test = k.utils.to_categorical(y_test, num_classes=10)

    # Train the model
    # progress = model.fit(x_train, y_train, epochs=num_epochs, batch_size=100)
    progress = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=24, epochs=num_epochs)
    p = model.predict(x_test)

    # Evaluate the model
    score = model.evaluate(x_test, y_test, batch_size=100)
    print(score)

    report = classification_report(y_test.argmax(axis=1), p.argmax(axis=1), target_names=labels)
    print(report)

    train_loss = progress.history["loss"]
    train_acc = progress.history["acc"]
    val_loss = progress.history["val_loss"]
    val_acc = progress.history["val_acc"]

    time = np.arange(num_epochs)

    plt.figure()
    plt.plot(time, train_loss, label="training loss")
    plt.plot(time, train_acc, label="training accuracy")
    plt.plot(time, val_loss, label="validation loss")
    plt.plot(time, val_acc, label="validation accuracy")
    plt.title("Accuracy vs Epochs")
    plt.xlabel("epochs")
    plt.ylabel("accuracy/loss")
    plt.legend()
    plt.show()