from matplotlib import pyplot as plt
import sklearn
import keras as k
import numpy as np
from sklearn.metrics import classification_report

def create_model(shape, chanDim, classes):
    model = k.models.Sequential()

    # Convolutional layer
    model.add(
        k.layers.Conv2D(
            32,(3,3),
            padding = "same",
            activation='relu',
            input_shape = shape
        )
    )

    # Convolutional layer
    model.add(
        k.layers.Conv2D(
            32,(3,3),
            padding = "same",
            activation='relu'
        )
    )

    # Max pool layer
    model.add(
        k.layers.MaxPooling2D(
            pool_size=(2, 2)
        )
    )
	
    # Dropout layer
    model.add(k.layers.Dropout(0.25))

    # Flatten layers to move into FC
    model.add(k.layers.Flatten())

    # Fully Connected
    model.add(
        k.layers.Dense(
            392,
            activation='relu'
        )
    )

    # Softmax
    model.add(
        k.layers.Dense(
            10,
            activation='softmax'
        )
    )

    return model

if __name__ == '__main__':
    num_epochs = 15
    lr_start = 0.01
    batch_n = 24

    ((x_train, y_train), (x_test, y_test)) = k.datasets.fashion_mnist.load_data()

    ns_train = x_train.shape[0]
    ns_test = x_test.shape[0]

    x_train = x_train.reshape((ns_train, 28, 28, 1))
    x_test = x_test.reshape((ns_test, 28, 28, 1))
    
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = k.utils.to_categorical(y_train, num_classes=10)
    y_test = k.utils.to_categorical(y_test, num_classes=10)

    labels = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

    opt_sgd = k.optimizers.SGD(lr=lr_start, momentum=0.9, decay=lr_start / num_epochs)
    model = create_model((28, 28, 1), -1, 10)
    # model = MiniVGGNet.build(width=28, height=28, depth=1, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt_sgd, metrics=["accuracy"])

    progress = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_n,
        epochs=num_epochs
    )
    p = model.predict(x_test)

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
