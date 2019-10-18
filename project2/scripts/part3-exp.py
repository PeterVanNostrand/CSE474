import numpy as np
import matplotlib.pyplot as plt
import util_mnist_reader
import keras as k


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
            512,
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

    # return the constructed network architecture
    return model


def py_create_model(width, height, depth, classes):
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    model = k.models.Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    # first CONV => RELU => CONV => RELU => POOL layer set
    model.add(k.layers.Conv2D(32, (3, 3), padding="same",
    input_shape=inputShape))
    model.add(k.layers.Activation("relu"))
    model.add(k.layers.BatchNormalization(axis=chanDim))
    model.add(k.layers.Conv2D(32, (3, 3), padding="same"))
    model.add(k.layers.Activation("relu"))
    model.add(k.layers.BatchNormalization(axis=chanDim))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.Dropout(0.25))

    # second CONV => RELU => CONV => RELU => POOL layer set
    model.add(k.layers.Conv2D(64, (3, 3), padding="same"))
    model.add(k.layers.Activation("relu"))
    model.add(k.layers.BatchNormalization(axis=chanDim))
    model.add(k.layers.Conv2D(64, (3, 3), padding="same"))
    model.add(k.layers.Activation("relu"))
    model.add(k.layers.BatchNormalization(axis=chanDim))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.Dropout(0.25))

    # first (and only) set of FC => RELU layers
    model.add(k.layers.Flatten())
    model.add(k.layers.Dense(512))
    model.add(k.layers.Activation("relu"))
    model.add(k.layers.BatchNormalization())
    model.add(k.layers.Dropout(0.5))

    # softmax classifier
    model.add(k.layers.Dense(classes))
    model.add(k.layers.Activation("softmax"))

    # return the constructed network architecture
    return model

if __name__ == '__main__':
    print("Part 1: Neural Network")
    fashion_path="C:\\Users\peter\Documents\Peter\\7-College\\7-Fall-2019\CSE474\Projects\project2\data\\fashion"

    # Import datasets
    ((x_train, y_train), (x_test, y_test)) = k.datasets.fashion_mnist.load_data()

    # Normalize inputs to [0,1]
    x_train = x_train / 255
    y_train = y_train / 255

    # Create a CNN model using Keras
    # model = create_model((28, 28, 1), -1, 10)
    model = py_create_model(28, 28, 1, 10)

    # Compile the model, optimizing with SGD
    learn_rate = 0.01
    num_epochs = 30
    batch_size = 32
    sgd = k.optimizers.SGD(lr=learn_rate, momentum=0.5, decay=learn_rate / num_epochs)
    # NUM_EPOCHS = 25
    # INIT_LR = 1e-2
    # BS = 32
    # sgd = k.optimizers.SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
    model.compile(
        optimizer=sgd,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # reshape data to num_samples x rows x columns x depth
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    # Convert labels to one hot
    y_train = k.utils.to_categorical(y_train, num_classes=10)
    y_test = k.utils.to_categorical(y_test, num_classes=10)

    # Train the model
    H = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=30)
    # model.fit(x_train, y_train,	validation_data=(x_test, y_test), batch_size=BS, epochs=NUM_EPOCHS)

    # Evaluate the model
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print(score)