from matplotlib import pyplot as plt
import keras
import numpy as np
import copy
import math
import random
from sklearn import metrics
from sklearn.cluster import KMeans
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import regularizers
from keras.models import Model, Sequential


def evaluate_clusters(k, clusterAssment, labels):
    clustsByLabel = np.zeros((k,k))
    clusterAssment = clusterAssment.reshape((len(clusterAssment),))
    correctlyAssigned = 0

    for i in range(0, len(clusterAssment)):
        clustsByLabel[clusterAssment[i]][labels[i]] += 1
        if clusterAssment[i] == labels[i] : correctlyAssigned += 1

    maxLabelFreq = np.max(clustsByLabel, axis=1)
    freqPerClust = np.sum(clustsByLabel, axis=1)

    purity = [0.0]* k
    for i in range(0, k):
        if freqPerClust[i] > 0:
            purity[i] = maxLabelFreq[i] / freqPerClust[i]

    accuracy = correctlyAssigned / len(clusterAssment)
    ars = metrics.adjusted_rand_score(clusterAssment, labels)

    print("\tAccuracy: ", accuracy)
    print("\tARS: ", ars)
    print("\tPurity: ", purity)

    return accuracy, ars, purity, clustsByLabel

def plot_stats(progress, num_epochs):
    train_loss = progress.history["loss"]
    val_loss = progress.history["val_loss"]

    time = np.arange(num_epochs)

    plt.figure()
    plt.plot(time, train_loss, label="training loss")
    plt.plot(time, val_loss, label="validation loss")
    plt.title("Losss vs Epochs")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()

def create_autoencoder(shape):
    autoencoder = Sequential()

    # Encoder Layers
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=shape))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))

    # Decoder Layers
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

    return autoencoder

def extract_encoder(autoencoder):
    # Separate Encoder model
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('max_pooling2d_2').output)
    return encoder

def train_autoencoder(autoencoder, x_train, x_test, ne):
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    progress = autoencoder.fit(x_train, x_train,  epochs=ne, batch_size=128, validation_data=(x_test, x_test))

    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    # progress = autoencoder.fit(x_train, x_train, epochs=ne, batch_size=256, validation_data=(x_test, x_test))

    return progress

def autoencode(x_train, x_test):
    # Reshape data to 2D for convolutional layers to use
    x_train = x_train.reshape((len(x_train), 28, 28, 1))
    x_test = x_test.reshape((len(x_test), 28, 28, 1))

    # Genereate an autoencoder model
    print("Creating autoencoder...")
    shape = x_train.shape[1:]
    autoencoder = create_autoencoder(shape)
    # Train the model on the trainind data
    print("Training autoencoder...")
    num_epochs = 50
    progress = train_autoencoder(autoencoder, x_train, x_test, num_epochs)
    # Plot the autoencoder loss over time
    plot_stats(progress, num_epochs)
    # Extract the encoder phase of the autoencoder
    encoder = extract_encoder(autoencoder)
    # Use the encoder to compress the inputs to 32 d
    print("Compressing training images...")
    compressed_train_images = encoder.predict(x_train)
    print("Compressing test images...")
    compressed_test_images = encoder.predict(x_test)
    
    return compressed_train_images, compressed_test_images

if __name__ == '__main__':
    print("kMeans start")
    k = 10 # Number of clusters

    print("Loading data...")
    ((x_train, y_train), (x_test, y_test)) = keras.datasets.fashion_mnist.load_data()
    x_train = np.reshape(x_train, (x_train.shape[0], 784))
    x_train = x_train / 255.0
    x_test = np.reshape(x_test, (x_test.shape[0], 784))
    x_test = x_test / 255.0

    # Use auto encoder to reduce dimensionality, returns compressed rep of x_train, x_test
    cx_train, c_xtest = autoencode(x_train, x_test)

    # Perform kMeans clustering
    clusterer = KMeans(n_clusters=k)
    clusterAssmentTrain = clusterer.fit_predict(cx_train)
    clusterAssmentTest = clusterer.predict(c_xtest)
    print("Done!")

    # Compute Metrics
    print("Training")
    evaluate_clusters(10, clusterAssmentTrain, y_train)
    print("Testing")
    evaluate_clusters(10, clusterAssmentTest, y_test)

    plt.show()