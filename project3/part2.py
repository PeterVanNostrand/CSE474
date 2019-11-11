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
    return accuracy, purity, clustsByLabel

def autoencode_step(x_train, x_test):
    input_dim = x_train.shape[1]
    encoding_dim = 32
    
    # Autoencoder
    autoencoder = Sequential()
    # Encoder Layers
    autoencoder.add(Dense(4 * encoding_dim, input_shape=(input_dim,), activation='relu'))
    autoencoder.add(Dense(2 * encoding_dim, activation='relu'))
    autoencoder.add(Dense(encoding_dim, activation='relu'))
    # Decoder Layers
    autoencoder.add(Dense(2 * encoding_dim, activation='relu'))
    autoencoder.add(Dense(4 * encoding_dim, activation='relu'))
    autoencoder.add(Dense(input_dim, activation='sigmoid'))
    # autoencoder.summary()

    # Separate Encoder model
    input_img = Input(shape=(input_dim,))
    encoder_layer1 = autoencoder.layers[0]
    encoder_layer2 = autoencoder.layers[1]
    encoder_layer3 = autoencoder.layers[2]
    encoder = Model(input_img, encoder_layer3(encoder_layer2(encoder_layer1(input_img))))
    # encoder.summary()

    # Train the model
    print("Training Auto Encoder...")
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, validation_data=(x_test, x_test))
    print("Done!")

    print("Compressing Images...")
    compressed_images = encoder.predict(x_test)
    print("Done!")
    return compressed_images

if __name__ == '__main__':
    print("kMeans start")
    k = 10 # Number of clusters
    ((x_train, y_train), (x_test, y_test)) = keras.datasets.fashion_mnist.load_data()
    
    x_train = np.reshape(x_train, (x_train.shape[0], 784))
    x_train = x_train / 255.0
    x_test = np.reshape(x_test, (x_test.shape[0], 784))
    x_test = x_test / 255.0

    # Use auto encoder to reduce dimensionality
    compressed_images = autoencode_step(x_train, x_test) # Returns compressed rep of x_test

    # Perform kMeans clustering
    clusterer = KMeans(n_clusters=k)
    clusterAssment = clusterer.fit_predict(compressed_images)

    # Compute Metrics
    ars = metrics.adjusted_rand_score(y_test, clusterAssment)
    accuracy, purity, clustsByLabel = evaluate_clusters(k, clusterAssment, y_train)

    print("Accuracy: ", accuracy)
    print("ARS: ", ars)
    print("Purity:\n", purity)
    print("Done!")