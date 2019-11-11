from matplotlib import pyplot as plt
import keras
import numpy as np
import copy
import math
import random
from sklearn import metrics
from sklearn.mixture import GaussianMixture
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

def create_autoencoder(input_dim, encoding_dim):    
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
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

def extract_encoder(input_dim, autoencoder):
    # Separate Encoder model
    input_img = Input(shape=(input_dim,))
    encoder_layer1 = autoencoder.layers[0]
    encoder_layer2 = autoencoder.layers[1]
    encoder_layer3 = autoencoder.layers[2]
    encoder = Model(input_img, encoder_layer3(encoder_layer2(encoder_layer1(input_img))))
    return encoder

def train_autoencoder(autoencoder, x_train, x_test, ne):
    progress = autoencoder.fit(x_train, x_train, epochs=ne, batch_size=256, validation_data=(x_test, x_test))
    return progress

def autoencode(x_train, x_test):
    # Get info about data
    input_dim = x_train.shape[1]
    # Genereate an autoencoder model
    print("Creating autoencoder...")
    encoding_dim = 32
    autoencoder = create_autoencoder(input_dim, encoding_dim)
    # Train the model on the trainind data
    print("Training autoencoder...")
    num_epochs = 50
    progress = train_autoencoder(autoencoder, x_train, x_test, num_epochs)
    # Plot the autoencoder loss over time
    plot_stats(progress, num_epochs)
    # Extract the encoder phase of the autoencoder
    encoder = extract_encoder(input_dim, autoencoder)
    # Use the encoder to compress the inputs to 
    print("Compressing tranining images...")
    compressed_train_images = encoder.predict(x_train)
    print("Compressing test images...")
    compressed_test_images = encoder.predict(x_test)
    
    return compressed_train_images, compressed_test_images

if __name__ == '__main__':
    print("Auto-Encoder with GMM Clustering")
    k = 10 # Number of clusters

    print("Loading dataset...")
    ((x_train, y_train), (x_test, y_test)) = keras.datasets.fashion_mnist.load_data()
    x_train = np.reshape(x_train, (x_train.shape[0], 784))
    x_train = x_train / 255.0
    x_test = np.reshape(x_test, (x_test.shape[0], 784))
    x_test = x_test / 255.0

    # Use auto encoder to reduce dimensionality, returns compressed rep of x_train, x_test
    cx_train, cx_test = autoencode(x_train, x_test)

    # Perform GMM clustering
    print("Training GMM...")
    gmm = GaussianMixture(n_components=k)
    gmm.fit(cx_train)
    print("Clustering training data...")
    clusterAssmentTrain = gmm.predict(cx_train)
    print("Clustering test data...")
    clusterAssmentTest = gmm.predict(cx_test)
    print("Done!")

    # Compute Metrics
    print("Training Metrics:")
    evaluate_clusters(10, clusterAssmentTrain, y_train)
    print("Testing Metrics:")
    evaluate_clusters(10, clusterAssmentTest, y_test)

    plt.show()