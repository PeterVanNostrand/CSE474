from matplotlib import pyplot as plt
import keras
import numpy as np
import copy
import math
import random
from sklearn import metrics
from sklearn.cluster import KMeans


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

if __name__ == '__main__':
    print("Autoencoder kMeans start")
    k = 10 # Number of clusters
    ((x_train, y_train), (x_test, y_test)) = keras.datasets.fashion_mnist.load_data()
    
    x_train = np.reshape(x_train, (x_train.shape[0], 784))
    x_train = x_train / 255.0
    x_test = np.reshape(x_test, (x_test.shape[0], 784))
    x_test = x_test / 255.0

    clusterer = KMeans(n_clusters=k)
    clusterAssment = clusterer.fit_predict(x_train)
    # centroids, clusterAssment = kMeans(x_train, 1000, 10)
    ars = metrics.adjusted_rand_score(y_train, clusterAssment)
    accuracy, purity, clustsByLabel = evaluate_clusters(k, clusterAssment, y_train)

    print("Accuracy: ", accuracy)
    print("ARS: ", ars)
    print("Purity:\n", purity)
    print("Done!")
