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
    ars = metrics.adjusted_rand_score(clusterAssment, labels)

    print("\tAccuracy: ", accuracy)
    print("\tARS: ", ars)
    print("\tPurity: ", purity)

    return accuracy, ars, purity, clustsByLabel

if __name__ == '__main__':
    print("kMeans start")
    k = 10 # Number of clusters
    ((x_train, y_train), (x_test, y_test)) = keras.datasets.fashion_mnist.load_data()
    
    x_train = np.reshape(x_train, (x_train.shape[0], 784))
    x_train = x_train / 255.0
    x_test = np.reshape(x_test, (x_test.shape[0], 784))
    x_test = x_test / 255.0

    clusterer = KMeans(n_clusters=k)
    clusterAssmentTrain = clusterer.fit_predict(x_train)
    clusterAssmentTest = clusterer.predict(x_test)

    print("Done!")

    print("Training")
    evaluate_clusters(k, clusterAssmentTrain, y_train)
    print("Testing")
    evaluate_clusters(k, clusterAssmentTest, y_test)

    
