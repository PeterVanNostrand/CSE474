from matplotlib import pyplot as plt
import keras
import numpy as np
import copy
import math
import random
from sklearn import metrics

def assignCluster(dataSet, k, centroids):
    print("\tAssigning Clusters...")
    clusterAssment = np.full((dataSet.shape[0],1), 0)
    minDist = np.full((dataSet.shape[0],1), float("inf"))
    for i in range(0, centroids.shape[0]):
        diff = dataSet - centroids[i]
        sqr_diff = np.square(diff)
        sos_diff = np.sum(sqr_diff, axis=1)
        euclDist = np.sqrt(sos_diff)
        clusterAssment[euclDist < minDist] = i
        minDist = np.minimum(minDist, euclDist)
    print("\tDone!")
    return clusterAssment

def getCentroid(dataSet, k, clusterAssment):
    print("\tUpdating Centroids...")
    centroids = np.mat(np.zeros((k, dataSet.shape[1]))) # array of new cluster centroids
    dpInCluster = np.zeros((k, 1)) # Number of datapoints in a given cluser
    # Compute the new centroids as average of all points within the corresponding cluster
    for i in range(0, dataSet.shape[0]): # Take the sum of all points within the cluster
        centroids[clusterAssment[i]] += dataSet[i]
        dpInCluster[clusterAssment[i]] += 1
    
    # Divide by the number of points in the cluster to get average
    for j in range(0, k):
        if dpInCluster[j] != 0:
            centroids[j] /= dpInCluster[j]
    print("\tDone!")
    return centroids

def kMeans(dataSet, T, k):
    # Initialize centroids array to random set of images
    centroids = np.mat(np.zeros((k, dataSet.shape[1])))
    for idx in range(0, k):
        randIndex = random.randint(0, dataSet.shape[0])
        centroids[idx] = dataSet[randIndex]

    clusterAssment = [0] * len(dataSet)
    pre_clusters  = [1] * len(dataSet)
    i=1
    while i < T and list(pre_clusters) != list(clusterAssment):
        print("Iteration:", i)
        pre_clusters = copy.deepcopy(clusterAssment) 
        clusterAssment = assignCluster(dataSet, k, centroids )
        centroids      = getCentroid(dataSet, k, clusterAssment)
        i=i+1
    print("Total Iterations:", i)
    return centroids, clusterAssment

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
    print("kMeans start")
    k = 10 # Number of clusters
    ((x_train, y_train), (x_test, y_test)) = keras.datasets.fashion_mnist.load_data()
    
    x_train = np.reshape(x_train, (x_train.shape[0], 784))
    x_train = x_train / 255.0
    x_test = np.reshape(x_test, (x_test.shape[0], 784))
    x_test = x_test / 255.0

    centroids, clusterAssment = kMeans(x_train, 1000, 10)
    accuracy = metrics.adjusted_rand_score(x_train, clusterAssment)
    accuracy, purity, clustsByLabel = evaluate_clusters(k, clusterAssment, y_train)

    print("Accuracy:\n", accuracy)
    print("Purity:\n", purity)
    print("Done")
