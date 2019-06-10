import numpy as np
import matplotlib.pyplot as plt


def generateRandomPoints(N):
    """This function generates N random points"""
    return 10*np.random.random((N,2))


def initCentroids(k):
    """This function intializes k random centroids"""
    return 10*np.random.random((k,2))


def computeEuclideanDistance(c, x):
    """This function computes the power of the Euclidean distance between two points"""
    return np.power(np.linalg.norm(c - x), 2)
    
    
def computeKMeans(k, centroids, datapoints):
    """This function computes the k-means:
        parameters:
        - k #number of centroids
        - centroids #the centroids list
        - datapoints #list of points to compute k-means on
        returns:
        - c_records #list of indeces references for each point
        - centroids #updated centroids
    """
    c_records = list()
    distances = list()

    #calcolo le distanze e creo una lista di riferimenti a chi possiede quale punto dei datapoints
    for x in datapoints:
        for c in centroids:
            distances.append(computeEuclideanDistance(c, x))
        c_records.append(np.argmin(distances))
        distances = list()

    #aggiorno i centroidi
    centroids_lengths = np.zeros(k, np.int)
    centroids = np.zeros((k, 2))
    
    for c_index, point in zip(c_records, datapoints):
        centroids[c_index] += point
        centroids_lengths[c_index] += 1
    
    for i in range(k):
        centroids[i] = centroids[i]/centroids_lengths[i]
    return c_records, centroids

def plot(k, indeces, centroids, datapoints):
    print("Plotting...")
    for i in range(k):
        x = []
        y = []
        plt.scatter(centroids[i][0], centroids[i][1])
        for index, point in zip(indeces, datapoints):
            if index == i:
                x.append(point[0])
                y.append(point[1])
        plt.scatter(x, y)


    plt.title("K-Means - Scatter plot")   
    plt.show()
    
k = 5 #Number of centroids
N = 100 #Number of test points
N_iter = 2000 #Max number of iterations

print("# K-Means: k = %d, N points = %d, N iterations = %d" % (k, N, N_iter))
datapoints = generateRandomPoints(100)
centroids = initCentroids(k)

for i in range(N_iter):
    print("Iteration: ", i+1)
    indeces, new_centroids = computeKMeans(k, centroids, datapoints)

    if (centroids == new_centroids).all():
        print("Stop condition: centroids not changing at iteration ", i+1)
        plot(k, indeces, new_centroids, datapoints)
        exit(1)
    else:
        centroids = new_centroids


print("Stop condition: end of iterations")
plot(k, indeces, centroids, datapoints)
exit(2)