#-*- coding: UTF-8 -*-
# Read in the first 20 instances from 1a dataset
# initialize k
# Generate projection matrix
# Projecting all pairs
# plot distortion
import numpy as np
import cPickle
import gzip
import itertools
from scipy import stats
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter  
from sklearn import preprocessing
from scipy.spatial.distance import cdist

def load_data():
    f = gzip.open('F:/DataMining/Assignment/set1/1454491793_864__data1a/data1a/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return training_data, validation_data, test_data

if __name__ == "__main__":
    tr_d, va_d, te_d = load_data()
    # Normalize the first 20 training data

    tr = tr_d[0][0:20]
    # initialize k
    k = int(raw_input('Please assign value of k (50 or 100 or 500) --> '))
    wanted_dist = float(raw_input('Please assign maximum value of allowed distortion error (e.g. 0.1) --> '))
    # Randomize projection matrix
    d = len(tr[0])
    print d
    prob_rij = [0.5,0.5]
    generater = stats.rv_discrete(values = ([-1,1], prob_rij))
    projection = generater.rvs(size = (k,d))
    projection = projection / (np.sqrt(d))
    # Projection & Normalize
    tr_pro = np.dot(projection, tr.T)

    #tr_pro = tr_pro / (np.sqrt(k))
    #tr_pro = preprocessing.normalize(tr_pro,norm = 'l2')
    
    tr_pro = tr_pro.T

    # Calculate distortion
    # It came to me that this part can actually also be calculated by dot product of matrix
    permutation = list(itertools.combinations(np.arange(len(tr)),2))
    # Initialize distortion list
    distortion = []
    dist_count = 0
    for i in range(0,len(permutation)):
        val_1 = permutation[i][0]
        val_2 = permutation[i][1]  

        dist_val = distance.euclidean(tr_pro[val_1],tr_pro[val_2])/distance.euclidean(tr[val_1],tr[val_2])
        distortion.append(dist_val)
        # print val_1 
        # print val_2
        # print distance.euclidean(tr_pro[val_1],tr_pro[val_2])
        # print distance.euclidean(tr[val_1],tr[val_2])
        if (np.absolute(dist_val-1) < wanted_dist):
            dist_count += 1

    # Plot distortion values
    distortion_average = sum(distortion)/ len(distortion) 
   
    
    plt.scatter(np.arange(len(distortion)),distortion, color = 'blue')
    plt.plot([-10, len(distortion)+20],[distortion_average, distortion_average], color = 'cyan')
    #plt.plot([-10, len(distortion)+20],[1-wanted_dist,1-wanted_dist], color = 'cyan')
    plt.axis([-10, len(distortion)+20, 0, 1.0])
    plt.title('Projection: ' + str(20) + 'x' + str(d) +' --> ' + str(20) + 'x' + str(k) + "\n" +
              'The average value is' + str(distortion_average))
    plt.show()
