#-*- coding: UTF-8 -*-
# Source: http://neuralnetworksanddeeplearning.com/chap1.html
# You need only train and test sets. Simply disregard the validation set.
import numpy as np
import cPickle
import gzip
import itertools
from scipy.spatial.distance import cdist
from scipy import stats
from scipy.spatial import distance
import matplotlib.pyplot as plt
from astropy.io.votable.converters import Boolean

def load_data():
    f = gzip.open('F:/DataMining/Assignment/set1/1454491793_864__data1a/data1a/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return training_data, validation_data, test_data

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return training_data, validation_data, test_data

def computMatrix(withProjection):
    tr_d, va_d, te_d = load_data()
    if withProjection == True:
        tr, te = doProjection()
    else:
        tr = tr_d[0]
        te = te_d[0]
    
    # Use dot product as mesurement of similarity
    
   
    dist = cdist(te,tr,metric = "euclidean")
    label = dist.argmin(axis = 1)
    
    # initial te_label to save labels of vectors in test set
    te_label = np.zeros(len(te_d[0]))

    # To each vector in test set
    # assign label of the matched vector in training set that maximize their similarity
    for i in range(0, len(label)):
        a = label[i]
        te_label[i] = tr_d[1][a]
        i += 1
    # initial confusion matrix, precision array, and recall array
    confusion = np.zeros((10,10),dtype=np.int)
    precision = np.zeros((10,1))
    recall = np.zeros((10,1))
    # calculate confusion matrix

    for i in range(0, len(label)):
        actual = te_d[1][i]
        predict = te_label[i]
        confusion[actual][predict] += 1

    print confusion
    
def doProjection():
    tr_d, va_d, te_d = load_data()
    # Normalize the first 20 training data

    tr = tr_d[0]
    te = te_d[0]
    # initialize k
    k = int(raw_input('Please assign value of k (50 or 100 or 500) --> '))
    d = len(tr[0])
    scale = int(raw_input('assign a value for scale '))  
    
    
    generater = stats.rv_discrete(values = ([-1,1], [0.5,0.5]))
    projection = generater.rvs(size = (k,d))
    projection = projection / (np.sqrt(d))
    # Projection & Normalize
    tr_pro = np.dot(projection, tr.T)
    #tr_pro = tr_pro / (np.sqrt(k))
    tr_pro = tr_pro.T * scale
    
    te_pro = np.dot(projection, te.T)
    #te_pro = te_pro / (np.sqrt(k))
    te_pro = te_pro.T * scale
    
    return tr_pro, te_pro


if __name__ == "__main__":
    
    k = int(raw_input('whether implement projection(yes -> 1/ no -> 2) '))    
    if k == 1 :
        computMatrix(True)
    else:
        computMatrix(False)
