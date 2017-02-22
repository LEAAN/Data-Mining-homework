# Source: http://neuralnetworksanddeeplearning.com/chap1.html
# You need only train and test sets. Simply disregard the validation set.
import numpy as np
import cPickle
import gzip
from sklearn import preprocessing
from scipy.spatial.distance import cdist

def load_data():
    f = gzip.open('./data1a/mnist.pkl.gz', 'rb')
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

def trainer():
    tr_d, va_d, te_d = load_data()
    # Use dot product as mesurement of similarity
    tr = preprocessing.normalize(tr_d[0],norm= 'l2')
    te = preprocessing.normalize(te_d[0],norm= 'l2')
    # dist = cdist(tr,te,metric = "euclidean")
    # label = dist.argmin(axis = 1)
    sim = np.dot(te, tr.T)
    label = sim.argmax(axis = 1)
    # initial te_label to save labels of vectors in test set
    te_label = np.zeros(len(te_d[0]))
    i = 0
    # To each vector in test set
    # assign label of the matched vector in training set that maximize their similarity
    while i < len(label):
        a = label[i]
        te_label[i] = tr_d[1][a]
        i += 1
    # initial confusion matrix, precision array, and recall array
    confusion = np.zeros((10,10))
    precision = np.zeros((10,1))
    recall = np.zeros((10,1))
    # calculate confusion matrix
    i = 0
    while i < len(label):
        actual = te_d[1][i]
        predict = te_label[i]
        confusion[actual][predict] += 1
        i +=1
    # calculate precision & recall 
    i = 0
    while i< len(precision):
        precision[i] = confusion[i][i]/np.sum(confusion, axis = 0)[i]
        i +=1
    i = 0
    while i< len(recall):
        recall[i] = confusion[i][i]/np.sum(confusion, axis = 1)[i]
        i += 1

    # Save confusion matrix, recall and precision array to seperate csv file
    np.savetxt('confusion.csv', confusion, fmt='%10.5f', delimiter=',')
    np.savetxt('precision.csv', precision, fmt='%10.5f', delimiter=',')
    np.savetxt('recall.csv', recall, fmt='%10.5f', delimiter=',')

tr_w, va_w, te_w = load_data_wrapper() 
trainer()


