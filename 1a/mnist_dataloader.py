#-*- coding: UTF-8 -*-
'''
Created on 2016年2月16日

@author: p c
'''
import numpy as np
import cPickle
import gzip

def load_data():
    f = gzip.open('F:/DataMining/Assignment/set1/1454491793_864__data1a/data1a/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return training_data, validation_data, test_data


def get_matrix():
    
    record_cos= []
    te_label = []
    for index_te in range(0,len(te_d[0])):
        record_cos.append(0)
        te_label.append(0)
        x = np.array(te_d[0][index_te])
        for index_tr in range(0,len(tr_d[0])):
            y = np.array(tr_d[0][index_tr])
            Lx = np.sqrt(x.dot(x))
            Ly = np.sqrt(y.dot(y))
            cos_angle = x.dot(y)/(Lx*Ly)
            if cos_angle > record_cos[index_te]:
                record_cos[index_te] = cos_angle
                te_label[index_te] = tr_d[1][index_tr]       
        #print te_label[index_te],te_d[1][index_te]
#     sim = np.dot(te_d[0], tr_d[0].T)
#     label = sim.argmax(axis = 1)
    
    #每行最大的index
    
    #创建全是0的矩阵
    confusion = np.zeros((10,10),dtype=np.int)
    precision = np.zeros((10,1))
    recall = np.zeros((10,1))
    
#     for i in range(0, len(label)):
#         a = label[i]
#         te_label.append(tr_d[1][a])
    
    
    for i in range(0, len(te_label)):

        actual = te_d[1][i]
        predict = te_label[i]
        confusion[actual][predict] += 1.0

    
    i = 0
    while i< len(precision):
        precision[i] = confusion[i][i]/np.sum(confusion, axis = 0)[i]
        i +=1
    i = 0
    while i< len(recall):
        recall[i] = confusion[i][i]/np.sum(confusion, axis = 1)[i]
        i += 1
    print confusion
    
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    #print (va_d[0].shape)
    #print (va_d[0][0].shape)
    #print (va_d[0][0][0])
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return training_data, validation_data, test_data

tr_d, va_d, te_d = load_data()
tr_w, va_w, te_w = load_data_wrapper() 
get_matrix()

