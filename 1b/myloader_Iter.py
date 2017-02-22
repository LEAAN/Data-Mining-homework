
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cdist

def load_data_1b(fpath):
    data = []
    f = open(fpath, 'r')
    for line in f:
        words = line.split()
        data.append(words)
    f.close()
    arr = np.array(data, dtype=np.float64)
    return arr[:, 1:]


# What these are? How it works? Why do we have to write it like this?
if __name__ == "__main__":
    c1 = load_data_1b("./data1b/C2.txt")
    dim = c1.shape[1]
def init_center(init):
	# clear previous center
	center  = {
 	'1': c1[0:k,:],
 	'2': c1[np.random.random_integers(0,len(c1),k)],
 	'3': kplus_center(c1,k),
 	'4': Gonzales_center(c1, k)
	}[init]
	return center

def kplus_center(c1, k):
	center = np.zeros((k, dim))
	center[0] = c1[np.random.randint(len(c1))]
	r = 1
	while r < k:
		dist = cdist(c1, center [0:r], metric = 'euclidean')
		fai = np.square(dist.min(axis = 1))
		fai /= sum(fai)
		generater = stats.rv_discrete(values=(np.arange(0,len(c1)), fai))
		center[r] = c1[generater.rvs(size = 1)]
		r += 1
	return center 

def Gonzales_center(c1, k):
	center = np.zeros((k, dim))
	center[0] = c1[0]
	r = 1
	while r < k:
		# dist = np.zeros((len(c1),center))
		dist = cdist(c1,center[0:r],metric = "euclidean")
		dist = dist.min(axis=1)
		center[r] = c1[np.argmax(dist)]
		r += 1
	return center 

def main():	

    # cluster_color = ['red','cyan','yellow','grey','black'] 
	init_method = ['the first k points of the data set',
		'k points of P picked uniformly at random',
		'Use k-means++',
		'Use Gonzales’ algorithm']

	init = (raw_input("Please choose an initializing method:" + "\n" + "1. the first k points of the data set"
		+ "\n" + "2. k points of P picked uniformly at random" + "\n" + "3. Use k-means++" + "\n" +
		"4. Use Gonzales’ algorithm" + "\n" + "--> "))
	# if (init == 3 | 4):
	# 	iterations = int(raw_input('Please assign number of iterations --> '))
	global k 
	k = int(raw_input('Please assign number of clusters --> '))	
	# initialize center 
	center = init_center(init)
	if (int(init) < 4):
		# K-means/ K-means++
		print "Init is Less than 4"
		count = 0
		new_center = np.zeros((k,dim))
		new_center, check, cluster = update_center(c1,center)
		while(check == False):
			count += 1
			center = new_center
			new_center = np.zeros((k,dim))
			new_center, check, cluster = update_center(c1,center)
		count += 1
	else:
		#  Gonzales’ algorithm
		count = k
		new_center = center 

	# Calculate for cost
	dist = cdist(c1,new_center,metric = "euclidean")
	cost = np.amax(dist)
	print count
	print cost 
	print "Set of center points is: " + "\n" + str(center)

	for i in range(0,k):
		index = np.where(cluster==i)[0]
		plt.scatter(c1[index][0],c1[index][1], color = 'red') 

	plt.title('Initializ Method : ' + init_method[int(init)-1] + '\n' +
		"Number of iterations : " + str(count) + '\n' 
		+ "Cost : " + str(cost))
	plt.show()

	# # Plot clusters
	plt.scatter(c1[:,0],c1[:,1],color = 'red')
	plt.scatter(center[:,0],center[:,1],color = 'blue')
	# plt.title('Initializ Method : ' + init_method[int(init)-1]  )
	plt.show()


def update_center(c1,center):

    # color = ['red','cyan','yellow','grey','black'] 
	# get distance matrix
	cluster = np.zeros((3,1))
	dist = cdist(c1,center,metric = "euclidean")
	# assign label
	label = dist.argmin(axis = 1)
	# update center 
	new = np.zeros((k,dim))
	for i in range(0,k):
		# index = np.where(label==i)[0]
		# new[i] = np.mean(c1[index], axis = 0)
		new[i] = np.mean(c1[np.where(label==i)[0]], axis = 0)
		# print np.mean(c1[index], axis = 0)
		# print index
		# cluster[str(i)] = c1[index]
		# plt.scatter(c1[index][0],c1[index][1],color = 'red')
	a = (new == center).all()
	# print a 
	# print "new_center"
	# print new
	# print "center"
	# print center
	return new, a ,label 
	
main()