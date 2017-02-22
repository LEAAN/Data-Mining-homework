from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Initialization
k = np.array(range(1,50))
# Number of features & centers
n_samples = 1000
n_features = 2
n_centers = 2
random_state = 0
test_size = 0.2
# SVM Parameters
kernel = ['linear','poly','rbf']
C = np.logspace( -15, 15, 100, base=2)
gamma = np.logspace(-15, 15, 100, base=2)
states = range(0,10)

def tune_svm(kernel_method, label, c_value = 1.0, gamma_value = 0.5):
		clf = SVC(kernel = kernel_method, C = c_value, gamma = gamma_value)
		a = cross_validation.cross_val_score(clf,X,y,scoring ='roc_auc', cv = 10)
		return np.mean(a)
		
def show_svm(title):
	plt.legend(loc=1, fontsize = 9, scatterpoints =1)
	plt.title(title, fontsize = 12)
	plt.show()

for k in kernel:
	aucs = []
	for state in states:
		print 'new state is: ' + str(state)
		print 'kernel is:' + k
		X, y = make_blobs(n_samples=n_samples, centers= n_centers, n_features= n_features, random_state= state)
		X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size, random_state= random_state)
		auc = tune_svm(k, k)
		aucs.append(auc)
	plt.plot(range(0, len(states)), aucs, label = k, marker = 'D')

plt.legend(loc=1, fontsize = 9, scatterpoints =1)
plt.ylim((0.7,1.1))
plt.title('AUC values for kernels in 10 sets of random samples', fontsize = 12)
plt.show()

# # In Git Folder

