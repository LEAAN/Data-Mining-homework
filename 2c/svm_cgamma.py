
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn import cross_validation
# from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

##############################################################
# Generate and prepare data set
n_samples = 1000
n_features = 2
n_centers = 2
random_state = 0
test_size = 0.2

X, y = make_blobs(n_samples=n_samples, centers= n_centers, n_features= n_features, random_state= random_state)
X_2d = X
y_2d = y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state= random_state)

scaler = StandardScaler()
X_2d = scaler.fit_transform(X_2d)

##############################################################
# Choose C & gamma values
C_2d_range = np.logspace( -15, 15, 8, base=2)
gamma_2d_range = np.logspace(-15, 15, 8, base=2)

classifiers = []
aucs = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma, probability = True)
        # clf = GridSearchCV(svm.SVC(kernel="rbf"), param_dict, cv=10, scoring="roc_auc")
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))

##############################################################################
# visualization
plt.figure(figsize=(9, 9))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # get auc scores for each calssifiers
    a = cross_validation.cross_val_score(clf,X,y,scoring ='roc_auc', cv = 10)
    # print 'C: '+ str(C) + 'gamma: ' + str(gamma) + 'auc: ' 
    # print a
    a = np.mean(a)
    aucs.append(a)

    # # visualize decision function for these parameters
    # plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    # plt.title("gamma=" + str(gamma) + '\n'+ "C=" + str(C),
    #           size='small')

    # # visualize parameter's effect on decision function
    # plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    # plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r)
    # plt.xticks(())
    # plt.yticks(())
    # plt.axis('tight')

# Heatmap for auc scores
scores = np.reshape(aucs, (len(C_2d_range),len(gamma_2d_range)))
plt.figure(figsize=(8, 8))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_2d_range)), gamma_2d_range, rotation=45)
plt.yticks(np.arange(len(C_2d_range)), C_2d_range)
plt.title('Validation AUC')
plt.show()


