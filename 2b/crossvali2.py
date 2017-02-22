from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.metrics import precision_score
# from sklearn import cross_validation, datasets
from sklearn.cross_validation import KFold
from statistics import mean

import graph as ga

# Choices of K
k = np.array(range(1,50))
# Array to store accuracy values
precision = []
# Random sample
n_samples = 1000
X, y = make_blobs(n_samples=n_samples, centers=3, n_features=2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# K-Fold
kf = KFold(1000,n_folds = 10)
for ki in k:
    # Average precision of 10-fold
    kip = []
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = KNeighborsClassifier(n_neighbors= ki )
        clf.fit(X_train,y_train)
        y_test_predict = clf.predict(X_test)
        p = precision_score(y_test, y_test_predict, average = 'micro')
        kip.append(p)
    precision.append(np.mean(kip))

# Misclassification rate
mce = 1 - np.array(precision)
mce_min = min(mce)
print 'min misclassification error is: ' + str(mce_min)
# Plot misclassification Rate
ga.plot_mce(k,mce,None)






clf = KNeighborsClassifier(n_neighbors= k[0])
clf.fit(X,y)
clf.predict(test)















# FROM THE WEBPAGE AS EXAMPLE
n_neighbors = 15

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()