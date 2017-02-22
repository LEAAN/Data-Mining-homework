  
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import graph as ga

# index = raw_input('Please Choose Classification Method (1. k-Nearest Neighbours 2. DecisionTree )--> ')

# Choices of K
k = np.array(range(1,50))
# Number of features & centers
n_samples = 1000
n_features = 2
n_centers = 3
random_state = 0
n_estimators = 100
test_size = 0.2
# Array to store accuracy values
accu_kf = []
accu_boot = []
# Random sample
X, y = make_blobs(n_samples=n_samples, centers= n_centers, n_features= n_features, random_state= random_state)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state= random_state)

# Accuracy for each k value
for i in k:
    # Initialize NN Classifier
    clf = tree.DecisionTreeClassifier(random_state= random_state, min_samples_leaf= i )
    # clf = KNeighborsClassifier(n_neighbors= i)
    bagging = BaggingClassifier(clf, max_features= n_features, n_estimators = n_estimators)
    # accuracy for KFold
    a = cross_validation.cross_val_score(clf,X,y,scoring = 'accuracy', cv = 10)
    ave_a = np.mean(a)
    accu_kf.append(ave_a)

    # accuracy for bagging
    bagging.fit(X_train,y_train)
    y_test_predict = bagging.predict(X_test)
    a = accuracy_score(y_test, y_test_predict)
    accu_boot.append(a)

# Classification error rate 
mce_kf = 1-np.array(accu_kf)
mce_boot = 1-np.array(accu_boot)
# Optimal K
opt_kf = np.argmin(mce_kf)+1
opt_boot = np.argmin(mce_boot)+1
print 'optimal k in KFold: ' + str(opt_kf)
print 'optimal k in Bootstrapping: ' + str(opt_boot)

ga.plot_mce(k, mce_kf, '10-Fold')
ga.plot_mce(k, mce_boot, 'Bootstrapping')

plt.show()

# Print optimal tree
features = ['F1','F2']
clf_kf = tree.DecisionTreeClassifier(random_state= random_state, min_samples_leaf= opt_kf)
util.export_tree( clf_kf, features, filename="decision-tree-"+str(s)+".png" )
clf_boot = tree.DecisionTreeClassifier(random_state= random_state, min_samples_leaf= opt_boot)
util.export_tree( clf, features, filename="decision-tree-"+str(s)+".png" )




