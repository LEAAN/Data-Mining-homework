import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import graph as ga
import calcu as ca

# Initialization
label = [ True,  True,  True,  True,  True,  True,  True, False, False,
       False, False, False, False]

a_predict = [ True,  True, False, False,  True,  True, False, False,  True,
       False, False, False, False]

b_predict = [ True,  True,  True,  True, False,  True,  True, False,  True,
       False,  True, False, False]

c_value = [0.8,0.9,0.7,0.6,0.4,0.8,0.4,0.4,0.6,0.4,0.4,0.4,0.2]

# Lists that save values
predict = []
fpr = []
tpr = []
auc = []

# Initial Color values
colors = ['red','blue','orange','cyan','black','pink']

def renew(f,t,a):
	fpr.append(f)
	tpr.append(t)
	auc.append(a)

# Get a_info b_info
fpr_a, tpr_a, auc_a, thresholds_a = ca.roc(a_predict, label)
fpr_b, tpr_b, auc_b, thresholds_b= ca.roc(b_predict, label)
renew(fpr_a,tpr_a,auc_a)
renew(fpr_b,tpr_b,auc_b)

# Get c_infos
fpr_c, tpr_c, auc_c, thresholds_c = ca.roc(c_value, label)
c_label = []
for value in thresholds_c:
	c_label.append('C ' + str(value))

# Scatter series of c
for f,t, thres,color in zip(fpr_c,tpr_c,thresholds_c,colors):
	l = 'C: ' + str(thres)
	# plt.scatter(f,t,marker = '>', color = color, label = 'C: ' + str(thres)) 
	ga.scatter(f,t,l,'>',color)

# Scatter A B
def scatter_ab():
	dot_a = ga.scatter(fpr_a[1],tpr_a[1],'A', 'o', colors[0])
	dot_b = ga.scatter(fpr_b[1],tpr_b[1],'B', 'o', colors[1])

# Convex Hullfor A B C
points = np.array([fpr_c,tpr_c]).T
points = np.vstack((points,np.array([fpr_a[1],tpr_a[1]])))
points = np.vstack((points,np.array([fpr_b[1],tpr_b[1]])))
points = np.vstack((points,np.array([0,0])))
points = np.vstack((points,np.array([1,1])))
hull = ConvexHull(points)
for simplex in hull.simplices:
	plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

ga.show_roc()

# plot connecting line of C
# Scatter series of C
for f,t, thres,color in zip(fpr_c,tpr_c,thresholds_c,colors):
	l = 'C: ' + str(thres)
	# plt.scatter(f,t,marker = '>', color = color, label = 'C: ' + str(thres)) 
	ga.scatter(f,t,l,'>',color)

scatter_ab()
plt.plot(fpr_c,tpr_c)

ga.show_roc()

# For c_threshold == 0.4
p = ca.check_threshold(c_value,0.4)
f, t, a, thres = ca.roc(p,label)
renew(f,t,a)
dot_c = ga.scatter(f[1],t[1], 'C: 0.4', 'o',colors[3])
# Calculate cost line properties
# fpr X [0]
# tpr Y [1]
slope1 = 1.0/5.0
slope2 = 1.0
ga.cost_line(slope1, [fpr_b[1],tpr_b[1]])
ga.cost_line(slope2, [fpr_b[1],tpr_b[1]])

# y_intersect = tpr_b[1]-cost_slope*fpr_b[1]
# y_intersect1 =tpr_b[1] - cost_slope1 * fpr_b[1]
# # Draw cost line
# ga.cost_line([0,1],[y_intersect, y_intersect +cost_slope])
# ga.cost_linep([0,1],[y_intersect1, y_intersect1 +cost_slope1])

# legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=5,fontsize = 8, scatterpoints =1)
scatter_ab()
ga.show_roc()


# plt.title('ROC of A, B, and C with thresholds')

# plt.xlim((-0.2,1.2))
# plt.ylim((0,1.2))

# plt.legend(bbox_to_anchor=(1.05, 1), loc=5,fontsize = 8, scatterpoints =1)
# plt.title('ROC of A, B, and C with thresholds')

# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()








