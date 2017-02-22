import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

label = [ True,  True,  True,  True,  True,  True,  True, False, False,
       False, False, False, False]

a_predict = [ True,  True, False, False,  True,  True, False, False,  True,
       False, False, False, False]

b_predict = [ True,  True,  True,  True, False,  True,  True, False,  True,
       False,  True, False, False]

c_value = [0.8,0.9,0.7,0.6,0.4,0.8,0.4,0.4,0.6,0.4,0.4,0.4,0.2]
# Here Zitong randomly setted some threshold values
c_threshold = [0.5, 0.6, 0.3]

# initialize c_info
c_predict = []
fpr_c = []
tpr_c = []
auc_c = []
# Pos & Neg values
pos = label.count(True)
neg = label.count(False)

def check_threshold(value, threshold):
	# Returns a list 
	predict = []
	# predict = np.zeros((len(value)),dtype = bool)
	for i in range(0,len(value)):
		predict.append(value[i]> threshold)
	return predict

def roc(predict,true):
	fpr, tpr, thresholds = roc_curve(true, predict)
	# roc_auc =  auc(fpr,tpr)
	roc_auc = 1
	return fpr, tpr,roc_auc
# Get a_info b_info
fpr_a, tpr_a, auc_a = roc(a_predict, label)
fpr_b, tpr_b, auc_b = roc(b_predict, label)

# Get c_info
for thres in c_threshold:
	p = check_threshold(c_value,thres)
	c_predict.append(p)
	fpr, tpr, auc = roc(p,label)
	fpr_c.append(fpr)
	tpr_c.append(tpr)
	auc_c.append(auc)
# Initial Color values
colors = ['red','blue','yellow','cyan','black']
# Draw scatters of A B C
dot_a = plt.scatter(fpr_a[1],tpr_a[1],label = 'A',color = colors[0])
dot_b = plt.scatter(fpr_b[1],tpr_b[1],label = 'B', color = colors[1])
for i in range(0,len(c_threshold)):
	plt.scatter(fpr_c[i][1],tpr_c[i][1],color = colors[i+2], label = 'C_'+ str(i) +' threshold value: ' + str(c_threshold[i]))

# Draw cost line --- How can I make the line full screen?
cost_slope = neg * 1.0 /(pos * 5.0)
line_cost = plt.plot([0,1],[0, cost_slope])

# Draw Convex Hull
# WELL I SHOULD HAVE APPEND ALL CLASSIFIER INFO IN A SINGLE LIST ...
# NOW MY CODE LOOKS REALLY UGLY
# WILL MAKE CHANGE TOMORROW
# convex_x = [fpr_a[1],fpr_c[0][0],fpr_b[1],fpr_c[2][0],fpr_c[1][0]]
# convex_y = [tpr_a[1],tpr_c[0][0],tpr_b[1],tpr_c[2][0],tpr_c[1][0]]
# convex = plt.plot(convex_x,convex_y)

legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=5,fontsize = 8, scatterpoints =1)
plt.title('ROC of A, B, and C with thresholds')

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()








