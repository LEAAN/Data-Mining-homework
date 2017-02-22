import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# label = np.array(['+','+','+','+','+','+','+','-','-','-','-','-','-'])
# a = np.array(['+','+','-','-','+','+','-','-','+','-','-','-','-'])
# b = np.array(['+','+','+','+','-','+','+','-','+','-','+','-','-'])

# label = np.array([ True,  True,  True,  True,  True,  True,  True, False, False,
#        False, False, False, False], dtype=bool)

# a = np.array([ True,  True, False, False,  True,  True, False, False,  True,
#        False, False, False, False], dtype=bool)

# b = np.array([ True,  True,  True,  True, False,  True,  True, False,  True,
#        False,  True, False, False], dtype=bool)
# build dictionary for A B C
# def confusion_matrix (predict, true):
# 	for i in range (len(predict)):
# 		tp[i] = predict[i] == '+' & label[i] == '+'
# 		fn[i] = predict[i] == '-' & label[i] == '+'
# 		fp[i] = predict[i] == '+' & label[i] == '-'
# 		tn[i] = predict[i] == '-' & label[i] == '-'
# 	return tp, fn, fp, tn

label = [ True,  True,  True,  True,  True,  True,  True, False, False,
       False, False, False, False]

a_predict = [ True,  True, False, False,  True,  True, False, False,  True,
       False, False, False, False]

b_predict = [ True,  True,  True,  True, False,  True,  True, False,  True,
       False,  True, False, False]


fpr, tpr, thresholds = roc_curve(label, a_predict)
roc_auc = auc(fpr,tpr)

plt.title('ROC of classifiere A, B, and C')
plt.plot(fpr,tpr, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
# plt.plot([0,1],[0,1],'r--')
# plt.xlim([-0.1,1.2])
# plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# def confusion_matrix(predict, true):
# 	tp = []
# 	fn = []
# 	fp = []
# 	tn = []
# 	for i in range (len(true):
# 		tp.append(predict[i] == True and label[i] == True)
# 		# fn[i] = predict[i] == '-' & label[i] == '+'
# 		# fp[i] = predict[i] == '+' & label[i] == '-'
# 		# tn[i] = predict[i] == '-' & label[i] == '-'
# 	return tp, fn, fp, tn




# # build dictionary for A B C
# a_correctness = np.bitwise_and(a,label)
# a_tp = np.bitwise_and(a,a_correctness)
# a_fn = a_predict==F label==T

# i = 1
# label[i]==True and a_predict[i] == True



# a_info['TP']
# a_info = {'predict': a}
# a_info['correctness'] = (np.bitwise_and(a,label))
# array([ True,  True, False, False,  True,  True, False, False, False,
#        False, False, False, False], dtype=bool)

# unique, counts = np.unique(a_info['correctness'], return_counts=True)
# a_info['tp'] = counts[1]

# a_info['TP'] ,a_info['FN'], a_info['FP'], a_info['TN'] = confusion_matrix(a, label)





























