import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize

diabetes = pd.read_csv('../result/simplenet_test_0.853.csv')
n_class = 10
list1 = diabetes['label_gt']
list2 = np.array(diabetes[['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9']])

y_one_hot = label_binarize(y=list1, classes=np.arange(n_class))
auc = metrics.roc_auc_score(y_one_hot.ravel(), list2.ravel())
fpr, tpr, thersholds = metrics.roc_curve(y_one_hot.ravel(), list2.ravel())

plt.plot(fpr, tpr, '-', label='simplenet (AUC = {0:.3f})'.format(auc), lw=1)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right", prop={'size': 8})
plt.show()
