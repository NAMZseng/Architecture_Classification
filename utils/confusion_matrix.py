import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

diabetes = pd.read_csv('../result/simplenet_test_0.853.csv')

fact = diabetes['label_gt']
guess = diabetes['label-pre']

print("每个类别的精确率和召回率：\n", classification_report(y_true=fact, y_pred=guess))

classes = list(set(fact))
classes.sort()
confusion = confusion_matrix(y_true=fact, y_pred=guess)
plt.imshow(confusion, cmap=plt.cm.Blues)
indices = range(len(confusion))
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
        plt.text(second_index, first_index, confusion[first_index][second_index], va='center', ha='center')
plt.show()
