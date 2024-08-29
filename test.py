import os
import numpy as np

def calculate_pearson(y_true, y_pred):
    corr_sum = 0
    for i in range(len(y_true)):
        # 检查预测值是否全部相同
        if np.all(y_pred[i] == y_pred[i][0]):
            corr_sum += -1.0
        else:
            corr_sum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    
    return corr_sum / len(y_true)

y_true = np.array([[1, 2, 3], [4, 5, 6]])
y_pred = np.array([[0,0,0], [0,0,0]])
print(calculate_pearson(y_true, y_pred))