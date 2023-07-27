#coding=utf-8
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, plot_roc_curve
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from Utils import DataRead
import numpy as np
import pandas as pd


# parameters = { 'C':np.arange(x_min, x_max, h)}

# 超参
C = 1
kernel = 'rbf'
gamma = 0.5

# 导入数据
X = DataRead.X
y = DataRead.y

# 标准归一化
X = StandardScaler().fit_transform(X)
# 随机划分样本数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 训练 SVM
clf = SVC(C=C, kernel=kernel, gamma=gamma, shrinking=True, probability=True)
clf.fit(X_train, y_train)
# 预测测试集
predictions = clf.predict(X_test)
# 计算准确度
Accuracy_score = accuracy_score(y_test,predictions)
print('SVM Accuracy_score:', Accuracy_score)
# 绘制ROC曲线
'''fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
display.plot()
plt.show()'''
# 以下绘制ROC曲线的函数在新版本已被弃用，不建议使用
# plot_roc_curve(clf, X_test, y_test)
# plt.show()