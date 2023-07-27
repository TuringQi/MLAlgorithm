# coding=utf-8
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from Bagging import RFC
from DecisionTree import DecisionTreeC
from SVM import SVC_main
from Boosting import AdaBoostC
from Boosting import GradientBoostingC
from Boosting import XGBoostC
from NeutralNetwork import MLPC
from Utils import DataRead

# 模型名称
names = [
         'SVM',
         'DecisionTree',
         'RF',
         'AdaBoost',
         'GradientBoosting',
         'XGBoost',
         'MLP'
         ]

# 训练好的模型实例
sampling_methods = [
                    SVC_main.clf,
                    DecisionTreeC.clf,
                    RFC.clf,
                    AdaBoostC.clf,
                    GradientBoostingC.clf,
                    XGBoostC.clf,
                    MLPC.clf
                   ]

# 曲线的颜色 https://blog.csdn.net/weixin_52071682/article/details/113856233
colors = [
          'crimson',
          'orange',
          'lawngreen',
          'cyan',
          'darkorchid',
          'palegreen',
          'pink'
          ]


def multi_models_roc(names, sampling_methods, colors, X_test, y_test, save=True, dpin=100):
    """
    将多个机器模型的roc图输出到一张图上

    Args:
        names: list, 多个模型的名称
        sampling_methods: list, 多个模型的实例化对象
        save: 选择是否将结果保存（默认为png格式）

    Returns:
        返回图片对象plt
    """
    plt.figure(figsize=(20, 20), dpi=dpin)

    for (name, method, colorname) in zip(names, sampling_methods, colors):

        y_test_preds = method.predict(X_test)
        y_test_predprob = method.predict_proba(X_test)[: ,1]
        fpr, tpr, thresholds = roc_curve(y_test, y_test_predprob, pos_label=1)

        plt.plot(fpr, tpr, lw=5, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=colorname)
        plt.plot([0, 1], [0, 1], '--', lw=5, color = 'grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=30)
        plt.ylabel('True Positive Rate', fontsize=30)
        plt.tick_params(labelsize=20)
        plt.title('ROC Curve', fontsize=25)
        plt.legend(loc='lower right', fontsize=20)


    if save:
        plt.savefig('..//Outputs//ROC_Test_all.png')

    return plt


# 将读入的数据集划分，所有模型的随机数种子设置为42，以复现出相同的划分结果才可以比较
X_train, X_test, y_train, y_test = train_test_split(
    DataRead.X, DataRead.y, test_size=0.2, random_state=42
)

# ROC curves
test_roc_graph = multi_models_roc(names, sampling_methods, colors, X_test, y_test, save=False)  # 这里可以改成训练集
plt.show()
