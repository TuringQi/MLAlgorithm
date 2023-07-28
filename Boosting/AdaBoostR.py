#coding=utf-8
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from Utils import DataRead

# 超参
base_estimator = None  # 基学习器默认DecisionTreeClassifier
# base_estimator = DecisionTreeClassifier(criterion='gini',
#                                         splitter='best',
#                                         max_depth=None,
#                                         min_samples_split=2,
#                                         max_features=None,
#                                         max_leaf_nodes=None
#                                         )
n_estimators = 100  # 终止提升的估计器的最大数量，默认=50。如果完美契合学，习过程就会提前停止
loss = 'square'  # 默认线性损失'linear',这里默认异常值已经被处理好所以使用平方损失
learning_rate = 1.0  # 每个基学习器的权重缩减系数：[0,1]。如果过大，容易错过最优值，如果过小，则收敛速度会很慢。当回归器迭代次数较少时，学习率可以小一些，当迭代次数较多时，学习率可以适当放大。

Regression_metrics = {}

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
clf = AdaBoostRegressor(base_estimator=base_estimator,
                         loss=loss,
                         n_estimators=n_estimators,
                         learning_rate=learning_rate
                         )
clf.fit(X_train, y_train)
# 预测测试集
predictions = clf.predict(X_test)
# 计算回归指标
Regression_metrics['MAE'] = round(mean_absolute_error(y_test, predictions), 3)
Regression_metrics['MAPE'] = round(mean_absolute_percentage_error(y_test, predictions), 3)
Regression_metrics['MSE'] = round(mean_squared_error(y_test, predictions), 3)
Regression_metrics['RMSE'] = round(np.sqrt(Regression_metrics['MSE']).real, 3)
Regression_metrics['R2'] = round(r2_score(y_test, predictions), 3)

print('AdaBoost Regression_metrics:')
print(Regression_metrics)