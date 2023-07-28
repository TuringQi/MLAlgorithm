#coding=utf-8
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from Utils import DataRead

# 超参
loss = 'squared_error'  # 平方损失
n_estimators = 150  # 迭代次数默认100
learning_rate = 0.1  # 基学习器的权重缩减系数默认0.1
subsample = 0.8  # 样本集采样比例，默认不采样:1，减少过拟合

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
clf = GradientBoostingRegressor(loss=loss,
                                 n_estimators=n_estimators,
                                 learning_rate=learning_rate,
                                 subsample=subsample
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

print('GradientBoosting Regression_metrics:')
print(Regression_metrics)