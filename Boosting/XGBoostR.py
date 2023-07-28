#coding=utf-8
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor

from Utils import DataRead

# XGB架构超参
objective = 'reg:linear'  # reg:linear线性回归的损失函数,'reg:squaredlogerror'以均方对数损失为损失函数的回归模型。指定学习任务和相应的学习目标或自定义目标函数使用
booster = 'gbtree'  # 基学习器默认决策树 ，'dart'采用dropout的决策树
n_estimator = 150  # 迭代次数默认100，取决于数据集的大小
learning_rate = 0.1  # 弱学习器的权重缩减系数 [0.01,0.2]，较小的值可以使模型更加稳定，但需要更多的弱学习器
# early_stopping_rounds = 10  # 在验证集上当连续n次迭代分数没有提高后提前终止训练,防止过拟合
# 弱学习器超参
max_depth = 6  # 基决策树深度，较大的值可以提高模型的复杂性，但也容易导致过拟合。通常推荐的值为3到10之间
subsample = 0.8  # 样本集采样比例，默认不采样:1 ，通常推荐的值为0.5到1之间
colsample_bytree = 1  # 使用的特征占全部特征的比例，默认1 [0.5,1]
eta = 0.1  # 学习率，控制每次迭代更新权重时的步长 默认0.2  推荐[0.01,0.2]
gamma = 0  # 弱学习器的分裂所需的最小损失减少，防止过拟合 [0,5]

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
clf = XGBRegressor(booster=booster,
                    objective=objective,
                    n_estimator=n_estimator,
                    learning_rate=learning_rate,
                    # early_stopping_rounds=early_stopping_rounds,
                    max_depth=max_depth,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    eta=eta,
                    gamma=gamma
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

print('XGBoost Regression_metrics:')
print(Regression_metrics)

