#coding=utf-8
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from Utils import DataRead

# 超参
n_estimators = 100  # 基学习器数量
criterion = 'squared_error'
max_depth = None  # 决策树最大深度 默认None(不限制)，数据量大or特征多时推荐限制深度(一般10-100)
min_samples_split = 2  # 内部节点再划分所需最小样本数，默认值2，若值为浮点数表示相对于样本集的比例
max_features = 'sqrt'  # 在某结点位置考虑的最大特征数，默认值为总特征个数开平方取整
max_leaf_nodes = None  # 叶子结点的最大数量

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
clf = RandomForestRegressor(criterion=criterion,
                             n_estimators=n_estimators,
                             max_depth=max_depth,
                             min_samples_split=min_samples_split,
                             max_features=max_features,
                             max_leaf_nodes=max_leaf_nodes
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

print('RandomForest Regression_metrics:')
print(Regression_metrics)