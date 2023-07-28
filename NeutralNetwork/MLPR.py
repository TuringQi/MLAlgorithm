#coding=utf-8
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from Utils import DataRead

# 超参
hidden_layer_sizes = (20, 10, 4)  # 隐藏层结构
activation = 'relu'  # 激活函数选择 默认relu
solver = 'adam'  # 优化器 默认自适应梯度下降
# 以下三个参数在solver = adam'时生效
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8
# 以下两个参数在solver = 'sgd'时生效
momentum = 0.9
Nesterovs_momentum = True  # 优化器选择SGD时，默认使用Nesterovs动量梯度下降

random_state = None  #
alpha = 0.0001  # L2正则项系数
batch_size = 'auto'
Learning_rate = 'constant'  # 学习率是否需要主动变化
power_t = 0.5  # 逆缩放学习率的指数，当learning_rate=‘invscaling’时生效
Learning_rate_init = 0.001  # 初始学习率 仅当solver='sgd'或'adam'时使用
max_iter = 200  # 最大迭代次数
shuffle = True  # 洗牌数据

Early_stopping = False  # 是否早停，取值True时下面三个参数生效
validation_fraction = 0.1  # 留作早期停止验证集的训练数据比例
n_iter_no_change = 10  # 仅当solver='sgd'或'adam'时有效
tol = 1e-4  # 优化误差，当损失值的变化连续在n_iter_no_change轮迭代中小于tol时，认为已收敛，停止训练（learning_rate='adaptive'的情况除外）
verbose = True  # 打印进度信息

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
clf = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    solver=solver,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    epsilon=epsilon,
                    momentum=momentum,
                    nesterovs_momentum=Nesterovs_momentum,
                    random_state=random_state,
                    alpha=alpha,
                    batch_size=batch_size,
                    learning_rate=Learning_rate,
                    power_t=power_t,
                    learning_rate_init=Learning_rate_init,
                    max_iter=max_iter,
                    shuffle=shuffle,
                    early_stopping=Early_stopping,
                    validation_fraction=validation_fraction,
                    n_iter_no_change=n_iter_no_change,
                    tol=tol,
                    verbose=verbose)
clf.fit(X_train, y_train)
# 预测测试集
predictions = clf.predict(X_test)
# 计算回归指标
Regression_metrics['MAE'] = round(mean_absolute_error(y_test, predictions), 3)
Regression_metrics['MAPE'] = round(mean_absolute_percentage_error(y_test, predictions), 3)
Regression_metrics['MSE'] = round(mean_squared_error(y_test, predictions), 3)
Regression_metrics['RMSE'] = round(np.sqrt(Regression_metrics['MSE']).real, 3)
Regression_metrics['R2'] = round(r2_score(y_test, predictions), 3)

print('MLP Regression_metrics:')
print(Regression_metrics)