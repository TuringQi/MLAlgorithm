# coding=utf-8
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from Utils import DataRead
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# XGB架构超参
objective = 'binary:logistic'  # 模型认为类型，默认二分类，返回概率
booster = 'gbtree'  # 基学习器默认决策树 ，'dart'采用dropout的决策树
n_estimator = 150  # 迭代次数默认100
# early_stopping_rounds = 10  # 在验证集上当连续n次迭代分数没有提高后提前终止训练,防止过拟合
# 弱学习器超参
num_class = 1  # 样本类别数
max_depth = 6  # 基决策树深度
subsample = 0.8  # 样本集采样比例，默认不采样:1
colsample_bytree = 1  # 使用的特征占全部特征的比例，默认1
eta = 0.1  # 学习率，控制每次迭代更新权重时的步长 默认0.2  推荐[0.01,0.2]

# 导入数据
X = DataRead.X
y = DataRead.y

# 标准归一化
X = StandardScaler().fit_transform(X)

# 随机划分样本数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练
clf = XGBClassifier(booster=booster,
                    objective=objective,
                    n_estimator=n_estimator,
                    # early_stopping_rounds=early_stopping_rounds,
                    num_class=num_class,
                    max_depth=max_depth,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    eta=eta)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
Accuracy_score = accuracy_score(y_test, predictions)
print('XGBoost Accuracy_score:', Accuracy_score)