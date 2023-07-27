# coding=utf-8
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from Utils import DataRead
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 超参
loss = 'deviance'  # 损失函数默认对数似然损失
n_estimators = 150  # 迭代次数默认100
learning_rate = 1  # 基学习器的权重缩减系数默认1
subsample = 0.8  # 样本集采样比例，默认不采样:1

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
clf = GradientBoostingClassifier(loss=loss,
                                 n_estimators=n_estimators,
                                 learning_rate=learning_rate,
                                 subsample=subsample
                                )
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
Accuracy_score = accuracy_score(y_test, predictions)
print('GradientBoosting Accuracy_score:', Accuracy_score)