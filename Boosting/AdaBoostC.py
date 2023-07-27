# coding=utf-8
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from Utils import DataRead
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 超参
base_estimator = None  # 基学习器默认DecisionTreeClassifier
# base_estimator = DecisionTreeClassifier(criterion='gini',
#                                         splitter='best',
#                                         max_depth=None,
#                                         min_samples_split=2,
#                                         max_features=None,
#                                         max_leaf_nodes=None
#                                         )
algorithm = 'SAMME.R'  # 使用SAMME.R时基分类学习器参数base_estimator必须限制使用支持概率预测的分类器，也就是在scikit-learn中基分类学习器对应的预测方法除了predict还需要有predict_proba
n_estimators = 100  # 最大迭代次数，默认50
learning_rate = 1.0  # 每个基学习器的权重缩减系数：[0,1]。如果过大，容易错过最优值，如果过小，则收敛速度会很慢。当分类器迭代次数较少时，学习率可以小一些，当迭代次数较多时，学习率可以适当放大。


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
clf = AdaBoostClassifier(base_estimator=base_estimator,
                         algorithm=algorithm,
                         n_estimators=n_estimators,
                         learning_rate=learning_rate
                         )
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
Accuracy_score = accuracy_score(y_test, predictions)
print('AdaBoost Accuracy_score:', Accuracy_score)