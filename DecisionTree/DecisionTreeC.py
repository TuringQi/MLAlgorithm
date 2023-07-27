# coding=utf-8
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from Utils import DataRead
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 超参
criterion = 'gini'  # CART：gini ID3：entropy
splitter = 'best'  # 某一特征的值集合的划分标准，样本量大的时候锐推荐选择random
max_depth = None  # 决策树最大深度 默认None(不限制)，数据量大or特征多时推荐限制深度(一般10-100)
min_samples_split = 2  # 内部节点再划分所需最小样本数，默认值2，若值为浮点数表示相对于样本集的比例
max_features = None  # 在某结点位置考虑的最大特征数，一般小与50的特征可以全部考虑，否则随机采样n个特征
max_leaf_nodes = None  # 最大叶子结点数

# 导入数据
X = DataRead.X
y = DataRead.y

# 标准归一化
X = StandardScaler().fit_transform(X)

# 随机划分样本数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练分类决策树
clf = DecisionTreeClassifier(criterion=criterion,
                             splitter=splitter,
                             max_depth=max_depth,
                             min_samples_split=min_samples_split,
                             max_features=max_features,
                             max_leaf_nodes=max_leaf_nodes
                             )
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
Accuracy_score = accuracy_score(y_test, predictions)
print('DecisionTree Accuracy_score:', Accuracy_score)