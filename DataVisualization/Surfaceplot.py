# -*- coding: utf-8 -*-
# libraries

import matplotlib.pyplot as plt
import pandas as pd

# 如果用这个数据需要挂vpn
url = 'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/volcano.csv'
data = pd.read_csv(url)

# Transform it to a long format
df=data.unstack().reset_index()
df.columns=["X", "Y", "Z"]

# And transform the old column name in something numeric
df['X'] = pd.Categorical(df['X'])
df['X'] = df['X'].cat.codes
df = df.sample(frac=0.1, random_state=42)  # 随机下采样一下，红点会少一些，假设这些点是论文中的“随机搜索对应的位置”
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
# fig, ax = plt.subplots(figsize=(15, 15))
surf = ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.jet, linewidth=0.2, alpha=0.8)
fig.colorbar(surf, shrink=0.5, aspect=15)  # 颜色条

ax.scatter(df['Y'], df['X'], df['Z'], marker='*', c='r', s=40)
ax.w_xaxis.pane.set_edgecolor('gray')   # 图标的网格背景颜色
ax.w_yaxis.pane.set_edgecolor('gray')
ax.w_zaxis.pane.set_edgecolor('gray')
ax.view_init(30, 60)  # 旋转角度
plt.show()