# -*- coding: utf-8 -*-
'''
    得用matplotlib3.7以上
'''
import random
from matplotlib import pyplot as plt
import pandas as pd

data = pd.read_excel(
    io='Resource//identity_new1.xlsx',
    sheet_name=3,
    header=0,  # 指定行索引为表头
    # skiprows=1,
    # skipcols=0
    index_col=0
)
data = pd.DataFrame(data)
array = data.values
# 定义热图的横纵坐标
xLabel = data.columns.tolist()
yLabel = data.index.tolist()

fig, ax = plt.subplots(figsize=(15, 15))

ax.set_yticks(range(len(yLabel)))
ax.set_yticklabels(yLabel)
# plt.yticks(rotation=45)
ax.set_xticks(range(len(xLabel)))
ax.set_xticklabels(xLabel)
plt.xticks(rotation=90)
# 作图并选择热图的颜色填充风格，这里选择hot
im = ax.imshow(data, cmap='Blues')
# 为每个小方块标注对应值
for i in range(len(yLabel)):
    for j in range(len(xLabel)):
        plt.annotate(str(round(array[i, j],2)), xy=(j, i), ha='center', va='center')
# 增加右侧的颜色刻度条
colorbar = plt.colorbar(im, shrink=0.7)
colorbar.set_ticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75])
colorbar.set_ticklabels(['-0.75', '-0.5', '-0.25', '0', '0.25', '0.5', '0.75'])
# 增加标题
plt.title("随机森林前30个特征之间的相关性")
plt.rcParams['font.sans-serif']=['SimHei']
# show
plt.show()

fig, ax = plt.subplots()
