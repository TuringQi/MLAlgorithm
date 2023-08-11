# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon


data = pd.read_excel(
    io='Resource//identity_new1.xlsx',
    sheet_name=2,
    header=0,  # 指定行索引为表头
    # skiprows=1,
    # skipcols=0
    #index_col=0
)
data = pd.DataFrame(data)

fig, axs = plt.subplots(2, 3)

# 基本箱线图
axs[0, 0].boxplot(data)
axs[0, 0].set_title('basic plot')
# 在绘图中添加水平网格
axs[0, 0].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

# 凹口箱线图
axs[0, 1].boxplot(data, 1)
axs[0, 1].set_title('notched plot')

# 更改异常点符号
axs[0, 2].boxplot(data, 0, 'gD')
axs[0, 2].set_title('change outlier\npoint symbols')

# 不显示异常点
axs[1, 0].boxplot(data, 0, '')
axs[1, 0].set_title("don't show\noutlier points")

# 水平箱线图
axs[1, 1].boxplot(data, 0, 'rs', 0)
axs[1, 1].set_title('horizontal boxes')

# 修改上下须与上下四分位的距离，1.5倍的四分位差
axs[1, 2].boxplot(data, 0, 'rs', 0, 0.75)
axs[1, 2].set_title('change whisker length')

fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.4, wspace=0.3)

# # 将箱线图画到一个axes里，如果值域相差很大的话效果不好
# fig, ax = plt.subplots()
# ax.boxplot([data['maxHBd'], data['BCUTw-1h']])
# ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
#               alpha=0.5)

plt.show()


