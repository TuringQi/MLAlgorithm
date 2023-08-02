# coding:utf-8
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 特征缺失值热图
def miss_heatamp(data, title):
    cols = data.columns
    colours = ['#006699', '#ffff99']  ## 第一项没缺失的颜色，第二项，缺失的颜色

    plt.figure(figsize=(12, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决无法显示符号的问题
    sns.set(font='SimHei', font_scale=1)
    f = sns.heatmap(data[cols].isnull(), cmap=sns.color_palette(colours))
    f.set_title(title)
    plt.show()


data = pd.read_excel(
    io='..//Resources//identity_new1.xlsx',
    sheet_name=1,
    header=0,  # 指定行索引为表头
    # skiprows=1,
    # skipcols=0
    #index_col=0
)
data = pd.DataFrame(data)
miss_heatamp(data,'identity_new1_heatmap')