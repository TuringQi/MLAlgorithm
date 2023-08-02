# coding:utf-8
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 特征缺失值直方图
def miss_bar(data, title):
    '''
    data : dataframe格式的数据，行为数据，列为特征
    title : 图表名字
    '''
    missValue2miss_num = {}
    for col in data.columns:
        missing = data[col].isnull()
        num_missing = np.sum(missing)
        if num_missing > 0:
            missValue2miss_num[col] = num_missing
    df = pd.DataFrame([missValue2miss_num])
    df.index = ['miss_num']
    df = df.T

    plt.figure(figsize=(10, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决无法显示符号的问题
    sns.set(font='SimHei', font_scale=1)

    f = sns.barplot(x=df.index, y=df['miss_num'], color='#336699')
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
miss_bar(data,'identity_new1_bar')