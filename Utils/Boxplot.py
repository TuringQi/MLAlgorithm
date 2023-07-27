# -*- coding: utf-8 -*-
"""
Grouped boxplots
================
"""
import seaborn as sns
import pandas as pd
import matplotlib as plt

sns.set(style="ticks", palette="pastel")

# Load the example tips dataset
data = pd.read_excel(
    io='identity_new1.xlsx',
    sheet_name=0,
    header=0,  # 指定行索引为表头
    # skiprows=1,
    # skipcols=0
    #index_col=0
)
data = pd.DataFrame(data)

# Draw a nested boxplot to show bills by day and time
# ————————————————————————OK————————————————————————
# boxplot = sns.boxplot(x=None, y='naAromAtom',
#             data=data
#             )
# boxplot = sns.boxplot(x=None, y=data['maxHBd'],
#                       width=.5
#             )
boxplot = sns.boxplot(x=None, y=None,
                      data=data,
                      width=.5
            )

# 保存箱线图
fig = boxplot.get_figure()
fig.savefig("Boxplot.png")