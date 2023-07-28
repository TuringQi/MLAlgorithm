# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from SVM import SVR
from DecisionTree import DecisionTreeR as DTR

# 把所有回归模型的指标值汇总起来
value = [[SVR.Regression_metrics['MAE'], SVR.Regression_metrics['MAPE'], SVR.Regression_metrics['MSE'],
          SVR.Regression_metrics['RMSE'], SVR.Regression_metrics['R2']],
         [DTR.Regression_metrics['MAE'], DTR.Regression_metrics['MAPE'], DTR.Regression_metrics['MSE'],
          DTR.Regression_metrics['RMSE'], DTR.Regression_metrics['R2']]
         ]
# 转型DataFrame
data = pd.DataFrame(value, index=['SVR', 'DecisionTree'], columns=['MAE', 'MAPE', 'MSE', 'RMSE', 'R2'], dtype=float)
# 设置条形宽度、颜色
bar_width = 0.2
colors = ['#499c9f', '#c76813', '#00FA9A', '#8B4513', '#FF8C00']
index_names = data.index.tolist()
columns_names = data.columns.tolist()
# 循环绘制每一种模型的指标数据条形图
for j in range(len(data)):
    if j == 0:
        plt.bar(np.arange(len(columns_names)), height=data.iloc[j].values, width=bar_width, color=colors[j], label=index_names[j])
    else:
        plt.bar(np.arange(len(columns_names)) + j * bar_width, height=data.iloc[j].values, width=bar_width,
                color=colors[j], label=index_names[j])

plt.legend()  # 图例
plt.xticks(np.arange(len(columns_names)) + bar_width*(len(data)-1)/2, columns_names)  # 标签+位置
# plt.ylabel('')  # 纵坐标轴标题
plt.title('不同指标下模型评估对比')
plt.rcParams['font.sans-serif']=['SimHei']  # 解决标题中文乱码
plt.rcParams['axes.unicode_minus'] = False
plt.show()
