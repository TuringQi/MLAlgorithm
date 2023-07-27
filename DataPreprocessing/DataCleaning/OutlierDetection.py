# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# 拉依达准则
def PauTa(columns_value):
    outlier = []  # 存放异常值的列表
    columns_value = list(map(float, columns_value))
    mean = np.mean(columns_value)  # 计算均值
    std_deviation = np.sqrt(np.var(columns_value))  # 计算标准差
    for i in range(len(columns_value)):  # 对列中所有数值使用拉依达准则判断是否异常
        if abs(columns_value[i] - mean) >= std_deviation * 3:
            outlier.append(columns_value[i])  # 如果异常，存入异常值列表
            columns_value[i] = 0  # 对异常值暂时赋0处理，等剔除相关特征和样本后再进行插值
    return columns_value, outlier

# 异常值检测函数
def outlierDetection(data,columns_names):
    # 异常值判断使用拉依达准则
    for columns_name in columns_names:  # 遍历每一列
        columns_value = data[columns_name].tolist()  # 转型
        columns_value, outlier = PauTa(columns_value) # 调用拉依达函数处理列表
        new_columns_value = pd.DataFrame({columns_name: columns_value})  # 更新DataFrame中的列值
        data.update(new_columns_value)
        print(columns_name + '列中的异常值：')
        print(outlier)
    data.to_excel("..//Outputs//OutlierDetection.xlsx", sheet_name='Sheet1', index=False)

data = pd.read_excel(
    io='..//Resources//Molecular_Descriptor.xlsx',
    sheet_name=0,
    header=0,  # 指定行索引为表头
    # skiprows=1,
    # skipcols=0
    #index_col=0
)
data = pd.DataFrame(data)
columns_names = data.columns.tolist()
outlierDetection(data, columns_names)



