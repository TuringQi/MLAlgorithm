# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# 剔除空缺值和异常值超过阈值的特征
def FeatureCleaning(data,columns_names,threshold):
    cleaned = {}
    for columns_name in columns_names:  # 遍历每一列
        columns_value = data[columns_name].tolist()  # 转型
        count = 0
        for i in range(len(columns_value)):
            if(columns_value[i] == 0):
                count = count + 1
        if(count/len(columns_value) >= threshold):
            data.drop(columns=columns_name, axis=1, inplace=True)
            cleaned[columns_name] = round(count/len(columns_value), 2)
    return data, cleaned

# 剔除空缺值和异常值超过阈值的样本
def SampleCleaning(data,threshold):
    cleaned = {}
    for j in range(len(data)):  # 遍历每一行
        row_value = data.loc[j].tolist()  # 转型
        count = 0
        for i in range(len(row_value)):
            if(row_value[i] == 0):
                count = count + 1
        if(count/len(row_value) >= threshold):
            data.drop(index=j, axis=0, inplace=True)
            cleaned[j+1] = round(count/len(row_value), 2)
    return data, cleaned



data = pd.read_excel(
    io='..//..//Resources//Molecular_Descriptor_OutlierDetection.xlsx',
    sheet_name=0,
    header=0,  # 指定行索引为表头
    # skiprows=1,
    # skipcols=0
    #index_col=0
)
data = pd.DataFrame(data)
data = data.iloc[:, 1:]
columns_names = data.columns.tolist()
data, cleaned = FeatureCleaning(data, columns_names, 0.5)
print('去除的特征名称及其坏值率：')
print(cleaned)
data, cleaned =SampleCleaning(data, 0.5)
print('去除的样本编号及其坏值率：')
print(cleaned)
data.to_excel("..//..//Outputs//Molecular_Descriptor_Feature_sampleCleaning.xlsx", sheet_name='Sheet1', index=False)