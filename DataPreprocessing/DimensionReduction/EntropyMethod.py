# -*- coding: utf-8 -*-
import math
import pandas as pd
import numpy as np
from math import ceil

# 默认剔除权重在后20%的特征
threshold = 0.2

# MaxMin-Normalization
def Standardization(data,cols1=None, cols2=None):
    """
    :param cols1: 正向指标列名列表
    :param cols2: 负向指标列名列表
    """
    if cols1 == None and cols2 == None:
        return data
    elif cols1 != None and cols2 == None:
        return (data[cols1] - data[cols1].min())/(data[cols1].max()-data[cols1].min())
    elif cols1 == None and cols2 != None:
        return (data[cols2].max - data[cols2])/(data[cols2].max()-data[cols2].min())
    else:
        a = (data[cols1] - data[cols1].min())/(data[cols1].max()-data[cols1].min())
        b = (data[cols2].max() - data[cols2])/(data[cols2].max()-data[cols2].min())
        return pd.concat([a, b], axis=1)

# 计算特征权重
def FeatureWeightFun(data):
    K = 1/np.log(len(data))
    e = -K*np.sum(data*np.log(data))
    d = 1-e
    w = d/d.sum()
    return w

# data = pd.DataFrame({'指标1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                     '指标2': [2, 4, 6, 8, 10, 2, 4, 6, 8, 10],
#                     '指标3': [1, 2, 1, 3, 2, 1, 3, 2, 3, 1],
#                     '指标4': [3, 1, 2, 3, 5, 8, 7, 8, 8, 9]
#                    })

data = pd.read_excel(
    io='..//..//Resources//Molecular_Descriptor_Feature_sampleCleaning.xlsx',
    sheet_name=0,
    header=0,  # 指定行索引为表头
    # skiprows=1,
    # skipcols=0
    #index_col=0
)
data = pd.DataFrame(data)
# data = data.iloc[:, 1:]  # 第一列是样本名，根据需要可裁掉

# 指定正向特征和反向特征
Forward_feature = data.columns.tolist()
Reverse_feature = []
# 归一化
SD_data = Standardization(data, Forward_feature, Reverse_feature)
# 根据熵值法计算特征权重
Feature_weight = FeatureWeightFun(SD_data)
# 各样本综合得分以及样本内各特征的得分
# Sample_score = SD_data * Feature_weight
# Sample_score['综合指标'] = Sample_score.sum(axis=1)
# data.to_excel("..//Outputs//EntropyMethod_Sample_score.xlsx", sheet_name='Sheet1', index=False)

# 将特征权重排序 小->大
Feature_weight = Feature_weight.sort_values()
# 转型 Series->dataframe
Feature_weight = pd.DataFrame(Feature_weight, columns=['权重'])
# 存储所有特征的权重
Feature_weight.to_excel("..//..//Outputs//EntropyMethod_Feature_weight.xlsx", sheet_name='Sheet1', index=True)
# 剔除权重最小的 threshold% 的特征，并存储
delete = []
Feature_names = Feature_weight.index.tolist()
for i in range(math.ceil(len(Feature_weight) * threshold)):
    delete.append(Feature_names[i])
    data.drop(columns=Feature_names[i], axis=1, inplace=True)
print('剔除的特征名称:')
print(delete)
data.to_excel("..//..//Outputs//EntropyMethod.xlsx", sheet_name='Sheet1', index=False)





