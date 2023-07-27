# -*- coding: utf-8 -*-
import pandas as pd
from scipy.interpolate import lagrange


# s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
def ployinterp_column(s, n, k=5):
    if n >= k:
        if n + 1 + k < len(s):
            y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]
        else:
            y = s[list(range(n - k, n)) + list(range(n + 1, len(s)))]
    else:
        if n + 1 + k < len(s):
            y = s[list(range(0, n)) + list(range(n + 1, n + 1 + k))]
        else:
            y = s[list(range(0, n)) + list(range(n + 1, len(s)))]
    y = del_zero(y)
    return lagrange(y.index, list(y))(n)
    # y = y[y.notnull()]  # 剔除空值
    # print(y.index)
    # return lagrange(y.index, list(y))(n)

def del_zero(y):
    index = y.index
    for n in index:
        if y[n] == 0:
            del y[n]
    return y

data = pd.read_excel(
    io='..//..//Resources//identity_new1.xlsx',
    sheet_name=0,
    header=0,  # 指定行索引为表头
    # skiprows=1,
    # skipcols=0
    # index_col=0
)
data = pd.DataFrame(data)

# -------------------
# data = data.iloc[:, 1:]
# -------------------

for i in data.columns:
    for j in range(len(data)):
        if (data[i])[j] == 0:  # 如果为0即插值    检测空值代码：if (data[i].isnull())[j]:
            data.loc[j][i] = ployinterp_column(data[i], j)
print(data)
data.to_excel("..//..//Outputs//Lag.xlsx", sheet_name='Sheet1', index=False)
