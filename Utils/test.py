# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

# data = pd.read_excel(
#     io='..//Resources//identity_new1.xlsx',
#     sheet_name=0,
#     header=0,  # 指定行索引为表头
#     # skiprows=1,
#     # skipcols=0
#     # index_col=0
# )
# data = pd.DataFrame(data)
#
# # -------------------
# # data = data.iloc[:, 1:]
# # -------------------
#
# s = data['姓名']
# del s[1]
# print(s[2])

matrix = np.matrix([[1, 4, 9, 8, 13, 16, 31, 40, 51], [7, 6, 9, 11, 12, 15, 19, 20, 31]])
# print(matrix)
matrix[0, 2:4], matrix[0, 4:9] = matrix[0, 7:9], matrix[0, 2:7]
print(matrix)
