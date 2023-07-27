# -*- coding: utf-8 -*-
import xlsxwriter as xw
import numpy as np
import pandas as pd

X = pd.read_excel(
    io='identity_new1.xlsx',
    sheet_name=1,
    header=0,  # 指定行索引为表头
    # skiprows=1,
    # skipcols=0
    #index_col=0
)
X = pd.DataFrame(X) # 转换数据类型object——>DateFrame
print(X)
new_col = [8, 9, 10] # 要指定修改的那一列的新值
new_X = pd.DataFrame({'年龄': new_col}) # 参数是一个字典，键是待更新表头的名字，值是一个有新内容的列表
X.update(new_X)
print(X)
X.to_excel("identity_new1.xlsx", sheet_name='Sheet2', index=False)
# with pd.ExcelWriter("identity_new1.xlsx") as writer:
#     X.to_excel(writer, sheet_name="Sheet2")

# writer = pd.ExcelWriter('identity_new1.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace')
# X.to_excel(writer, sheet_name='Sheet2', index=False)