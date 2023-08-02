# coding:utf-8
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# 对缺失值随机森林插值
def miss_rf_impute(data):
    '''
    data:dataframe格式
    '''

    copy_data = data.copy()
    miss_columns = copy_data.isnull().sum()[copy_data.isnull().sum() != 0].sort_values().index.tolist()
    unmiss_columns = copy_data.isnull().sum()[copy_data.isnull().sum() == 0].sort_values().index.tolist()
    for col in miss_columns:
        X_train = copy_data[copy_data[col].notnull()][unmiss_columns].values
        Y_train = copy_data[copy_data[col].notnull()][col].values
        X_test = copy_data[copy_data[col].isnull()][unmiss_columns][unmiss_columns].values
        rfr=RandomForestRegressor()
        rfr.fit(X_train,Y_train)
        predict_value = rfr.predict(X_test)
        copy_data.loc[(copy_data[col].isnull()),col] = predict_value
        unmiss_columns.append(col)
    return copy_data

data = pd.read_excel(
    io='..//..//Resources//identity_new1.xlsx',
    sheet_name=1,
    header=0,  # 指定行索引为表头
    # skiprows=1,
    # skipcols=0
    #index_col=0
)
data = pd.DataFrame(data)
data = miss_rf_impute(data)
print(data)