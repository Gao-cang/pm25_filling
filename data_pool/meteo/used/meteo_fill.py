import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb

def fill_missing_temperature(data_with_nan):
    # 分离出有温度数据和无温度数据的行
    data_not_null = data_with_nan.dropna(subset=['pressure'])
    data_is_null = data_with_nan[data_with_nan['pressure'].isnull()]

    # 准备特征X和目标y
    X = data_not_null[['year','month','day','hour', 'temperature']]
    y = data_not_null['pressure']

    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建随机森林回归器
    rf = RandomForestRegressor(n_estimators=200, random_state=42)

    # 训练模型
    rf.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = rf.predict(X_test)

    # 评估模型性能
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared on the test set for pressure: {r2}")

    # 预测缺失值
    predicted_temperatures = rf.predict(data_is_null[['year','month','day','hour', 'temperature']])

    # 将预测结果填充回原数据
    data_with_nan.loc[data_with_nan['pressure'].isnull(), 'pressure'] = predicted_temperatures

    return data_with_nan


# data_with_nan = pd.read_csv("../../data_pool/meteo/meteo_with_nan.csv", index_col=False)
# print(data_with_nan)

# 假设你已经加载了data_with_nan DataFrame
# temperature_filled_data = fill_missing_temperature(data_with_nan)

# temperature_filled_data.to_csv("../../data_pool/meteo/meteo_fill_temp.csv", index=False)

def fill_missing_col(data_with_nan, known_cols, unknown_col):
    # 分离出有温度数据和无温度数据的行
    data_not_null = data_with_nan.dropna(subset=[unknown_col])
    data_is_null = data_with_nan[data_with_nan[unknown_col].isnull()]

    # 准备特征X和目标y
    X = data_not_null[known_cols]
    y = data_not_null[unknown_col]

    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 创建随机森林回归器

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 定义参数
    params = {
        'objective': 'reg:squarederror',  # 回归问题
        'eval_metric': 'rmse',  # 评估指标
        'num_parallel_tree': 5,  # 设置树的数量以匹配目标数量
    }

    # 训练模型
    num_rounds = 500
    bst = xgb.train(params, dtrain, num_rounds)

    # 在测试集上进行预测
    y_pred = bst.predict(dtest)
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared on the test set for {unknown_col}: {r2}")

    # 预测缺失值
    dunknow = xgb.DMatrix(data_is_null[known_cols])
    predicted_currant_col = bst.predict(dunknow)
    # print(type(predicted_currant_col))
    # 将预测结果填充回原数据
    data_with_nan.loc[data_with_nan[unknown_col].isnull(), unknown_col] = predicted_currant_col

    return data_with_nan

temperature_filled_data = pd.read_csv("meteo_fill_temp.csv", index_col=False)

#   , 'rel_hum', 'wind_dir', 'wind_speed','precipitation', 'conditions'
all_col = ['year', 'month', 'day', 'hour', 'temperature','pressure']
for i in range(5, len(all_col)):
    known_col_list = all_col[:i]
    unknown_col_i = all_col[i]
    pressure_filled_data = fill_missing_col(temperature_filled_data, known_col_list, unknown_col_i)
# pressure_filled_data = fill_missing_temperature(temperature_filled_data)
pressure_filled_data.to_csv("../../data_pool/meteo/meteo_fill_pressure.csv", index=False)