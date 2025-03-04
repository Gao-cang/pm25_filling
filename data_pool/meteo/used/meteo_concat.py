import pandas as pd
import numpy as np

raw_meteo_data = pd.read_csv("../../data_pool/meteo/clean_meteo_19.csv", index_col=False)
raw_meteo_data['year'] = 2019

for y in range(20,24):
    y_meteo = pd.read_csv(f"../../data_pool/meteo/clean_meteo_{y}.csv", index_col=False)
    y_meteo['year'] = int(f'20{y}')

    raw_meteo_data = pd.concat([raw_meteo_data, y_meteo], axis=0)

raw_meteo_data = raw_meteo_data[['year', 'month', 'day', 'time', 'temperature', 'rel_hum', 'pressure', 'wind_dir',
                                 'wind_speed', 'precipitation', 'conditions']].sort_values(by=['year', 'month', 'day', 'time'])
raw_meteo_data.columns = ['year', 'month', 'day', 'hour', 'temperature', 'rel_hum', 'pressure','wind_dir', 'wind_speed',
                          'precipitation', 'conditions']

def fill_missing_times(df):
    # 将年月日时转换为datetime格式
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

    # 设置datetime列为索引
    df.set_index('datetime', inplace=True)

    # 重新采样以确保每小时都有记录，如果缺失则填充NaN
    full_df = df.resample('3H').asfreq()

    # 重置索引，以便可以访问到datetime列
    full_df.reset_index(inplace=True)

    # 更新year, month, day, hour列
    full_df['year'] = full_df['datetime'].dt.year
    full_df['month'] = full_df['datetime'].dt.month
    full_df['day'] = full_df['datetime'].dt.day
    full_df['hour'] = full_df['datetime'].dt.hour

    # 删除辅助用的datetime列
    full_df.drop(columns=['datetime'], inplace=True)

    # 确保station_x列全部为NaN
    for col in df.columns[4:]:  # 假设从第7列开始是'station_x'列
        if col not in full_df:
            full_df[col] = np.nan

    return full_df

fill_time_meteo_data = fill_missing_times(raw_meteo_data)
print(fill_time_meteo_data)
#
# for column in fill_time_d2_data.columns[6:]:  # 假设从第7列开始是'station_x'列
#     fill_time_d2_data[column] = fill_time_d2_data[column].interpolate(method='linear')
#     # 如果第一行仍有缺失值，使用前向填充
#     if fill_time_d2_data[column].isna().iloc[0]:
#         fill_time_d2_data[column] = fill_time_d2_data[column].fillna(method='ffill').fillna(method='bfill')
#
# fill_time_d2_data[[f'station_{i}' for i in range(1,24)]] = fill_time_d2_data[[f'station_{i}' for i in range(1,24)]].astype(int)
# fill_time_d2_data = fill_time_d2_data.drop(["station_12"], axis=1)
# print(fill_time_d2_data)
fill_time_meteo_data.to_csv("../../data_pool/meteo/meteo_with_nan.csv", index=False)