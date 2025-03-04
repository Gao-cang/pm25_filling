import pandas as pd
import numpy as np

def fill_missing_times(df):
    df = df[df['year'].isin([2022,2023])]
    # 将年月日时转换为datetime格式
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

    # 设置datetime列为索引
    df.set_index('datetime', inplace=True)

    # 重新采样以确保每小时都有记录，如果缺失则填充NaN
    full_df = df.resample('h').asfreq()

    # 重置索引，以便可以访问到datetime列
    full_df.reset_index(inplace=True)

    # 填充weekday和season信息
    full_df['weekday'] = full_df['datetime'].dt.weekday + 1  # 星期一为1
    full_df['season'] = (full_df['datetime'].dt.month % 12 + 3) // 3  # 季节计算

    # 更新year, month, day, hour列
    full_df['year'] = full_df['datetime'].dt.year
    full_df['month'] = full_df['datetime'].dt.month
    full_df['day'] = full_df['datetime'].dt.day
    full_df['hour'] = full_df['datetime'].dt.hour

    # 删除辅助用的datetime列
    full_df.drop(columns=['datetime'], inplace=True)

    # 确保station_x列全部为NaN
    for col in df.columns[6:]:  # 假设从第7列开始是'station_x'列
        if col not in full_df:
            full_df[col] = np.nan

    return full_df

raw_d2_data = pd.read_csv("../../data_source/pm25/pm_1924_raw.csv", index_col=False)
# raw_d2_data['station_12'] = 0
fill_time_d2_data = fill_missing_times(raw_d2_data)

# fill_time_d2_data[[f'station_{i}' for i in range(1,24)]] = fill_time_d2_data[[f'station_{i}' for i in range(1,24)]].astype(int)
# fill_time_d2_data = fill_time_d2_data.drop(["station_12"], axis=1)
print(fill_time_d2_data)
fill_time_d2_data.to_csv("../../data_pool/pm25/pm_2223_raw.csv", index=False)