import pandas as pd
import numpy as np

def calculate_pm25_increment(file_path, output_file):
    """
    计算每个站点的 PM2.5 增量（当前时刻值减去前一时刻值）。

    参数:
    file_path (str): 原始数据的 Excel 文件路径。
    output_file (str): 保存增量数据的 CSV 文件路径。

    输出:
    生成包含 PM2.5 增量数据的 CSV 文件。
    """
    # 读取 Excel 文件
    df = pd.read_csv(file_path)
    
    # 提取所有监测站的数据列
    station_columns = [f'station_{i}' for i in range(1, 24)]
    
    # 创建一个新的 DataFrame 存储增量数据
    increment_df = df.copy()
    
    # 计算每个站点的 PM2.5 增量
    for station in station_columns:
        # 计算当前时刻值减去前一时刻值
        increment_df[station] = df[station].diff()
        
        # 如果当前时刻或前一时刻的数据缺失，则增量为 np.nan
        increment_df[station] = increment_df[station].where(
            df[station].notna() & df[station].shift().notna(), np.nan
        )
    
    # 保存增量数据到 CSV 文件
    # increment_df = increment_df.dropna()
    increment_df.to_csv(output_file, index=False)
    print(f"增量数据已保存到 {output_file}")

# 调用函数
# calculate_pm25_increment('../../data_pool/pm25/pm_d2_raw.csv', 'pm25_increment.csv')

import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(pm25_increment_file, meteo_file):
    """
    绘制每个站点的 PM2.5 增量与时间、气象数据的 Pearson 相关性热力图。

    参数:
    pm25_increment_file (str): PM2.5 增量数据的 CSV 文件路径。
    meteo_file (str): 气象数据的 CSV 文件路径。

    输出:
    生成每个站点的相关性热力图，并保存为 PNG 文件。
    """
    # 读取 PM2.5 增量数据
    pm25_increment_df = pd.read_csv(pm25_increment_file)
    
    # 读取气象数据
    meteo_df = pd.read_csv(meteo_file)
    
    # 将 PM2.5 增量数据与气象数据内连接
    merged_df = pd.merge(pm25_increment_df, meteo_df, on=['year', 'month', 'day', 'hour'], how='inner')
    
    # 提取所有监测站的数据列
    station_columns = [f'station_{i}' for i in range(1, 24)]
    
    # 遍历每个站点
    for station in station_columns:
        # 提取当前站点的 PM2.5 增量数据
        station_data = merged_df[['year', 'month', 'day', 'hour', station]].copy()
        
        # 添加时间特征（例如小时、星期几等）
        station_data['hour'] = station_data['hour'].astype(int)  # 确保小时为整数
        station_data['weekday'] = pd.to_datetime(station_data[['year', 'month', 'day']]).dt.weekday  # 星期几
        
        # 提取气象数据列（假设气象数据列名为 'temp', 'humidity', 'wind_speed' 等）
        meteo_columns = [col for col in meteo_df.columns if col not in ['year', 'month', 'day', 'hour']]
        station_data = pd.concat([station_data, merged_df[meteo_columns]], axis=1)
        
        # 计算相关性矩阵
        correlation_matrix = station_data.corr(method='pearson')
        
        # 绘制相关性热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
        plt.title(f'{station} PM2.5 Increment Correlation Heatmap')
        plt.savefig(f'增量相关性/{station}_correlation_heatmap.png')  # 保存热力图
        plt.close()
        print(f"{station} 的相关性热力图已保存为 {station}_correlation_heatmap.png")

# 调用函数
plot_correlation_heatmap('pm25_increment.csv', '../../data_pool/meteo/meteo_2223_clean.csv')