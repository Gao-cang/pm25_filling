import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def calculate_correlation(file_path, max_lag=6):
    """
    计算每个站点前 n 个时刻的 PM2.5 浓度与当前时刻浓度的相关性。

    参数:
    file_path (str): Excel 文件的路径。
    max_lag (int): 最大时间步长（默认为 6）。

    输出:
    生成每个站点的相关性热力图，并保存 r 和 p 值到 CSV 文件中。
    """
    # 读取 Excel 文件
    df = pd.read_csv(file_path)
    
    # 提取所有监测站的数据列
    station_columns = [f'station_{i}' for i in range(1, 23)]
    
    # 遍历每个站点
    for station in station_columns:
        # 提取当前站点的 PM2.5 数据
        pm25_data = df[station].dropna()  # 去除缺失值
        
        # 初始化存储 r 和 p 值的矩阵
        r_values = np.zeros(max_lag)
        p_values = np.zeros(max_lag)
        
        # 计算前 n 个时刻的相关性
        for lag in range(1, max_lag + 1):
            # 创建滞后序列
            lagged_data = pm25_data.shift(lag)
            # 计算 Pearson 相关系数和 p 值
            r, p = pearsonr(pm25_data[lag:], lagged_data[lag:])
            r_values[lag - 1] = r
            p_values[lag - 1] = p
        
        # 保存 r 和 p 值到 CSV 文件
        r_df = pd.DataFrame({'Lag': range(1, max_lag + 1), 'r': r_values})
        p_df = pd.DataFrame({'Lag': range(1, max_lag + 1), 'p': p_values})
        r_df.to_csv(f'{station}_r.csv', index=False)
        p_df.to_csv(f'{station}_p.csv', index=False)
        
        # 绘制相关性热力图
        plt.figure(figsize=(8, 6))
        sns.heatmap(r_values.reshape(1, -1), annot=True, cmap='coolwarm', 
                    xticklabels=range(1, max_lag + 1), yticklabels=[station])
        plt.title(f'{station} Correlation Heatmap')
        plt.xlabel('Lag (hours)')
        plt.ylabel('Station')
        plt.savefig(f'{station}_correlation_heatmap.png')  # 保存热力图
        plt.close()

# 调用函数
calculate_correlation('../../data_pool/pm25/pm_2223_raw.csv', max_lag=6)