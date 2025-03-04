import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# 设置英文字体为 Times New Roman
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']

# 设置中文字体为宋体
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

def generate_missing_data_heatmap(file_path):
    # 读取Excel文件
    df = pd.read_csv(file_path, index_col=None)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    fig, ax = plt.subplots()

    # 设置datetime列为索引
    df.set_index('datetime', inplace=True)
    print(df)
    # 提取所有监测站的数据列
    station_columns = [f'station_{i}' for i in range(1, 23)]
    station_data = df[station_columns]
    
    # 创建一个22行（监测站）x 17520列（时间点）的矩阵
    # 初始化矩阵为1（白色，表示有数据）
    heatmap_data = np.ones((22, 17520))
    
    # 遍历每个监测站和时间点，标记缺失数据为0（黑色）
    for i, station in enumerate(station_columns):
        missing_indices = df[station].isna()
        heatmap_data[i, missing_indices] = 0

    # 创建时间标签（每月第一个小时）
    dates = station_data.index
    xticks = [d for d in dates if d.day == 1 and d.hour == 0]
    xtick_labels = [d.strftime('%Y-%m') for d in xticks]
    
    

    # 绘制灰度图
    plt.figure(figsize=(15, 5))
    img = plt.imshow(heatmap_data, cmap='gray', aspect='auto')
    
    # plt.xlabel('Time')
    # plt.ylabel('Stations')
    plt.title('Missing Data Heatmap')
    plt.yticks(range(22), [f'S{i}' for i in range(1, 23)])
    plt.xticks([dates.get_loc(d) for d in xticks], xtick_labels, rotation=45)
    plt.savefig("缺失情况.png", dpi=300)
    plt.show()

# 调用函数生成灰度图
# generate_missing_data_heatmap('../../data_pool/pm25/pm_2223_raw.csv')

def generate_aggregated_missing_heatmap(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path, index_col=None)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

    # 设置datetime列为索引
    df.set_index('datetime', inplace=True)

    # 提取所有监测站的数据列
    station_columns = [f'station_{i}' for i in range(1, 23)]
    station_data = df[station_columns]

    # 计算每个时间点的缺失情况（True表示该时间点有至少一个缺失值）
    missing_mask = station_data.isna().any(axis=1)

    # 创建颜色矩阵（二维数组）
    heatmap_data = np.where(missing_mask, 0, 1).reshape(1, -1)  # 0=黑色 1=白色
    print(heatmap_data)
    # 创建时间标签（每月第一个小时）
    dates = station_data.index
    xticks = [d for d in dates if d.day == 1 and d.hour == 0]
    xtick_labels = [d.strftime('%Y-%m') for d in xticks]

    # 绘制光谱带式热力图
    plt.figure(figsize=(15, 2))  # 调整高度为更紧凑的显示
    img = plt.imshow(heatmap_data,
                     cmap='gray',
                     aspect='auto',
                     extent=[0, len(dates), 0, 1])

    # 设置坐标轴
    plt.yticks([])  # 隐藏y轴
    plt.xticks([dates.get_loc(d) for d in xticks], xtick_labels, rotation=45)

    # 添加辅助元素
    plt.title('Aggregated Missing Data Timeline', pad=20)
    # plt.xlabel('All')
    plt.xlabel('Time')

    # 添加图例
    # colorbar = plt.colorbar(img, orientation='vertical', shrink=0.3)
    # colorbar.set_ticks([0.25, 0.75])
    # colorbar.set_ticklabels(['Missing', 'Complete'])

    # 保存和显示
    plt.tight_layout()
    plt.savefig("aggregated_missing.png", dpi=300, bbox_inches='tight')
    plt.show()

# 使用示例
generate_aggregated_missing_heatmap("../../data_pool/pm25/pm_2223_raw.csv")

def calculate_missing_ratios(file_path):
    """
    统计每个站点的数据缺失比例。

    参数:
    file_path (str): Excel 文件的路径。

    返回:
    pd.DataFrame: 包含每个站点缺失比例的 DataFrame。
    """
    # 读取 Excel 文件
    df = pd.read_csv(file_path)
    
    # 提取所有监测站的数据列
    station_columns = [f'station_{i}' for i in range(1, 23)]
    station_data = df[station_columns]
    
    # 计算每个站点的缺失比例
    missing_ratios = station_data.isna().mean()
    
    # 将结果转换为 DataFrame 并返回
    missing_ratios_df = missing_ratios.reset_index()
    missing_ratios_df.columns = ['Station', 'Missing Ratio']
    missing_ratios_df.to_csv("missing_ratio.csv")
    return missing_ratios_df

# calculate_missing_ratios('../../data_pool/pm25/pm_2223_raw.csv')