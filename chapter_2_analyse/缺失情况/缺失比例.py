import pandas as pd

def calculate_missing_ratios(file_path):
    """
    统计每个站点的数据缺失比例。

    参数:
    file_path (str): Excel 文件的路径。

    返回:
    pd.DataFrame: 包含每个站点缺失比例的 DataFrame。
    """
    df = pd.read_csv(file_path)
    
    # 提取所有监测站的数据列
    station_columns = [f'station_{i}' for i in range(1, 23)]
    station_data = df[station_columns]
    
    # 计算每个站点的缺失比例
    missing_ratios = station_data.isna().mean()
    
    # 将结果转换为 DataFrame 并返回
    missing_ratios_df = missing_ratios.reset_index()
    missing_ratios_df.columns = ['Station', 'Missing Ratio']
    
    return missing_ratios_df

# 调用函数并打印结果
missing_ratios = calculate_missing_ratios('../../data_pool/model/data_raw.csv')
print(missing_ratios)