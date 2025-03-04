import pandas as pd
import numpy as np

class sameHour_avg_fill_test():
    def __init__(self):
        return
    
    def get_filled_data(self, all_data_csv):
        self.all_data = pd.read_csv(all_data_csv, index_col=0)
        self.filled_data = self.all_data.copy()
        
        station_columns = [col for col in self.filled_data.columns if col.startswith('station_')]
        
        for col in station_columns:
            # 获取当前列缺失值的位置
            missing_mask = self.filled_data[col].isnull()
            
            # 如果当前列没有缺失值，则跳过
            if not missing_mask.any():
                continue
            
            # 遍历每个缺失值的位置
            for idx in self.filled_data.index[missing_mask]:
                # 获取对应小时
                current_hour = self.filled_data.loc[idx, 'hour']
                
                # 找到同小时的非缺失值
                same_hour_data = self.filled_data.loc[self.filled_data['hour'] == current_hour, col].dropna()
                
                # 如果没有同小时的数据，跳过（可以考虑其他填补方法）
                if same_hour_data.empty:
                    continue
                
                # 计算均值并填补缺失值
                mean_value = same_hour_data.mean()
                self.filled_data.loc[idx, col] = mean_value

        return self.filled_data




def fill_missing_with_same_hour_mean(df, station_columns=None, hour_col='hour'):
    """
    使用同小时均值填补缺失值。
    
    参数：
    - df: 包含数据的 Pandas DataFrame。
    - station_columns: 要处理的站点列名，默认为 None，函数会自动识别形如 "station_" 的列。
    - hour_col: 包含小时信息的列名，默认为 'hour'。
    
    返回：
    - 填补后的 DataFrame。
    """
    # 如果 station_columns 未提供，则自动识别形如 "station_" 的列
    if station_columns is None:
        station_columns = [col for col in df.columns if col.startswith('station_')]
    
    # 创建一个新的 DataFrame，避免修改原始数据
    df_filled = df.copy()
    
    # 遍历每个站点列
    for col in station_columns:
        # 获取当前列缺失值的位置
        missing_mask = df_filled[col].isnull()
        
        # 如果当前列没有缺失值，则跳过
        if not missing_mask.any():
            continue
        
        # 遍历每个缺失值的位置
        for idx in df_filled.index[missing_mask]:
            # 获取对应小时
            current_hour = df_filled.loc[idx, hour_col]
            
            # 找到同小时的非缺失值
            same_hour_data = df_filled.loc[df_filled[hour_col] == current_hour, col].dropna()
            
            # 如果没有同小时的数据，跳过（可以考虑其他填补方法）
            if same_hour_data.empty:
                continue
            
            # 计算均值并填补缺失值
            mean_value = same_hour_data.mean()
            df_filled.loc[idx, col] = mean_value
    
    return df_filled