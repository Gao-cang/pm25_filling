import pandas as pd
import numpy as np
from scipy.linalg import svd
from sklearn.linear_model import LinearRegression
from datetime import timedelta

class DCCEOF_Filler:
    def __init__(self):
        # self.stations = pd.read_csv(station_file)
        self.station_ids = [f"station_{i}" for i in range(1, 23)]
        
        # 参数配置
        self.time_window = 3  # 前后3天
        self.spatial_radius = 100  # 空间半径(km)
        self.k_neighbors = 5
    
    def _build_neighbor_matrix(self, target_time, target_station):
        """构建目标时间点邻域矩阵"""
        # 时间切片
        start = target_time - timedelta(days=self.time_window)
        end = target_time + timedelta(days=self.time_window)
        time_slice = self.data.loc[start:end]
        
        # 空间邻域（这里简化处理为所有站点）
        neighbor_stations = [s for s in self.station_ids if s != target_station]
        
        return time_slice[neighbor_stations]

    def _eof_reconstruction(self, neighbor_matrix):
        """EOF重建日变化模式"""
        filled = neighbor_matrix.copy()
        filled[np.isnan(filled)] = np.nanmean(filled)
        
        # 迭代重建
        for _ in range(10):
            U, s, Vt = svd(filled, full_matrices=False)
            reconstructed = U[:, :1] @ np.diag(s[:1]) @ Vt[:1, :]
            filled[np.isnan(neighbor_matrix)] = reconstructed[np.isnan(neighbor_matrix)]
            
        return reconstructed.flatten()  # 返回目标时间的日变化模式
    
    def impute_test(self, data_path):
        """主填补函数"""
        # 数据加载与预处理
        raw_data = pd.read_csv(data_path)
        raw_data['datetime'] = pd.to_datetime(raw_data[['year','month','day','hour']])
        self.data = raw_data.set_index('datetime')
        
        # 遍历每个站点
        for station in self.station_ids:
            print(f"Processing {station}...")
            station_series = self.data[station]
            missing_dates = station_series[station_series.isna()].index
            
            for target_time in missing_dates:
                # 步骤1：构建邻域矩阵
                neighbor_matrix = self._build_neighbor_matrix(target_time, station)
                
                # 步骤2：EOF重构
                try:
                    diurnal_pattern = self._eof_reconstruction(neighbor_matrix.values)
                except:
                    diurnal_pattern = np.nanmean(neighbor_matrix, axis=1)
                
                # 步骤3：获取有效掩码（仅当前时间窗口）
                window_mask = (neighbor_matrix.index == target_time)
                valid_in_window = ~np.isnan(neighbor_matrix[station].values) if station in neighbor_matrix else np.zeros(len(neighbor_matrix), dtype=bool)
                
                # 确保维度匹配
                if len(diurnal_pattern) != len(window_mask):
                    diurnal_pattern = np.resize(diurnal_pattern, len(window_mask))
                
                # 线性回归预测
                if np.sum(valid_in_window) > 5:
                    model = LinearRegression()
                    model.fit(diurnal_pattern[valid_in_window].reshape(-1,1), 
                             neighbor_matrix[station].values[valid_in_window])
                    pred = model.predict(diurnal_pattern[window_mask].reshape(-1,1))
                    self.data.loc[target_time, station] = pred[0]
                else:
                    # 回退到时空均值
                    self.data.loc[target_time, station] = np.nanmean(neighbor_matrix.values)
        
        return self.data

# 使用示例
if __name__ == "__main__":
    imputer = DCCEOF_Filler('../../data_pool/model/station_lng_lat.csv')
    filled_data = imputer.impute_test('../../data_pool/model/data_raw.csv')
    filled_data.to_csv("filled_result.csv")

