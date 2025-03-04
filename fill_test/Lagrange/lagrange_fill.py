import pandas as pd
import numpy as np

class lagrange_fill_test:
    def __init__(self, window_size=3):
        self.window_size = window_size  # 用于插值的窗口大小
    
    def _lagrange_interpolate(self, series):
        """对单个序列执行拉格朗日插值"""
        series = series.copy()
        n = len(series)
        for i in range(n):
            if not np.isnan(series[i]):
                continue
            
            # 确定滑动窗口边界
            start = max(0, i - self.window_size)
            end = min(n, i + self.window_size + 1)
            window = series[start:end]
            
            # 获取已知点的索引和值
            known_indices = [x for x in range(len(window)) if not np.isnan(window[x])]
            known_values = window[~np.isnan(window)]
            
            # 至少需要2个点才能进行插值
            if len(known_values) >= 2:
                # 拉格朗日插值多项式
                poly = np.polynomial.Polynomial.fit(
                    known_indices,
                    known_values,
                    deg=len(known_values)-1
                )
                series[i] = poly(i - start)
        
        # 用均值填补剩余的 NaNs（如有）
        mean_val = np.nanmean(series)
        series[np.isnan(series)] = mean_val
        
        return series

    def impute_test(self, all_data_csv):
        """主函数：读取数据并执行插值"""
        # 读取数据并设置时间索引
        self.all_data = pd.read_csv(all_data_csv)
        if 'datetime' not in self.all_data.columns:
            raise ValueError("Data must include a 'datetime' column.")
        self.all_data['datetime'] = pd.to_datetime(self.all_data['datetime'])
        self.all_data.set_index('datetime', inplace=True)
        
        self.filled_data = self.all_data.copy()
        
        # 按列进行插值
        for col in self.filled_data.columns:
            # 跳过非数值列（假设站点数据是数值列）
            if np.issubdtype(self.filled_data[col].dtype, np.number):
                self.filled_data[col] = self._lagrange_interpolate(self.filled_data[col].values)
        
        # 确保所有 NaNs 都被填补
        if self.filled_data.isnull().values.any():
            # 假设 NaNs 已经被均值填补，但可以进一步检查
            raise Warning("Some NaNs remain in the data after imputation.")
        
        return self.filled_data