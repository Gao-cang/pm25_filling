import pandas as pd
import numpy as np
from tqdm import tqdm

class DataMissingSimulator:
    def __init__(self, data_path):
        """
        初始化数据模拟器
        :param data_path: 原始数据路径
        """
        # 读取原始数据并预处理
        self.raw_df = pd.read_csv(data_path)
        self._preprocess_data()
        
    def _preprocess_data(self):
        """数据预处理：创建时间索引并进行线性插值"""
        # 创建datetime列
        self.raw_df['datetime'] = pd.to_datetime(
            self.raw_df[['year', 'month', 'day', 'hour']]
        )
        # 设置时间索引并排序
        self.df = self.raw_df.set_index('datetime').sort_index()
        
        # 对每个监测站进行线性插值
        stations = [f'station_{i}' for i in range(1, 23)]
        for station in stations:
            self.df[station] = self.df[station].interpolate(
                method='linear', limit_direction='both'
            )
        
        # 重置索引保留原始行号
        self.df = self.df.reset_index()
        
    def _generate_random_positions(self, n_total, num_missing):
        """生成不重叠不相邻的随机位置"""
        candidates = list(range(n_total))
        selected = []
        while len(selected) < num_missing and candidates:
            idx = np.random.choice(candidates)
            selected.append(idx)
            # 移除相邻位置
            to_remove = {max(0, idx-1), idx, min(n_total-1, idx+1)}
            candidates = [x for x in candidates if x not in to_remove]
        return selected
    
    def _generate_continuous_blocks(self, n_total, block_size, num_blocks):
        """生成连续时间块"""
        blocks = []
        occupied = set()
        attempts = 0
        max_attempts = 10000
        
        while len(blocks) < num_blocks and attempts < max_attempts:
            start = np.random.randint(0, n_total - block_size)
            end = start + block_size - 1
            
            # 检查冲突
            conflict = any(
                (start <= e + 1) and (end >= s - 1) 
                for (s, e) in blocks
            )
            
            if not conflict:
                blocks.append((start, end))
                # 标记占用区域
                occupied.update(range(max(0, start-1), min(n_total, end+2))
                
            attempts += 1
        
        return blocks
    
    def create_random_missing(self, station, num_missing=50):
        """创建随机缺失数据"""
        # 生成随机位置
        positions = self._generate_random_positions(len(self.df), num_missing)
        
        # 创建副本并制造缺失
        simulated_df = self.df.copy()
        records = simulated_df.iloc[positions][['datetime', station]].copy()
        simulated_df.loc[positions, station] = np.nan
        
        # 保存结果
        records.to_csv(f'random_missing_{station}_records.csv', index=False)
        simulated_df.to_csv(f'simulated_random_missing_{station}.csv', index=False)
        return simulated_df, records
    
    def create_temporal_missing(self, station, n_values, num_blocks=50):
        """创建时间连续性缺失"""
        results = {}
        for n in tqdm(n_values, desc="Processing temporal missing"):
            # 生成连续块
            blocks = self._generate_continuous_blocks(len(self.df), n, num_blocks)
            
            # 创建副本并制造缺失
            simulated_df = self.df.copy()
            records = []
            
            for start, end in blocks:
                # 记录真实值
                block_data = simulated_df.iloc[start:end+1][['datetime', station]]
                block_data['block_id'] = f"{start}_{end}"
                records.append(block_data)
                
                # 设置缺失值
                simulated_df.loc[start:end, station] = np.nan
            
            # 保存结果
            records_df = pd.concat(records)
            records_df.to_csv(f'temporal_missing_{station}_n{n}_records.csv', index=False)
            simulated_df.to_csv(f'simulated_temporal_{station}_n{n}.csv', index=False)
            results[n] = (simulated_df, records_df)
        
        return results
    
    def create_spatiotemporal_missing(self, n_values, num_blocks=50):
        """创建时空聚集性缺失"""
        stations = [f'station_{i}' for i in range(1, 23)]
        results = {}
        
        for n in tqdm(n_values, desc="Processing spatiotemporal missing"):
            # 生成连续块
            blocks = self._generate_continuous_blocks(len(self.df), n, num_blocks)
            
            # 创建副本并制造缺失
            simulated_df = self.df.copy()
            records = []
            
            for start, end in blocks:
                # 记录所有站点的真实值
                block_data = simulated_df.iloc[start:end+1][['datetime'] + stations]
                block_data['block_id'] = f"{start}_{end}"
                records.append(block_data)
                
                # 设置缺失值
                simulated_df.loc[start:end, stations] = np.nan
            
            # 保存结果
            records_df = pd.concat(records)
            records_df.to_csv(f'spatiotemporal_missing_n{n}_records.csv', index=False)
            simulated_df.to_csv(f'simulated_spatiotemporal_n{n}.csv', index=False)
            results[n] = (simulated_df, records_df)
        
        return results

# 使用示例
if __name__ == "__main__":
    # 初始化模拟器
    simulator = DataMissingSimulator("your_raw_data.csv")
    
    # 创建随机缺失数据集
    random_simulated, random_records = simulator.create_random_missing("station_12")
    
    # 创建时间连续性缺失数据集（示例使用12和24小时）
    temporal_results = simulator.create_temporal_missing(
        "station_12", 
        n_values=[12, 24],  # 完整列表应为[12,14,16,18,20,22,24]
        num_blocks=50
    )
    
    # 创建时空聚集性缺失数据集（示例使用12和24小时）
    spatiotemporal_results = simulator.create_spatiotemporal_missing(
        n_values=[12, 24],  # 完整列表应为[12,14,16,18,20,22,24]
        num_blocks=50
    )