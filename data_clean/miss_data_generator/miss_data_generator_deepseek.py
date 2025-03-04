import pandas as pd
import numpy as np
from sklearn.utils import resample

class MissingDataConstructor:
    def __init__(self, file_path):
        # 加载数据并预处理
        self.original_data = pd.read_csv(file_path)
        self.original_data['datetime'] = pd.to_datetime(self.original_data[['year', 'month', 'day', 'hour']])
        self.original_data.set_index('datetime', inplace=True)
        self.original_data.sort_index(inplace=True)
        
        # 定义监测站列
        self.station_columns = [f'station_{i}' for i in range(1, 23)]
        
        # 填充原始缺失值（线性插值）
        self.base_data = self.original_data.copy()
        self.base_data[self.station_columns] = self.base_data[self.station_columns].interpolate(method='linear')
        
        # 检测有效片段（连续36小时无缺失）
        self.valid_segments = self._detect_valid_segments()
    
    def _detect_valid_segments(self, min_hours=36):
        """检测连续min_hours小时无缺失的有效片段"""
        missing_mask = self.base_data[self.station_columns].isna().any(axis=1)
        valid_segments = []
        current_start = None
        
        for i, (ts, is_missing) in enumerate(zip(self.base_data.index, missing_mask)):
            if not is_missing:
                if current_start is None:
                    current_start = ts  # 片段开始
            else:
                if current_start is not None:
                    current_end = self.base_data.index[i-1]
                    if (current_end - current_start).total_seconds() >= min_hours * 3600:
                        valid_segments.append((current_start, current_end))
                    current_start = None
        # 处理最后一个片段
        if current_start is not None:
            current_end = self.base_data.index[-1]
            if (current_end - current_start).total_seconds() >= min_hours * 3600:
                valid_segments.append((current_start, current_end))
        return valid_segments
    
    def _get_segment_indices(self, segment):
        """获取片段内所有时间点"""
        start, end = segment
        return self.base_data.loc[start:end].index
    
    def _generate_positions(self, n, num_samples, station=None):
        """生成连续n小时的缺失位置"""
        candidates = []
        for seg in self.valid_segments:
            seg_indices = self._get_segment_indices(seg)
            seg_length = len(seg_indices)
            if seg_length >= n:
                # 计算可选的起始点数量
                for start in range(0, seg_length - n + 1):
                    candidates.append((seg, start))
        
        if len(candidates) < num_samples:
            raise ValueError(f"Not enough candidates for {n}-hour missing ({len(candidates)} < {num_samples})")
        
        # 无放回抽样
        selected = resample(candidates, n_samples=num_samples, replace=False, random_state=42)
        positions = []
        for (seg, start_idx) in selected:
            seg_indices = self._get_segment_indices(seg)
            start_ts = seg_indices[start_idx]
            end_ts = seg_indices[start_idx + n - 1]
            positions.append((start_ts, end_ts))
        return positions
    
    def construct_random_missing(self, station, num_missing=50):
        """构造随机缺失（单个时间点）"""
        all_indices = []
        for seg in self.valid_segments:
            all_indices.extend(self._get_segment_indices(seg))
        
        if len(all_indices) < num_missing:
            raise ValueError("Not enough valid indices for random missing")
        
        selected = np.random.choice(all_indices, num_missing, replace=False)
        # 转换为时间范围列表（单点）
        return [(ts, ts) for ts in selected]
    
    def construct_missing_data(self, missing_type, n_list, num_samples=50):
        """主函数：构造指定类型的缺失数据"""
        results = {}
        
        if missing_type == "random":
            positions = self.construct_random_missing('station_12', num_samples)
            data = self.base_data.copy()
            for start, end in positions:
                data.loc[start:end, 'station_12'] = np.nan
            results['random'] = (data, positions)
        
        elif missing_type == "temporal":
            for n in n_list:
                positions = self._generate_positions(n, num_samples)
                data = self.base_data.copy()
                for start, end in positions:
                    data.loc[start:end, 'station_12'] = np.nan
                results[f'temporal_{n}'] = (data, positions)
        
        elif missing_type == "spatiotemporal":
            for n in n_list:
                positions = self._generate_positions(n, num_samples)
                data = self.base_data.copy()
                for start, end in positions:
                    data.loc[start:end, self.station_columns] = np.nan
                results[f'spatio_{n}'] = (data, positions)
        
        else:
            raise ValueError("Invalid missing type")
        
        return results
    
    def save_results(self, results, data_prefix="", pos_prefix="missing_positions_"):
        """保存生成的数据和缺失位置"""
        for key, (data, positions) in results.items():
            data.to_csv(f"{data_prefix}{key}.csv")
            pd.DataFrame(positions, columns=["start_time", "end_time"]).to_csv(
                f"{pos_prefix}{key}.csv", index=False
            )

# 使用示例
if __name__ == "__main__":
    constructor = MissingDataConstructor("../../data_pool/model/data_raw.csv")
    
    # 构造随机缺失
    random_results = constructor.construct_missing_data(
        "random", n_list=[], num_samples=50
    )
    constructor.save_results(random_results, "random_", "random_pos_")
    
    # 构造时间连续性缺失
    temporal_results = constructor.construct_missing_data(
        "temporal", n_list=[12,14,16,18,20,22,24], num_samples=50
    )
    constructor.save_results(temporal_results, "temporal_", "temporal_pos_")
    
    # 构造时空缺失
    spatio_results = constructor.construct_missing_data(
        "spatiotemporal", n_list=[12,14,16,18,20,22,24], num_samples=50
    )
    constructor.save_results(spatio_results, "spatio_", "spatio_pos_")