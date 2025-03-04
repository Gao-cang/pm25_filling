import pandas as pd
import numpy as np


class MissingDataConstructor:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.station_cols = [col for col in self.data.columns if 'station_' in col]
        self.continuous_segments = []

    def find_continuous_segments(self):
        """
        找到连续36h及以上连续无缺失的数据片段
        """
        is_nan = self.data[self.station_cols].isnull().any(axis=1)
        start = 0
        for i in range(1, len(is_nan)):
            if is_nan[i] and not is_nan[i - 1]:
                if i - start >= 36:
                    self.continuous_segments.append((start, i - 1))
                start = i
            elif not is_nan[i] and i == len(is_nan) - 1 and i - start + 1 >= 36:
                self.continuous_segments.append((start, i))

    def select_missing_positions(self):
        """
        分别在数据片段上随机选取50处不重叠、不相邻的不同类型的缺失位置
        """
        single_missing = []
        time_continuous_missing = {n: [] for n in [2, 3, 6, 12, 24]}
        spatiotemporal_missing = {n: [] for n in [2, 3, 6, 12, 24]}

        for segment in self.continuous_segments:
            segment_length = segment[1] - segment[0] + 1
            available_indices = set(range(segment[0], segment[1] + 1))

            # 随机单个值缺失
            while len(single_missing) < 50 and available_indices:
                idx = np.random.choice(list(available_indices))
                single_missing.append((idx, 'station_12'))
                available_indices -= {idx - 1, idx, idx + 1}

            # 时间连续缺失
            for n in [2, 3, 6, 12, 24]:
                while len(time_continuous_missing[n]) < 50 and available_indices:
                    start_idx = np.random.choice(list(available_indices))
                    if start_idx + n - 1 <= segment[1] and all(
                            i in available_indices for i in range(start_idx, start_idx + n)):
                        time_continuous_missing[n].append((start_idx, 'station_12', n))
                        available_indices -= set(range(start_idx - 1, start_idx + n + 1))

            # 时空聚集性缺失
            for n in [2, 3, 6, 12, 24]:
                while len(spatiotemporal_missing[n]) < 50 and available_indices:
                    start_idx = np.random.choice(list(available_indices))
                    if start_idx + n - 1 <= segment[1] and all(
                            i in available_indices for i in range(start_idx, start_idx + n)):
                        spatiotemporal_missing[n].append((start_idx, n))
                        available_indices -= set(range(start_idx - 1, start_idx + n + 1))

        return single_missing, time_continuous_missing, spatiotemporal_missing

    def construct_missing_data(self):
        """
        构造假缺失数据并保存到文件
        """
        self.find_continuous_segments()
        single_missing, time_continuous_missing, spatiotemporal_missing = self.select_missing_positions()

        # 线性插值填充真实缺失值
        filled_data = self.data.copy()
        filled_data[self.station_cols] = filled_data[self.station_cols].interpolate(method='linear')

        # 构造随机缺失数据
        random_missing_data = filled_data.copy()
        random_missing_values = []
        for idx, col in single_missing:
            random_missing_values.append((idx, col, random_missing_data.at[idx, col]))
            random_missing_data.at[idx, col] = np.nan
        random_missing_values_df = pd.DataFrame(random_missing_values, columns=['index', 'column', 'true_value'])
        random_missing_values_df.to_csv('/mnt/random_missing_values.csv', index=False)
        random_missing_data.to_csv('/mnt/random_missing_data.csv', index=False)

        # 构造时间连续性缺失数据
        for n, positions in time_continuous_missing.items():
            time_continuous_missing_data = filled_data.copy()
            time_continuous_missing_values = []
            for start_idx, col, num in positions:
                for i in range(num):
                    time_continuous_missing_values.append(
                        (start_idx + i, col, time_continuous_missing_data.at[start_idx + i, col]))
                    time_continuous_missing_data.at[start_idx + i, col] = np.nan
            time_continuous_missing_values_df = pd.DataFrame(time_continuous_missing_values,
                                                             columns=['index', 'column', 'true_value'])
            time_continuous_missing_values_df.to_csv(f'/mnt/time_continuous_missing_values_{n}.csv', index=False)
            time_continuous_missing_data.to_csv(f'/mnt/time_continuous_missing_data_{n}.csv', index=False)

        # 构造时空聚集性缺失数据
        for n, positions in spatiotemporal_missing.items():
            spatiotemporal_missing_data = filled_data.copy()
            spatiotemporal_missing_values = []
            for start_idx, num in positions:
                for col in self.station_cols:
                    for i in range(num):
                        spatiotemporal_missing_values.append(
                            (start_idx + i, col, spatiotemporal_missing_data.at[start_idx + i, col]))
                        spatiotemporal_missing_data.at[start_idx + i, col] = np.nan
            spatiotemporal_missing_values_df = pd.DataFrame(spatiotemporal_missing_values,
                                                            columns=['index', 'column', 'true_value'])
            spatiotemporal_missing_values_df.to_csv(f'/mnt/spatiotemporal_missing_values_{n}.csv', index=False)
            spatiotemporal_missing_data.to_csv(f'/mnt/spatiotemporal_missing_data_{n}.csv', index=False)


# 使用示例
constructor = MissingDataConstructor('../../data_pool/model/data_raw.csv')
constructor.construct_missing_data()