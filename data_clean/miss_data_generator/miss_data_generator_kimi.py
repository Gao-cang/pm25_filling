import pandas as pd
import numpy as np

class MissingDataSimulator:
    def __init__(self, data, min_length=30, station_col_prefix="station_"):
        """
        初始化模拟器
        :param data: 输入的原始数据 (DataFrame)
        :param min_length: 可用时间段的最小长度 (默认 30 小时)
        :param station_col_prefix: 监测站点列的前缀 (默认 "station_")
        """
        self.data = data
        self.min_length = min_length
        self.station_col_prefix = station_col_prefix
        self.time_cols = ["year", "month", "day", "hour"]
        self.stations = [col for col in data.columns if col.startswith(self.station_col_prefix)]

    def _detect_valid_segments(self, station_col):
        """
        检测特定站点的有效时间片段
        :param station_col: 目标站点列名
        :return: 有效时间段的索引列表
        """
        valid_segments = []
        current_segment = []
        for idx, row in self.data.iterrows():
            if not pd.isna(row[station_col]):
                current_segment.append(idx)
                if len(current_segment) >= self.min_length:
                    if current_segment[-1] - current_segment[0] + 1 == self.min_length:
                        valid_segments.append(current_segment.copy())
                        current_segment = []
            else:
                current_segment = []
        return valid_segments

    def generate_random_missing(self, target_station, n_samples=50, output_file="random_missing.csv"):
        """
        构造随机缺失数据
        :param target_station: 目标站点 (station_12)
        :param n_samples: 每次随机选取消失的样本数 (默认 50)
        :param output_file: 输出文件名
        """
        valid_segments = self._detect_valid_segments(target_station)
        if not valid_segments:
            raise ValueError(f"No valid segments found for {target_station}")

        # 随机选择 n_samples 个非重叠、非相邻的缺失值
        valid_indices = []
        for segment in valid_segments:
            valid_indices.extend(segment)
        selected_indices = np.random.choice(valid_indices, size=n_samples, replace=False)

        # 确保选中的索引不相邻
        selected_indices = sorted(selected_indices)
        clean_indices = [selected_indices[0]]
        for idx in selected_indices[1:]:
            if idx - clean_indices[-1] > 1:
                clean_indices.append(idx)
        selected_indices = clean_indices[:n_samples]

        # 记录位置和真实值
        missing_info = []
        for idx in selected_indices:
            missing_info.append({
                "year": self.data.loc[idx, "year"],
                "month": self.data.loc[idx, "month"],
                "day": self.data.loc[idx, "day"],
                "hour": self.data.loc[idx, "hour"],
                "station": target_station,
                "true_value": self.data.loc[idx, target_station]
            })

        # 复制数据并构造缺失值
        sim_data = self.data.copy()
        sim_data.loc[selected_indices, target_station] = np.nan
        sim_data.to_csv(output_file, index=False)
        return missing_info

    def generate_temporal_missing(self, target_station, n=2, n_samples=50, output_base_name="temporal_missing"):
        """
        构造时间连续性缺失数据
        :param target_station: 目标站点 (station_12)
        :param n: 连续缺失值的数量 (2/3/6/12/24)
        :param n_samples: 每次随机选取消失的样本数 (默认 50)
        :param output_base_name: 输出文件的基本名称
        """
        valid_segments = self._detect_valid_segments(target_station)
        if not valid_segments:
            raise ValueError(f"No valid segments found for {target_station}")

        # 随机选择 n_samples 个连续 n 个值的片段
        valid_start_indices = []
        for segment in valid_segments:
            max_start = len(segment) - n
            if max_start > 0:
                valid_start_indices.extend(segment[:max_start])

        selected_starts = np.random.choice(valid_start_indices, size=n_samples, replace=False)
        selected_indices = []
        for start in selected_starts:
            selected_indices.extend(range(start, start + n))

        # 确保定片段不重叠且不相邻
        selected_indices = sorted(set(selected_indices))
        clean_indices = [selected_indices[0]]
        for idx in selected_indices[1:]:
            if idx - clean_indices[-1] > 1:
                clean_indices.append(idx)
        selected_indices = clean_indices[:n_samples * n]

        # 记录位置和真实值
        missing_info = []
        for idx in selected_indices:
            missing_info.append({
                "year": self.data.loc[idx, "year"],
                "month": self.data.loc[idx, "month"],
                "day": self.data.loc[idx, "day"],
                "hour": self.data.loc[idx, "hour"],
                "station": target_station,
                "true_value": self.data.loc[idx, target_station]
            })

        # 复制数据并构造缺失值
        sim_data = self.data.copy()
        sim_data.loc[selected_indices, target_station] = np.nan
        sim_data.to_csv(f"{output_base_name}_n{n}.csv", index=False)
        return missing_info

    def generate_spatiotemporal_missing(self, n=2, n_samples=50, output_base_name="spatiotemporal_missing"):
        """
        构造时空聚集性缺失数据
        :param n: 连续缺失值的数量 (2/3/6/12/24)
        :param n_samples: 每次随机选取消失的样本数 (默认 50)
        :param output_base_name: 输出文件的基本名称
        """
        valid_segments = []
        for station in self.stations:
            valid_segments.extend(self._detect_valid_segments(station))

        valid_segments = sorted(list(set(valid_segments)), key=lambda x: x[0])

        # 随机选择 n_samples 个连续 n 个值的片段
        selected_starts = []
        for segment in valid_segments:
            max_start = len(segment) - n
            if max_start > 0:
                selected_starts.extend(segment[:max_start])

        selected_starts = np.random.choice(selected_starts, size=n_samples, replace=False)
        selected_indices = []
        for start in selected_starts:
            selected_indices.extend(range(start, start + n))

        # 确保定片段不重叠且不相邻
        selected_indices = sorted(set(selected_indices))
        clean_indices = [selected_indices[0]]
        for idx in selected_indices[1:]:
            if idx - clean_indices[-1] > 1:
                clean_indices.append(idx)
        selected_indices = clean_indices[:n_samples * n]

        # 记录位置和真实值
        missing_info = []
        for idx in selected_indices:
            for station in self.stations:
                if not pd.isna(self.data.loc[idx, station]):
                    missing_info.append({
                        "year": self.data.loc[idx, "year"],
                        "month": self.data.loc[idx, "month"],
                        "day": self.data.loc[idx, "day"],
                        "hour": self.data.loc[idx, "hour"],
                        "station": station,
                        "true_value": self.data.loc[idx, station]
                    })

        # 复制数据并构造缺失值
        sim_data = self.data.copy()
        sim_data.loc[selected_indices, self.stations] = np.nan
        sim_data.to_csv(f"{output_base_name}_n{n}.csv", index=False)
        return missing_info


# 示例用法
if __name__ == "__main__":
    # 假设原始数据已经加载到 df 中
    df = pd.read_csv("../../data_pool/model/data_raw.csv")  # 请替换为你的原始数据文件路径

    simulator = MissingDataSimulator(df)

    # 构造随机缺失数据
    print("Generating Random Missing...")
    simulator.generate_random_missing(target_station="station_12", n_samples=50, output_file="random_missing.csv")

    # 构造时间连续性缺失数据
    for n in [2, 3, 6, 12, 24]:
        print(f"Generating Temporal Missing (n={n})...")
        simulator.generate_temporal_missing(target_station="station_12", n=n, n_samples=50, output_base_name="temporal_missing")

    # 构造时空聚集性缺失数据
    for n in [2, 3, 6, 12, 24]:
        print(f"Generating Spatiotemporal Missing (n={n})...")
        simulator.generate_spatiotemporal_missing(n=n, n_samples=50, output_base_name="spatiotemporal_missing")