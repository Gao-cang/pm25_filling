import pandas as pd
import numpy as np
from collections import defaultdict

# 读取数据，处理缺失值
df = pd.read_csv('../../data_pool/model/data_raw.csv', index_col=None)
# , na_values=['', 'NaN']
stations = [col for col in df.columns if col.startswith('station_')]

# ============================== 任务1：统计连续缺失值 ==============================
def count_consecutive_missing(series):
    counts = defaultdict(int)
    current_count = 0
    for val in series:
        if pd.isna(val):
            current_count += 1
        else:
            if current_count > 0:
                counts[current_count] += 1
                current_count = 0
    # 处理末尾连续缺失的情况
    if current_count > 0:
        counts[current_count] += 1
    return counts

missing_stats = {}
for station in stations:
    missing_stats[station] = count_consecutive_missing(df[station])

# ============================== 任务2：统计单调性比例 ==============================
def is_monotonic(arr):
    increasing = all(x <= y for x, y in zip(arr, arr[1:]))
    decreasing = all(x >= y for x, y in zip(arr, arr[1:]))
    return increasing or decreasing

# 收集所有连续非缺失序列
all_sequences = []
for station in stations:
    series = df[station].dropna().values
    if len(series) >= 3:  # 至少需要3个数据点
        all_sequences.append(series)

# 统计各长度单调比例
monotonic_ratios = {}
for n in range(3, 13):
    total = 0
    valid = 0
    for seq in all_sequences:
        if len(seq) < n:
            continue
        # 滑动窗口检查
        for i in range(len(seq) - n + 1):
            window = seq[i:i+n]
            total += 1
            if is_monotonic(window):
                valid += 1
    if total > 0:
        monotonic_ratios[n] = round(valid / total, 2)
    else:
        monotonic_ratios[n] = 0.0

# ============================== 输出结果 ==============================
print("任务1结果：")
for station in list(missing_stats.keys())[:23]:  # 示例输出前2个站点
    print(f"{station}: {dict(missing_stats[station])}")

print("\n任务2结果：")
print(monotonic_ratios)