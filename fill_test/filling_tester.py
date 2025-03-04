"""
# 填充效果测试框架

## 单次（一个算法 + 一份数据）
1. 链接构造数据
2. 链接填充算法
3. 计算待填充值
4. 链接准备好的原始值
5. 计算准确性指标
6. 记录训练、填充时间

## 循环
1. 确定一个待试验的算法
2. 确定其在随机缺失上的计算流程
3. 确定其在时间连续缺失上的计算流程
4. 确定其在时空聚集性缺失上的计算流程
5. 连接【随机性缺失构造数据】1份+【原始数据】1份、【时间连续性缺失构造数据】5份+【原始数据】1份、【时空聚集性缺失构造数据】5份+【原始数据】1份。

# 对应的模块开发

## tester模块
1. 数据1：待填充数据
2. 接口？：填充算法
3. 数据2：原始数据
4. 方法1：计算准确性指标的方法
5. 方法2：计算训练时间的方法
6. 方法3：整体输出函数
7. 方法4：调用填充方法+提取填充结果
"""

import time
import pandas as pd
import numpy as np
from itertools import product
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from linear_fill.linear_fill import linear_fill_test
from KNN_fill.knn_fill import knn_fill_test
# from GAN_Fill.gan_fill import GANImputer
# from Lagrange.lagrange_fill import lagrange_fill_test
from RF_fill.rf_fill import RF_fill_test
# from XGBoost_fill.xgb_space import XGB_fill_Forward
# from XGBoost_fill.xgb_new import XGB_fill_sing
# from DCCEOF.dcceof_fill import DCCEOF_Filler
# from Bayes.Bayes_fill import BMF_imputer

#================================================

#===============================================


def cols_index(cols_info, index_info_csv):
    index_df = pd.read_csv(index_info_csv)

    # 假设 cols 是一个列表，包含目标列名
    cols = cols_info
    
    # 初始化一个空的 DataFrame 来存储目标矩阵
    target = pd.DataFrame()
    
    # 遍历每一行索引数据
    for _, row in index_df.iterrows():
        start_time = pd.to_datetime(row['start_time'])
        end_time = pd.to_datetime(row['end_time'])
        
        # 生成从 start_time 到 end_time 的时间序列
        time_range = pd.date_range(start=start_time, end=end_time, freq='h')  # 按小时生成时间戳
        time_range_str = time_range.strftime('%Y-%m-%d %H:%M:%S')
        
        combinations = list(product(time_range_str, cols))

        # 创建目标矩阵 target
        temp_df = pd.DataFrame(combinations, columns=['row', 'column'])
        
        # 将临时 DataFrame 添加到目标矩阵中
        target = pd.concat([target, temp_df], ignore_index=True)
    
    return target


class MissingDataTestingFramework:
    def __init__(self, missing_data, missing_timeslice, complete_data, single_station_flag, imputation_algorithm):
        self.missing_data = missing_data
        self.missing_timeslice = missing_timeslice
        self.single_station_flag = single_station_flag
        self.complete_data = pd.read_csv(complete_data, index_col=0)
        self.imputation_algorithm = imputation_algorithm
        
        if self.single_station_flag:
            self.missing_locations = cols_index(['station_12'], self.missing_timeslice)
        else:
            self.missing_locations = cols_index([f'station_{i}' for i in range(1, 23)], self.missing_timeslice)
        


    def evaluate_imputation(self):
        start_time = time.time()
        imputed_data = self.imputation_algorithm.impute_test(self.missing_data)
        path = self.missing_data.split('/')[-1]
        imputed_data.to_csv(f'space_24/3_linear.csv', index=True)
        # print(imputed_data)
        end_time = time.time()
        elapsed_time = end_time - start_time

        mae = 0
        rmse = 0
        r2 = 0
        
        true_value = []
        test_value = []

        for index, row in self.missing_locations.iterrows():
            col = row['column']
            row_num = row['row']
            true_value.append(self.complete_data.at[row_num, col])
            test_value.append(imputed_data.at[row_num, col])
        
        # print(true_value, test_value)
        
        mae = mean_absolute_error(true_value, test_value)
        rmse = np.sqrt(mean_squared_error(true_value, test_value))
        r2 = r2_score(true_value, test_value)

        return mae, rmse, r2, elapsed_time

# 示例使用
if __name__ == "__main__":
    missing_data_list = [
        # "../data_pool/generate_data/random_random.csv",
        # "../data_pool/generate_data/temporal_temporal_2.csv",
        # "../data_pool/generate_data/temporal_temporal_3.csv",
        # "../data_pool/generate_data/temporal_temporal_6.csv",
        # "../data_pool/generate_data/temporal_temporal_12.csv",
        # "../data_pool/generate_data/temporal_temporal_24.csv",
        # "../data_pool/generate_data/spatio_spatio_2.csv",
        # "../data_pool/generate_data/spatio_spatio_3.csv",
        # "../data_pool/generate_data/spatio_spatio_6.csv",
        # "../data_pool/generate_data/spatio_spatio_12.csv",
        "../data_pool/generate_data/spatio_spatio_24.csv",
        ]
    
    miss_data_position_list = [
        # "../data_pool/generate_data/random_pos_random.csv",
        # "../data_pool/generate_data/temporal_pos_temporal_2.csv",
        # "../data_pool/generate_data/temporal_pos_temporal_3.csv",
        # "../data_pool/generate_data/temporal_pos_temporal_6.csv",
        # "../data_pool/generate_data/temporal_pos_temporal_12.csv",
        # "../data_pool/generate_data/temporal_pos_temporal_24.csv",
        # "../data_pool/generate_data/spatio_pos_spatio_2.csv",
        # "../data_pool/generate_data/spatio_pos_spatio_3.csv",
        # "../data_pool/generate_data/spatio_pos_spatio_6.csv",
        # "../data_pool/generate_data/spatio_pos_spatio_12.csv",
        "../data_pool/generate_data/spatio_pos_spatio_24.csv",
        ]
    
    origin_data = "../data_pool/model/data_lfill.csv"
    
    columns = ['mae', 'rmse', 'r2', 'time']
    test_log = pd.DataFrame(columns = columns)
    #====================================
    imputer = linear_fill_test()
    
    #====================================
    # 假设这里读取11份数据
    for i in range(0, 1):
        missing_data = missing_data_list[i]
        missing_locations = miss_data_position_list[i]
        complete_data = origin_data
        #============================
        # imputation_algorithm = linear_fill_test()
        
        # filled_data = imputer.impute_test("../../data_pool/model/data_lfill.csv")
        #============================
        if i < 6:
            single_station_flag = 1
        else:
            single_station_flag = 0
        framework = MissingDataTestingFramework(missing_data, missing_locations, complete_data, single_station_flag, imputer)
        mae, rmse, r2, elapsed_time = framework.evaluate_imputation()
        print(f"Data {missing_data_list[i]}, \nMAE: {mae}, RMSE: {rmse}, R2: {r2}, \nTime: {elapsed_time}")

        test_log.loc[len(test_log)] = [mae, rmse, r2, elapsed_time]
    
    # test_log.to_csv("测试结果/test_16_xgb_space.csv", index=0)

        