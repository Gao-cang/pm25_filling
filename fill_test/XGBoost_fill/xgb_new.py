import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

class XGB_fill_sing():
    def __init__(self):
        """
        初始化类
        """
        self.best_rf_models = {}  # 保存训练好的模型

    def impute_test(self, all_data_csv):
        """
        主函数，接收数据文件地址并返回填补完毕的数据
        :param all_data_csv: 数据文件路径
        :return: 填补后的DataFrame
        """
        # 读取数据
        self.all_data = pd.read_csv(all_data_csv, index_col=0)
        self.filled_data = self.all_data.copy()
        
        

        # 定义站点列
        station_columns = [f'station_{i}' for i in range(1, 23)]  # station_1~station_23

        for station_target in ['station_12']:
            # 提取子数据集（仅包含当前station和其他特征）
            
            all_stations = [f'station_{i}' for i in range(1, 23)]
            other_stations = [col for col in all_stations if col != station_target]
            self.filled_data['station_avg'] = self.filled_data[other_stations].mean(axis=1, skipna=True)
            self.filled_data['station_avg_pre'] = self.filled_data['station_avg'].shift(1)
            
            data_to_fill = self._extract_data_for_station(station_target)
            
            # 训练随机森林模型
            best_rf_model = self._preprocess_and_train_rf_model(data_to_fill, station_target)

            # 使用训练好的模型循环填补缺失值
            filled_data_for_station = self._fill_missing_values(data_to_fill, best_rf_model, station_target)

            # 更新全局数据
            self.filled_data.loc[filled_data_for_station.index, station_target] = filled_data_for_station[station_target]

        return self.filled_data

    def _extract_data_for_station(self, station_target):
        """
        提取目标station及其相关特征
        :param station: 当前处理的station列名（e.g., 'station_1'）
        :return: 包含指定station及相关特征的DataFrame
        """
        # 选择需要的列：时间特征、气象特征和当前station

        
        columns_to_include = ['year', 'month', 'day', 'hour', 'is_weekday', 'season',
                             'temperature', 'rel_hum', 'wind_speed', 'wind_dir',
                             'pressure', 'precipitation', 'station_avg', 'station_avg_pre', station_target]
        return self.filled_data[columns_to_include]

    def _preprocess_and_train_rf_model(self, df, station_target):
        """
        预处理数据并训练随机森林回归模型
        :param df: 当前处理的DataFrame
        :param station: 当前处理的station列名
        :return: 训练好的随机森林模型
        """
        # 添加station_pre列（上一行值）
        df['station_pre'] = df[station_target].shift(1)

        # 分离已知和未知数据
        known_data = df[df[station_target].notnull()]
        unknown_data = df[df[station_target].isnull()]

        # 进一步分离出用于训练的数据
        known_data_to_train = known_data[known_data['station_pre'].notnull()]
        known_data_cannot_train = known_data[known_data['station_pre'].isnull()]

        # 准备训练集
        X = known_data_to_train.drop(columns=[station_target])
        y = known_data_to_train[station_target]

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 定义随机森林回归器和参数网格
        rf = XGBRegressor(random_state=42)

        rf.fit(X_train, y_train)

        # 预测并评估模型
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        print(f"{station_target} Test set R^2: {r2}")
        print(f"{station_target} Test set MSE: {rmse}")
        print(f"{station_target} Test set MAE: {mae}")

        return rf

    def _fill_missing_values(self, df, best_rf_model, station):
        """
        循环填补缺失值
        :param df: 当前处理的DataFrame
        :param best_rf_model: 训练好的随机森林模型
        :param station: 当前处理的station列名
        :return: 填补后的DataFrame
        """
        df_filled = df.copy()

        while True:
            # 重新计算station_pre列（每次循环都要重新计算）
            df_filled['station_pre'] = df_filled[station].shift(1)

            # 分离已知和未知数据
            known_data = df_filled[df_filled[station].notnull()]
            unknown_data = df_filled[df_filled[station].isnull()]

            # 分离出可以用来预测的数据
            unknow_data_to_predict = unknown_data[unknown_data['station_pre'].notnull()]

            # 如果没有可以预测的数据，结束循环
            if unknow_data_to_predict.empty:
                break

            # 准备预测集
            X_predict = unknow_data_to_predict.drop(columns=[station])

            # 使用训练好的模型进行预测
            predictions = best_rf_model.predict(X_predict)

            # 将预测结果填充到原DataFrame中
            for idx, pred in zip(unknow_data_to_predict.index, predictions):
                df_filled.loc[idx, station] = pred

        return df_filled

if __name__ == '__main__':
    data_raw = '../../data_pool/generate_data/temporal_temporal_24.csv'
    filler = XGB_fill_sing()
    daa_filled = filler.impute_test(data_raw)