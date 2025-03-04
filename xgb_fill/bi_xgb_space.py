import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

class XGB_fill_Forward():
    def __init__(self, lag=1):
        self.lag = lag
        self.models = {}  # 保存每个站点的模型
        self.station_columns = [f'station_{i}' for i in range(1, 23)]
        self.time_features = ['year', 'month', 'day', 'hour', 'is_weekday', 'season']
        self.weather_features = ['temperature', 'rel_hum', 'wind_speed', 
                               'wind_dir', 'pressure', 'precipitation']

    def impute_test(self, all_data_csv):
        # 读取数据并确保时间索引
        self.all_data = pd.read_csv(all_data_csv, parse_dates=True, index_col=0)
        self.filled_data = self.all_data.copy().reset_index()
        
        # 生成滞后特征（前3小时数据）
        self._create_lag_features()
        
        # 训练所有站点的模型
        self._train_all_models()
        
        # 找出并填补整行缺失的小时
        self._fill_missing_hours()
        
        self.filled_data.set_index("datetime", inplace=True)
        # print(self.filled_data.head(10))
        # self.filled_data.to_csv("filled2.csv")
        
        return self.filled_data

    def _create_lag_features(self):
        # 为每个站点创建前3小时的滞后特征
        for station in self.station_columns:
            for lag in range(1, self.lag+1):
                self.filled_data[f'{station}_lag{lag}'] = self.filled_data[station].shift(lag)

    def _get_feature_columns(self):
        # 获取特征列的顺序（必须与训练时一致）
        lag_features = []
        for lag in range(1, self.lag+1):
            for station in self.station_columns:
                lag_features.append(f'{station}_lag{lag}')
        return self.time_features + self.weather_features + lag_features

    def _prepare_training_data(self, station_target):
        # 准备训练数据集
        features = self._get_feature_columns()
        df = self.filled_data[features + [station_target]].dropna()
        return df

    def _train_model(self, station_target):
        # 训练单个站点的模型
        data = self._prepare_training_data(station_target)
        if data.empty:
            return None
        
        X = data[self._get_feature_columns()]
        y = data[station_target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = XGBRegressor()
        model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = model.predict(X_test)
        print(f"Model for {station_target}:")
        print(f"R2: {r2_score(y_test, y_pred):.3f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
        print("="*50)
        return model

    def _train_all_models(self):
        # 为每个站点训练模型
        for station in self.station_columns:
            print(f"Training model for {station}...")
            model = self._train_model(station)
            self.models[station] = model

    def _fill_missing_hours(self):
        # 找出所有完全缺失的小时（所有站点同时缺失）
        missing_mask = self.filled_data[self.station_columns].isnull().all(axis=1)
        missing_hours = self.filled_data[missing_mask].index.tolist()
        
        # 按时间顺序处理缺失小时
        for hour in sorted(missing_hours):
            # 构造特征向量
            features = self._get_hour_features(hour)
            if features is None:
                continue
                
            # 为所有站点预测值
            predictions = {}
            for station in self.station_columns:
                if self.models[station] is not None:
                    predictions[station] = self.models[station].predict(features)[0]
            
            # 更新填补数据
            self.filled_data.loc[hour, self.station_columns] = pd.Series(predictions)

    def _get_hour_features(self, target_hour):
        # 构造目标小时的特征向量
        try:
            # 时间气象特征
            time_weather = self.filled_data.loc[target_hour, 
                                             self.time_features + self.weather_features].values
            
            # 滞后特征（前3小时数据）
            lag_features = []
            for lag in range(1, self.lag+1):
                lag_hour = target_hour - lag
                if lag_hour in self.filled_data.index:
                    lag_values = self.filled_data.loc[lag_hour, self.station_columns].values
                else:
                    lag_values = [np.nan] * len(self.station_columns)
                lag_features.extend(lag_values)
            
            # 合并所有特征
            return np.concatenate([time_weather, lag_features]).reshape(1, -1)
            
        except KeyError:
            print(f"Cannot find data for hour {target_hour}")
            return None
    
class XGB_fill_Backward():
    def __init__(self, lag=1):
        self.lag = lag
        self.models = {}  # 保存每个站点的模型
        self.station_columns = [f'station_{i}' for i in range(1, 23)]
        self.time_features = ['year', 'month', 'day', 'hour', 'is_weekday', 'season']
        self.weather_features = ['temperature', 'rel_hum', 'wind_speed', 
                               'wind_dir', 'pressure', 'precipitation']

    def impute_test(self, all_data_csv):
        # 读取数据并确保时间索引
        self.all_data = pd.read_csv(all_data_csv, parse_dates=True, index_col=0)
        self.filled_data = self.all_data.copy()
        
        self.filled_data = self.filled_data[::-1].reset_index()

        # 生成滞后特征（前3小时数据）
        self._create_lag_features()
        
        # 训练所有站点的模型
        self._train_all_models()
        
        # 找出并填补整行缺失的小时
        self._fill_missing_hours()

        self.filled_data = self.filled_data[::-1].reset_index().drop(['index'],axis=1)
        self.filled_data.set_index("datetime", inplace=True)
        # self.filled_data.to_csv("filled.csv")
        
        return self.filled_data

    def _create_lag_features(self):
        # 为每个站点创建前3小时的滞后特征
        for station in self.station_columns:
            for lag in range(1, self.lag+1):
                self.filled_data[f'{station}_lag{lag}'] = self.filled_data[station].shift(lag)

    def _get_feature_columns(self):
        # 获取特征列的顺序（必须与训练时一致）
        lag_features = []
        for lag in range(1, self.lag+1):
            for station in self.station_columns:
                lag_features.append(f'{station}_lag{lag}')
        return self.time_features + self.weather_features + lag_features

    def _prepare_training_data(self, station_target):
        # 准备训练数据集
        features = self._get_feature_columns()
        df = self.filled_data[features + [station_target]].dropna()
        return df

    def _train_model(self, station_target):
        # 训练单个站点的模型
        data = self._prepare_training_data(station_target)
        if data.empty:
            return None
        
        X = data[self._get_feature_columns()]
        y = data[station_target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = XGBRegressor()
        model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = model.predict(X_test)
        print(f"Model for {station_target}:")
        print(f"R2: {r2_score(y_test, y_pred):.3f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
        print("="*50)
        return model

    def _train_all_models(self):
        # 为每个站点训练模型
        for station in self.station_columns:
            print(f"Training model for {station}...")
            model = self._train_model(station)
            self.models[station] = model

    def _fill_missing_hours(self):
        # 找出所有完全缺失的小时（所有站点同时缺失）
        missing_mask = self.filled_data[self.station_columns].isnull().all(axis=1)
        missing_hours = self.filled_data[missing_mask].index.tolist()
        
        # 按时间顺序处理缺失小时
        for hour in sorted(missing_hours):
            # 构造特征向量
            features = self._get_hour_features(hour)
            if features is None:
                continue
                
            # 为所有站点预测值
            predictions = {}
            for station in self.station_columns:
                if self.models[station] is not None:
                    predictions[station] = self.models[station].predict(features)[0]
            
            # 更新填补数据
            self.filled_data.loc[hour, self.station_columns] = pd.Series(predictions)

    def _get_hour_features(self, target_hour):
        # 构造目标小时的特征向量
        try:
            # 时间气象特征
            time_weather = self.filled_data.loc[target_hour, 
                                             self.time_features + self.weather_features].values
            
            # 滞后特征（前3小时数据）
            lag_features = []
            for lag in range(1, self.lag+1):
                lag_hour = target_hour - lag
                if lag_hour in self.filled_data.index:
                    lag_values = self.filled_data.loc[lag_hour, self.station_columns].values
                else:
                    lag_values = [np.nan] * len(self.station_columns)
                lag_features.extend(lag_values)
            
            # 合并所有特征
            return np.concatenate([time_weather, lag_features]).reshape(1, -1)
            
        except KeyError:
            print(f"Cannot find data for hour {target_hour}")
            return None

class Bi_XGBoost():
    def __init__(self):
        
        
        self.forward_model = XGB_fill_Forward(1)
        self.backward_model = XGB_fill_Backward(1)
    
    def record_miss(self):
        df = pd.read_csv(self.raw_data_file, index_col=0)
        # 确保 index 是 datetime 类型
        df.index = pd.to_datetime(df.index)
        
        # 定义需要处理的列
        columns = [f'station_{i}' for i in range(1, 23)]
    
        # 遍历每一列
        for column in columns:
            # 使用自然数索引遍历
            natural_index = np.arange(len(df))
            # 获取缺失值的行
            missing_rows = df[column].isnull()
            
            start_indices = []
            end_indices = []
            start_time_list = []
            end_time_list = []
            
            current_start = None
            current_end = None
            
            # 遍历自然数索引
            for i, (idx, is_missing) in enumerate(zip(natural_index, missing_rows)):
                if is_missing:
                    # 缺失块开始或扩展
                    if current_start is None:
                        current_start = i
                        current_end = i
                    else:
                        current_end = i
                else:
                    # 缺失块结束
                    if current_start is not None:
                        # 保存区间
                        start_indices.append(current_start)
                        end_indices.append(current_end)
                        # 保存时间戳
                        start_time = df.index[current_start].strftime("%Y/%m/%d %H:%M")
                        end_time = df.index[current_end].strftime("%Y/%m/%d %H:%M")
                        start_time_list.append(start_time)
                        end_time_list.append(end_time)
                        # 重置
                        current_start = None
                        current_end = None
            
            # 检查最后是否有未结束的缺失块
            if current_start is not None:
                start_indices.append(current_start)
                end_indices.append(current_end)
                # 保存时间戳
                start_time = df.index[current_start].strftime("%Y/%m/%d %H:%M")
                end_time = df.index[current_end].strftime("%Y/%m/%d %H:%M")
                start_time_list.append(start_time)
                end_time_list.append(end_time)
            
            # 创建结果 DataFrame
            result = pd.DataFrame({
                'start_time': start_time_list,
                'end_time': end_time_list
            })
    
            # 保存结果到文件
            file_name = f"space_rec/miss_{column}_interval.csv"
            result.to_csv(file_name, index=False)
            print(f"Missing intervals for {column} saved to {file_name}")
        
    def impute_test(self, raw_data_file):
        self.raw_data_file = raw_data_file
        
        self.record_miss()

        # 加载数据集 A 和 B
        A = self.forward_model.impute_test(self.raw_data_file)
        B = self.backward_model.impute_test(self.raw_data_file)

        # 复制数据集 A 作为基础
        C = A.copy()
    
        # 处理每一列
        for col in A.columns:
            # 加载缺失数据位置文件
            missing_file = f"space_rec/miss_{col}_interval.csv"
            try:
                missing_df = pd.read_csv(missing_file)
            except FileNotFoundError:
                print(f"警告：未找到缺失数据文件 {missing_file}，跳过列 {col}")
                continue
            
            # 遍历该列的每一行缺失区间
            for _, row in missing_df.iterrows():
                start_time = pd.to_datetime(row['start_time'])
                end_time = pd.to_datetime(row['end_time'])
                
                # 在 A 和 B 中找到对应的行
                A_interval = A.loc[start_time:end_time, col]
                B_interval = B.loc[start_time:end_time, col]
                
                # 计算权重
                n = len(A_interval)
                if n == 0:
                    continue  # 跳过空区间
                
                # 权重从 1 到 0（A 的权重）
                x = np.linspace(np.pi/2, 0, n)
                
                weights_a = np.sin(x)
                # 权重从 0 到 1（B 的权重）
                y = np.linspace(0, np.pi/2, n)
                weights_b = np.sin(y)
                weight_sum = weights_a + weights_b
                
                # 计算加权平均值
                # 修复 nan 值
                A_values = A_interval.fillna(0).values
                B_values = B_interval.fillna(0).values
                
                weighted_avg = ((A_values * weights_a) + (B_values * weights_b)) / weight_sum
                
                # 更新数据集 C
                C.loc[start_time:end_time, col] = weighted_avg
        
        return C


if __name__ == '__main__':
    test_file = "../data_pool/generate_data/spatio_spatio_24.csv"
    # imputer = XGB_fill_Backward(lag=1)
    # imputed_data = imputer.impute_test("../data_pool/generate_data/spatio_spatio_24.csv")
    bi_xgb_filler = Bi_XGBoost()
    final = bi_xgb_filler.impute_test(test_file)
    final.to_csv("filled.csv", index=True)