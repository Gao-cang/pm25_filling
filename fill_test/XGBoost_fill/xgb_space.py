import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

class XGB_fill_Forward():
    def __init__(self, lag=3):
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
        
        # 生成滞后特征（前3小时数据）
        self._create_lag_features()
        
        # 训练所有站点的模型
        self._train_all_models()
        
        # 找出并填补整行缺失的小时
        self._fill_missing_hours()
        
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
                lag_hour = target_hour - pd.Timedelta(hours=lag)
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

if __name__ == '__main__':
    data_raw = '../../data_pool/generate_data/spatio_spatio_24.csv'
    filler = XGB_fill_timeblock()
    filled_data = filler.impute_test(data_raw)
    filled_data.to_csv('filled_data.csv')