import pandas as pd

pm_data_raw = pd.read_csv('../../data_pool/pm25/pm_2223_raw.csv', index_col=None)
meteo_data = pd.read_csv('../../data_pool/meteo/meteo_2223_clean.csv', index_col=None)

model_data = pd.merge(pm_data_raw, meteo_data, on=['year', 'month', 'day', 'hour'])
model_data = model_data[['year', 'month', 'day', 'hour', 'is_weekday', 'season', 'condition', 'temperature', 'rel_hum',
                         'wind_speed', 'wind_dir', 'pressure', 'precipitation', 'station_1', 'station_2', 'station_3',
                         'station_4', 'station_5', 'station_6', 'station_7', 'station_8', 'station_9', 'station_10',
                         'station_11', 'station_12', 'station_13', 'station_14', 'station_15', 'station_16',
                         'station_17', 'station_18', 'station_19', 'station_20', 'station_21', 'station_22']]
model_data.to_csv('../../data_pool/model/data_raw.csv', mode='w', index=False)
print(model_data.columns)