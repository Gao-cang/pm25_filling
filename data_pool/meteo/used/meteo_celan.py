#%%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
pd.set_option('display.max_columns', None)

raw_data_path = "../../../data_source/meteo/19meteo.xlsx"

#%%
# 数据清洗开始
raw_meteo_data = pd.read_excel(raw_data_path,sheet_name="Sheet1",header=0,index_col=None)
rename_columns = ["date", "time", "temperature", "rel_hum", "pressure",
                  "wind_dir", "wind_speed", "precipitation", "conditions"]
raw_meteo_data.columns = rename_columns

#%%
# （1）日期处理，将年删去，然后用月、日作为前两列
raw_meteo_data['date'] = pd.to_datetime(raw_meteo_data['date'],format="%d/%m/%Y")
# 从'date'列中提取月份和日
raw_meteo_data['month'] = raw_meteo_data['date'].dt.month.astype(int)
raw_meteo_data['day'] = raw_meteo_data['date'].dt.day.astype(int)
# 删除原始的'date'列
raw_meteo_data = raw_meteo_data.drop(columns=['date'])
# 重新排列列的顺序，将'month'和'day'列作为新的第一列和第二列
cols = list(raw_meteo_data.columns)
cols = cols[-2:] + cols[:-2]
raw_meteo_data = raw_meteo_data[cols]

#%%
# （2）时间列删除Z
raw_meteo_data['time'] = raw_meteo_data['time'].str.replace('Z', '').astype(int)

#%%
# （3）温度列和相对湿度列，转化为float
raw_meteo_data['temperature'] = raw_meteo_data['temperature'].replace('-', np.nan)
raw_meteo_data['temperature'] = raw_meteo_data['temperature'].astype('float64')
raw_meteo_data['rel_hum'] = raw_meteo_data['rel_hum'].replace('-', np.nan)
raw_meteo_data['rel_hum'] = raw_meteo_data['rel_hum'].astype('float32')

#%%
# （4）气压列，去掉单位然后转化为float
raw_meteo_data['pressure'] = raw_meteo_data['pressure'].replace('-', np.nan)
raw_meteo_data['pressure'] = raw_meteo_data['pressure'].str.replace(' Hpa', '').astype('float64')

#%%
# （5）风向列，进行方向编码
def map_wind_direction(wind_dir):
    cleaned_dir = ''.join(filter(str.isalpha, wind_dir.lower())) # 移除非字母字符并转换为小写
    # 根据条件返回对应的数值，如果没有匹配，则返回None
    return {"calm":0, "ºnw":1, "ºn":2,"ºne":3,"ºe":4,"ºse":5,"ºs":6,"ºsw":7,"ºw":8}.get(cleaned_dir, None)  #
# 应用函数到每一行的'wind_dir'列
raw_meteo_data['wind_dir'] = raw_meteo_data['wind_dir'].apply(map_wind_direction).astype(int)

#%%
# （6）风速列，非数字转化为0，
raw_meteo_data['wind_speed'] = raw_meteo_data['wind_speed'].fillna(0).astype('float64')

#%%
# （7）precipitation列有三种情况：
#     （1）如果是”-“则替换为0；
#     （2）如果是'0.5(24h)'这样“数字+'(24h)'”的形式，则删除后面的'(12h)'，保留前面的数字；
#     （3）如果是'0.5(12h)'这样“数字+'(12h)'”的形式，则删除后面的'(12h)'，然后保留前面的数字*2
#     （4）如果是'0.5(6h)'这样“数字+'(6h)'”的形式，则删除后面的'(6h)'，然后保留前面的数字*4
def process_precipitation(value):
    if value == '-' or value == np.nan:
        return 0.0  # 处理第一种情况
    elif '(24h)' in value:
        return float(value.split('(24h')[0])  # 处理第二种情况
    elif '(12h)' in value:
        return float(value.split('(12h')[0]) * 2  # 处理第三种情况
    elif '(6h)' in value:
        return float(value.split('(6h')[0]) * 4  # 处理第三种情况
    else:
        return float(value)  # 其他情况，直接转换为 float
# 应用处理函数
raw_meteo_data['precipitation'] = raw_meteo_data['precipitation'].apply(process_precipitation)

weather_class = {1:[], 2:[], 3:[], 4:[]}

for weather in sorted(list(raw_meteo_data['conditions'].unique())):
    if 'haze' in weather.lower():
        weather_class[4].append(weather)
    elif 'mist' in weather.lower() or 'fog' in weather.lower():
        weather_class[3].append(weather)
    elif 'rain' in weather.lower() or 'snow' in weather.lower() or 'drizzle' in weather.lower():
        weather_class[2].append(weather)
    else:
        weather_class[1].append(weather)

#%%
def map_weather_condition(condition):
    condition_map = weather_class
    for key, values in condition_map.items():
        if condition in values:
            return key
    return np.nan  # 如果没有匹配的情况发生，可以返回None或其他默认值

# 应用函数到每一行的'condition'列
# print(raw_meteo_data['conditions'].unique())
raw_meteo_data['conditions'] = raw_meteo_data['conditions'].apply(map_weather_condition).astype(int)


#%%
raw_meteo_data.dropna(axis=0,how='any')

raw_meteo_data.to_csv("../../data_pool/meteo/clean_meteo_19.csv", header=True, index=False, mode="w")