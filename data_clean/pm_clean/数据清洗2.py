import pandas as pd

def weekday_to_isweekday(raw_file):
    raw_data = pd.read_csv(raw_file, index_col=None)
    raw_data['weekday'] = (raw_data['weekday'] <= 5).astype(int)
    raw_data.rename(columns={'weekday': 'is_weekday'}, inplace=True)
    raw_data.to_csv(raw_file, index=False)
    print(raw_data)

# weekday_to_isweekday("../../data_pool/pm25/pm_2223_raw.csv ")

def get_removeNan(raw_file):
    raw_data = pd.read_csv(raw_file, index_col=None)
    removeNan = raw_data.dropna().reindex()
    removeNan.to_csv("pm_2223_nomiss.csv", index=False)

# get_removeNan("../../data_pool/pm25/pm_2223_raw.csv ")

def linear_fill_data(raw_file):
    raw_data = pd.read_csv(raw_file, index_col=None)
    l_fill = raw_data.interpolate(method='linear')
    l_fill.to_csv("../../data_pool/model/data_lfill.csv ", index=False)

# linear_fill_data("../../data_pool/model/data_raw.csv ")

def set_timeslice_index(raw_file):
    raw_data = pd.read_csv(raw_file, index_col=None)
    raw_data['datetime'] = pd.to_datetime(raw_data[['year', 'month', 'day', 'hour']])
    raw_data.set_index('datetime', inplace=True)
    raw_data.to_csv(raw_file, index=True)
    
set_timeslice_index("../../data_pool/model/data_lfill.csv ")