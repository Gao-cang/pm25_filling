import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取时间区间文件
pos_df = pd.read_csv('pos_temporal_24.csv', parse_dates=['start_time', 'end_time'])
pos_df
# 创建输出目录
os.makedirs('plots', exist_ok=True)

# 新增：定义原始数据的路径和样式
ORIGIN_PATH = 'OriginData.csv'  # 假设原始数据文件名
ORIGIN_STYLE = {
    'color': '#333333',  # 深灰色
    'linewidth': 3,      # 3倍粗
    'alpha': 0.7,        # 半透明
    'label': 'OriginData'
}

# 遍历每个时间段
for idx, row in pos_df.iterrows():
    start_time = row['start_time']
    end_time = row['end_time']
    plot_title = f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"

    plt.figure(figsize=(6, 4))
 
    methods = ['1_bayes', '2_dcceof', '3_knn', '4_linear', '5_rf', '6_xgb']
    
    # 遍历所有填补文件
    for fill_method in methods:
        fill_path = f'{fill_method}.csv'
        try:
            # 读取数据并处理时间列
            df = pd.read_csv(
                fill_path,
                parse_dates=['datetime'],  # 假设时间列名为'time'
                usecols=['datetime', 'station_12']
            )
            
            # df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
            # df.set_index('time', inplace=True)
            # print(df.head())
            # 筛选时间区间
            mask = (df['datetime'] >= start_time-pd.Timedelta(hours=6)) & (df['datetime'] <= end_time+pd.Timedelta(hours=6))
            filtered = df.loc[mask]
            
            if not filtered.empty:
                plt.plot(
                    filtered['datetime'],
                    filtered['station_12'],
                    # marker='o',
                    markersize=3,
                    linewidth=1,
                    label=f'{fill_method}'
                )
        except Exception as e:
            print(f"Error processing {fill_path}: {str(e)}")
        # break
    
    # ==============================最后画原始数据
    try:
        origin_df = pd.read_csv(
            ORIGIN_PATH,
            parse_dates=['datetime'],
            usecols=['datetime', 'station_12']
        )
        mask = (origin_df['datetime'] >= start_time-pd.Timedelta(hours=6)) & (origin_df['datetime'] <= end_time+pd.Timedelta(hours=6))
        filtered_origin = origin_df.loc[mask]
        if not filtered_origin.empty:
            plt.plot(
                filtered_origin['datetime'],
                filtered_origin['station_12'],
                **ORIGIN_STYLE  # 应用特殊样式
            )
    except Exception as e:
        print(f"Error processing OriginData: {str(e)}")
    
    # 图表装饰
    
    plt.axvline(start_time, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(end_time, color='gray', linestyle='--', alpha=0.5)
    
    plt.title(plot_title)
    plt.xlabel('Timestamp')
    plt.ylabel('PM2.5 Value (station_12)')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 新增/修改的图例配置
    plt.legend(
        loc='lower left',
        fontsize=8,
        framealpha=0.5,
        frameon=True,
        edgecolor='gray',
        borderaxespad=0.5,
        handlelength=1.5
    )
    
    # 保存图表
    filename = f"plots/plot_{start_time.strftime('%Y%m%d_%H%M')}.png"
    plt.savefig(filename, dpi=150)
    plt.close()

print("所有图表已生成到plots目录")