import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_clustered_barchart(df, type_algorithm):
    # 固定簇的顺序
    order = [
        'random', 'time_2', 'time_3', 'time_6', 'time_12', 'time_24',
        'spatemp_2', 'spatemp_3', 'spatemp_6', 'spatemp_12', 'spatemp_24'
    ]
    
    # 按指定顺序排序数据
    df['Unnamed: 0'] = pd.Categorical(df['Unnamed: 0'], categories=order, ordered=True)
    df = df.sort_values('Unnamed: 0')
    
    # 处理r2: 负值设为0
    df['r2'] = df['r2'].apply(lambda x: max(x, 0))
    
    # 创建画布和左侧坐标轴
    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = np.arange(len(order))
    width = 0.25  # 每个柱子的宽度
    
    # 绘制MAE和RMSE（左侧轴）
    ax1.bar(x - width, df['mae'], width, label='MAE', color='blue')
    ax1.bar(x, df['rmse'], width, label='RMSE', color='coral')
    
    # 设置左侧轴标签
    ax1.set_xlabel('Missing Data', fontsize=14)
    ax1.set_ylabel('MAE / RMSE', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=45, ha='right', fontsize=10)
    
    # 创建右侧坐标轴
    ax2 = ax1.twinx()
    
    # 绘制R²（右侧轴）
    ax2.bar(x + width, df['r2'], width, label='R²', color='green')
    
    # 设置右侧轴范围和标签
    ax2.set_ylabel('R²', fontsize=14)
    ax2.set_ylim(0, 1.0)  # 根据R²范围调整
    
    # 合并图例
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # 调整布局并保存
    plt.title(f'Performance of {type_algorithm}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{type_algorithm}.png', dpi=300)
    plt.show()

# 示例：加载数据并绘制柱状图
if __name__ == "__main__":
    for type_algorithm in ['Avg', 'Medium', 'H_avg', 'GAN', 'Linear', 'Lagrange', 'Knn', 'RF', 'XGBoost', 'Bayes', 'DccEof']:
        # 假设 'filled_test_records.xlsx' 是你的实验结果文件
        df = pd.read_excel('填补实验记录.xlsx', sheet_name=type_algorithm, engine='openpyxl')
        
        # 调用函数，保存到 'plots' 文件夹中
        
        plot_clustered_barchart(df, type_algorithm)