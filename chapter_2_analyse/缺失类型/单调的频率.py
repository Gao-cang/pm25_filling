import matplotlib.pyplot as plt
import numpy as np


def plot_monotonic_frequency(data_dict):
    # 配置中文字体（如系统无中文字体需注释此段）
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 数据解构与排序
    n_values = sorted(data_dict.keys())
    frequencies = [data_dict[n] for n in n_values]

    # 创建画布
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)

    # 绘制柱状图
    bars = ax.bar(n_values, frequencies,
                  width=0.5,
                  color='#2c7fb8',
                  # edgecolor='black',
                  alpha=0.8)

    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}',
                ha='center', va='bottom',
                fontsize=9)

    # 设置边框
    ax.spines['top'].set_visible(False)  # 移除顶部边框
    ax.spines['right'].set_visible(False)  # 移除右侧边框

    # 设置坐标轴
    ax.set_xlabel('连续监测值数量 (n)', fontsize=12, labelpad=10)
    ax.set_ylabel('单调频率', fontsize=12, labelpad=10)
    ax.set_title('连续n个监测值数量的单调频率', fontsize=14, pad=15)

    # 设置刻度
    ax.set_xticks(np.arange(min(n_values), max(n_values) + 1, 1))
    ax.tick_params(axis='both', which='major', labelsize=10)

    # 设置网格
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # 调整布局
    plt.tight_layout()

    # 保存并显示
    plt.savefig('monotonic_frequency.png', bbox_inches='tight')
    plt.show()


# 测试数据
data = {3: 0.68, 4: 0.44, 5: 0.29, 6: 0.2,
        7: 0.14, 8: 0.11, 9: 0.09, 10: 0.07,
        11: 0.06, 12: 0.06}

# 执行绘图
plot_monotonic_frequency(data)