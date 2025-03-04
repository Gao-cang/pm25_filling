import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_correlation_matrices(df):
    """
    计算相关系数与p值矩阵

    参数：
    df : pandas DataFrame
        输入数据（自动过滤非数值列）

    返回：
    corr_df : 相关系数矩阵 DataFrame
    p_df : p值矩阵 DataFrame
    """
    numeric_df = df.select_dtypes(include=[np.number])
    cols = numeric_df.columns
    n = len(cols)

    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i <= j:  # 避免重复计算
                corr, p = stats.pearsonr(numeric_df.iloc[:, i], numeric_df.iloc[:, j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr  # 对称填充
                p_matrix[i, j] = p
                p_matrix[j, i] = p

    return (
        pd.DataFrame(corr_matrix, index=cols, columns=cols),
        pd.DataFrame(p_matrix, index=cols, columns=cols)
    )


def plot_correlation_heatmap(corr_df, p_df, save_path=None, figsize=(10, 8)):
    """
    根据矩阵绘制热力图

    参数：
    corr_df : 相关系数矩阵 DataFrame
    p_df    : p值矩阵 DataFrame
    save_path: 图片保存路径（可选）
    figsize : 图像尺寸
    """
    plt.figure(figsize=figsize)
    # 绘制热力图
    heatmap = sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        mask=np.triu(np.ones_like(corr_df, dtype=bool)),
        cbar_kws={"label": "Pearson Correlation"}
    )

    # 添加显著性标记
    n = len(corr_df.columns)
    for i in range(n):
        for j in range(n):
            if i < j:  # 只在下三角显示
                p = p_df.iloc[i, j]
                star = ''
                if p < 0.001:
                    star = '***'
                elif p < 0.01:
                    star = '**'
                elif p < 0.05:
                    star = '*'
                heatmap.text(
                    j + 0.5, i + 0.5, star,
                    ha='center', va='center',
                    color='black' if abs(corr_df.iloc[i, j]) < 0.5 else 'white',  # 自动调整文字颜色
                    fontsize=10
                )

    plt.title("Pearson Correlation Matrix with Significance Levels")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def calculate_correlation_matrix(df):
    """
    计算数值型列的Pearson相关系数矩阵

    参数：
    df : pandas DataFrame
        输入数据（自动过滤非数值列）

    返回：
    corr_df : 相关系数矩阵 DataFrame
    """
    # 过滤非数值列
    numeric_df = df.select_dtypes(include=[np.number])

    # 计算相关系数矩阵
    corr_matrix = numeric_df.corr(method='pearson')

    return corr_matrix


def plot_correlation_heatmap(corr_matrix,
                             save_path=None,
                             figsize=(10, 8),
                             annot=True,
                             cmap="coolwarm"):
    """
    绘制相关系数矩阵热力图

    参数：
    corr_matrix : 相关系数矩阵 DataFrame
    save_path   : 图片保存路径（可选）
    figsize     : 图像尺寸
    annot       : 是否显示数值
    cmap        : 颜色映射
    """
    plt.figure(figsize=figsize)

    # 创建掩码隐藏上三角
    # mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # 自定义颜色映射
    if cmap == "custom":
        cmap = LinearSegmentedColormap.from_list(
            "custom_cmap", ["#2b83ba", "#ffffbf", "#d7191c"], N=256
        )

    # 绘制热力图
    sns.heatmap(
        corr_matrix,
        # mask=mask,
        annot=annot,
        annot_kws={"fontsize": 8},
        fmt=".2f",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"}
    )

    # 美化图形
    plt.title("Pearson Correlation Matrix", pad=20, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    pm_data = pd.read_csv("../../data_pool/pm25/pm_2223_linear_fill.csv")
    meteo_data = pd.read_csv("../../data_pool/meteo/meteo_2223_clean.csv")
    data = pd.merge(left=pm_data, right=meteo_data, on=['year', 'month', 'day', 'hour'])
    data = data[
        ['year', 'month', 'day', 'hour', 'is_weekday', 'season', 'condition', 'temperature', 'rel_hum', 'wind_speed',
         'wind_dir', 'pressure', 'precipitation', 'station_1', 'station_2', 'station_3', 'station_4', 'station_5',
         'station_6', 'station_7', 'station_8', 'station_9', 'station_10', 'station_11', 'station_12', 'station_13',
         'station_14', 'station_15', 'station_16', 'station_17', 'station_18', 'station_19', 'station_20', 'station_21',
         'station_22']]
    # print(data.columns)

    # 计算矩阵
    corr_matrix = calculate_correlation_matrix(data)

    # 绘制并保存热力图
    plot_correlation_heatmap(
        corr_matrix.iloc[13:,13:],
        save_path="correlation_heatmap.png",
        figsize=(10, 8),
        annot=True,
        cmap="coolwarm"  # 使用 "custom" 启用自定义颜色映射
    )
