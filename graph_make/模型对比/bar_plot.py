import pandas as pd
import matplotlib.pyplot as plt

def plot_bar_chart(data):
    """
    绘制柱状图，数据的第一列作为标签，第二列作为值。
    
    参数:
        data (pd.DataFrame): 输入的二维 Pandas 数组，第一列是标签，第二列是值。
    
    返回:
        None
    """
    # 检查数据是否为空或不符合条件
    if data.empty:
        print("数据为空")
        return
    if data.shape[1] < 2:
        print("数据需要至少两列")
        return
    
    # 获取标签和值
    labels = data.iloc[:, 0]
    values = data.iloc[:, 1]
    
    # 创建柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    
    # 添加标题和标签
    plt.title("柱状图")
    plt.xlabel("标签")
    plt.ylabel("值")
    
    # 添加每个柱子上方的数据值
    for i, v in enumerate(values):
        plt.text(i, v + 0.1, str(v), ha='center')
    
    # 显示图表
    plt.show()

# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    data = pd.read_excel("../")
    
    # 调用函数绘制柱状图
    plot_bar_chart(data)