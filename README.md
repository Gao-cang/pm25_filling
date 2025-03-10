# 项目介绍
## 研究目的
1. 分析PM2.5小时数据的特征，重点是缺失值模型
2. 对比从文献中提取出来的现有填补方法
3. 基于对比得到的最优方法，优化得到一个更好的填补模型

## 研究框架
1. 分析PM2.5数据的特征
	1. 数据分布
	2. 时序变化
	3. 相关性
	4. 缺失情况：缺失比例、缺失模式

2. 使用不同缺失值填补方法测试填补准确率
	1. 构造不同类型的缺失数据
	2. 对比不同缺失值填补方法在不同数据上的效果好坏
	3. 评选出针对不同缺失数据，最优的填补方法
	4. 整合为一套缺失值“监测-匹配算法-进行填补”的缺失数据填充模型

## 研究内容

### 数据分析部分
1. 数据分布
2. 时序变化
3. 相关性
4. 缺失值情况
   1. 随机缺失
   2. 时间连续性缺失
   3. 时空聚集性缺失

### 不同填充算法对比部分
1. 对比方法
   1. 构造“模拟缺失数据”
   2. 在“模拟缺失数据”上进行填补，分析误差、准确率和耗时等指标
2. 11种提炼出的基础填补方法
   1. 均值
   2. 中位数
   3. 小时均值
   4. 线性插值
   5. KNN
   6. 拉格朗日插值
   7. 随机森林
   8. XGBoost
   9. DCCEOF
   10. GAN
   11. Bayes矩阵分解
3. 针对3种缺失值类型，对比相对最优的方法，以及改进空间

### 基于混合方法的缺失值填充模型
1. 整体设计
   1. 缺失值检测（位置检测+类型检测）
   2. 分别对3种类型的缺失值，用最优的方法填补
2. 具体方法选择
   1. 随机缺失：线性插值
   2. 时间连续性缺失：参考邻近站点信息的双向XGBoost
   3. 时空聚集性缺失：双向XGBoost
