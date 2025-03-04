import pandas as pd
from sklearn.impute import KNNImputer

class knn_fill_test():
    def __init__(self):
        return
    
    def impute_test(self, all_data_csv):
        self.all_data = pd.read_csv(all_data_csv, index_col=0)
        self.filled_data = self.all_data.copy()
        
        imputer = KNNImputer(n_neighbors=5)
        
        # 使用 KNN Imputer 转换数据
        filled_data = imputer.fit_transform(self.filled_data)
        
        # 将填补后的数据转换为 DataFrame，保持原始列名
        self.filled_data = pd.DataFrame(filled_data, index=self.filled_data.index, columns=self.filled_data.columns)

        return self.filled_data

import pandas as pd
from sklearn.impute import KNNImputer

def impute_test(df, k=5):
    """
    使用 KNN 插值法填补缺失值。
    
    参数：
    - df: 包含数据的 Pandas DataFrame。
    - k: KNN 中的邻居数量，默认为 5。
    
    返回：
    - 填补后的 DataFrame。
    """
    # 创建 KNN Imputer 对象
    imputer = KNNImputer(n_neighbors=k)
    
    # 使用 KNN Imputer 转换数据
    filled_data = imputer.fit_transform(df)
    
    # 将填补后的数据转换为 DataFrame，保持原始列名
    filled_df = pd.DataFrame(filled_data, columns=df.columns)
    
    return filled_df