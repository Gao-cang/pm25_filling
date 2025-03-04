# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 21:21:53 2025

@author: Cang
"""

import pandas as pd

class linear_fill_test():
    def __init__(self):
        return
    
    def impute_test(self, all_data_csv):
        self.all_data = pd.read_csv(all_data_csv, index_col=0)
        self.filled_data = self.all_data.copy()
        
        # 使用线性插值填补缺失值
        self.filled_data = self.filled_data.interpolate(method='linear')

        return self.filled_data