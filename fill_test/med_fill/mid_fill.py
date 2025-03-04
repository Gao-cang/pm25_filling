import pandas as pd

class mid_fill_test():
    def __init__(self):
        return
    
    def get_filled_data(self, all_data_csv):
        self.all_data = pd.read_csv(all_data_csv, index_col=0)
        
        self.filled_data = self.all_data.copy()
        
        
        self.filled_data = self.filled_data.fillna(self.filled_data.median())

        return self.filled_data