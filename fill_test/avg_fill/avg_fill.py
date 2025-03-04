import pandas as pd

class avg_fill_test():
    def __init__(self):
        return
    
    def impute_test(self, all_data_csv):
        self.all_data = pd.read_csv(all_data_csv, index_col=0)
        self.filled_data = self.all_data.copy()
        
        for col in self.filled_data.columns:
            self.filled_data[col] = self.filled_data[col].fillna(self.filled_data[col].mean())

        return self.filled_data

        
        