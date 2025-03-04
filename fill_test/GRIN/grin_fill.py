import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from math import radians, sin, cos, sqrt, atan2

class TimeSeriesDataset(Dataset):
    def __init__(self, data, missing_mask):
        self.data = data
        self.mask = missing_mask  # 表示数据是否缺失的掩码

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx]

class GRIN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, adj_matrix, num_nodes):
        super(GRIN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.adj_matrix = adj_matrix
        self.num_nodes = num_nodes  # Add this line
  
        # Graph convolution layer (node-wise)
        self.graph_conv = nn.Linear(input_size // num_nodes, input_size // num_nodes)  # Divide input_size by num_nodes to get node_features
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, input_size)  # Bidirectional GRU has double the hidden size

    def forward(self, x):
        # Reshape input data to separate node features
        batch_size = x.size(0)
        node_features = x.size(-1) // self.num_nodes  # Number of features per node
        x_reshaped = x.view(batch_size, self.num_nodes, node_features)
        
        # Graph convolution
        x_conv = self.graph_conv(x_reshaped)
        
        # Aggregate neighbor information using adjacency matrix
        x_aggregated = torch.matmul(self.adj_matrix, x_conv)
        # Reshape back to input size
        x_aggregated = x_aggregated.view(batch_size, -1)
        
        # Prepare input for GRU: [batch_size, sequence_length, input_size]
        # If there is no time dimension, add a dummy sequence length of 1
        x_gru_input = x_aggregated.unsqueeze(1)  # Shape [batch_size, 1, input_size]
        
        # Bidirectional GRU
        out, _ = self.gru(x_gru_input)
        # Output shape: [batch_size, sequence_length, hidden_size * 2]
        out = out.squeeze(1)  # Remove sequence_length dimension
        
        # Feed into fully connected layer
        out = self.fc(out)
        return out


def build_adjacency_matrix(station_coords, threshold=0.1):
    # 计算各站点之间的距离权重
    num_stations = len(station_coords)
    adj_matrix = np.zeros((num_stations, num_stations))
    for i in range(num_stations):
        for j in range(num_stations):
            if i == j:
                continue
            # 计算两站点之间的距离
            lat1, lon1 = station_coords[i]
            lat2, lon2 = station_coords[j]
            # Haversine公式计算距离（单位：千米）
            R = 6371.0  # 地球半径
            lat1_rad = radians(lat1)
            lon1_rad = radians(lon1)
            lat2_rad = radians(lat2)
            lon2_rad = radians(lon2)
            dlon = lon2_rad - lon1_rad
            dlat = lat2_rad - lat1_rad
            a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            distance = R * c
            # 距离越近，权重越大
            weight = np.exp(-distance / threshold)  # 使用高斯核函数
            adj_matrix[i][j] = weight
    # 归一化邻接矩阵
    D = np.diag(np.sum(adj_matrix, axis=1))
    D_inv = np.linalg.inv(D)
    adj_normalized = np.matmul(D_inv, adj_matrix)
    return torch.tensor(adj_normalized, dtype=torch.float32)

def preprocess_data(to_ai_csv, station_lng_lat_csv):
    # 加载数据
    time_series_data = pd.read_csv(to_ai_csv, index_col=0)
    stations = pd.read_csv(station_lng_lat_csv)
    
    # 提取气象特征和站点监测值
    time_features = time_series_data[['is_weekday', 'season', 'condition', 
                                      'temperature', 'rel_hum', 'wind_speed', 
                                      'wind_dir', 'pressure', 'precipitation']]
    station_values = time_series_data.filter(like='station_').values
    
    # 构建邻接矩阵
    station_coords = stations[['sta_lat', 'sta_lng']].values
    adj_matrix = build_adjacency_matrix(station_coords)
    
    # 处理缺失值
    missing_mask = np.isnan(station_values)
    station_values[missing_mask] = 0.0  # 将缺失值暂时填充为0
    
    # 合并时间和站点数据
    # 这里假设时间特征和站点监测值需要某种方式的结合，这里简单地将时间特征作为全局特征
    # 由于时间特征是全局的，需要重复为每个站点使用
    time_features_expanded = np.expand_dims(time_features.values, axis=1)  # [batch, 1, time_features_dim]
    time_features_expanded = np.tile(time_features_expanded, (1, station_values.shape[1], 1))  # [batch, stations, time_features_dim]
    
    # 将数据重塑为 [batch, stations * (n_features + time_features_dim)]
    data = np.concatenate([station_values.reshape(station_values.shape[0], -1, 1), 
                           time_features_expanded], axis=-1)
    data = data.reshape(data.shape[0], -1)  # 展平为二维
    
    # 将数据和掩码转换为 torch.Tensor，并指定类型
    data = torch.tensor(data, dtype=torch.float32)
    missing_mask = torch.tensor(missing_mask, dtype=torch.bool)
    
    
    return data, missing_mask, adj_matrix


def train_grin(data, missing_mask, adj_matrix, num_nodes=22, hidden_size=100, 
               num_layers=2, num_epochs=100, batch_size=32):
    # Create data loader
    dataset = TimeSeriesDataset(data, missing_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    

    input_size = data.shape[1]  # Features per time step
    
    # Initialize model
    model = GRIN(input_size=input_size, hidden_size=hidden_size, 
                 num_layers=num_layers, adj_matrix=adj_matrix, num_nodes=num_nodes)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch, (x, mask) in enumerate(dataloader):
            x = x
            mask = mask
            
            # Forward pass
            outputs = model(x)
            # Compute loss using only non-missing values
            loss = criterion(outputs[mask], x[mask])
            total_loss += loss.item()
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / (batch+1):.4f}')
    
    return model

def impute_grin(model, data, missing_mask):
    model.eval()
    data = torch.tensor(data, dtype=torch.float32)
    missing_mask_tensor = torch.tensor(missing_mask, dtype=torch.bool)
    
    with torch.no_grad():
        imputed_data = model(data)
        # 用真实值覆盖非缺失部分，用预测值填充缺失部分
        imputed_values = data.clone()
        imputed_values[missing_mask_tensor] = imputed_data[missing_mask_tensor]
    
    return imputed_values.cpu().numpy()

if __name__ == '__main__':
    # 数据路径
    to_ai_csv = '../../data_pool/model/data_raw.csv'
    station_lng_lat_csv = '../../data_pool/model/station_lng_lat.csv'
    
    # 数据预处理
    data, missing_mask, adj_matrix = preprocess_data(to_ai_csv, station_lng_lat_csv)
    
    # 超参数
    hidden_size = 100
    num_layers = 2
    num_epochs = 100
    batch_size = 32
    
    # 训练模型
    model = train_grin(data, missing_mask, adj_matrix, hidden_size=hidden_size, 
                       num_layers=num_layers, num_epochs=num_epochs, batch_size=batch_size)
    
    # 预测缺失值
    imputed_data = impute_grin(model, data, missing_mask)
    
    # 输出结果
    print("Imputed data shape:", imputed_data.shape)
    # 您可以将 imputed_data 保存为 CSV 或其他格式
