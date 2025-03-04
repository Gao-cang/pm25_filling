import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class GANImputer:
    def __init__(self, num_epochs=500, batch_size=32, latent_dim=100, lr=0.0001):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.lr = lr
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化生成器和判别器
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # 定义优化器
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        self.loss_fn = nn.BCELoss()

    def _preprocess(self, data):
        """数据预处理：标准化并用简单方法填充缺失值"""
        # 初始填充
        initial_imputer = IterativeImputer(max_iter=10, random_state=0)
        data_filled = initial_imputer.fit_transform(data)
        # 标准化
        data_scaled = self.scaler.fit_transform(data_filled)
        return torch.FloatTensor(data_scaled).to(self.device)

    def _create_mask(self, data):
        """创建缺失值掩码"""
        return torch.isnan(torch.FloatTensor(data.values)).float().to(self.device)

    def train(self, X):
        """训练GAN"""
        X_tensor = self._preprocess(X)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            d_losses = []
            g_losses = []
            for real_data in dataloader:
                real_data = real_data[0]
                batch_size = real_data.size(0)

                # 训练判别器
                self.optimizer_D.zero_grad()
                # 生成假数据
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_data = self.generator(z)
                # 计算判别器损失
                real_loss = self.loss_fn(self.discriminator(real_data), torch.ones(batch_size, 1).to(self.device))
                fake_loss = self.loss_fn(self.discriminator(fake_data.detach()), torch.zeros(batch_size, 1).to(self.device))
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.optimizer_D.step()

                # 训练生成器
                self.optimizer_G.zero_grad()
                # 生成假数据并计算对抗损失
                fake_data = self.generator(z)
                g_loss = self.loss_fn(self.discriminator(fake_data), torch.ones(batch_size, 1).to(self.device))
                g_loss.backward()
                self.optimizer_G.step()
                
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

            # 每个epoch打印损失
            print(f"Epoch [{epoch+1}/{self.num_epochs}], D Loss: {np.mean(d_losses):.4f}, G Loss: {np.mean(g_losses):.4f}")

    def impute_test(self, filename):
        """执行数据填充"""
        raw_data = pd.read_csv(filename, index_col=0).iloc[:, 13:]
        mask = self._create_mask(raw_data)
        # 初始填充并标准化
        data_preprocessed = self._preprocess(raw_data)
        # 生成完整数据
        with torch.no_grad():
            z = torch.randn(data_preprocessed.size(0), self.latent_dim).to(self.device)
            generated_data = self.generator(z)
        # 反标准化
        filled_data = self.scaler.inverse_transform(generated_data.cpu().numpy())
        # 仅替换原始缺失值
        final_data = raw_data.values.copy()
        final_data[mask.cpu().numpy().astype(bool)] = filled_data[mask.cpu().numpy().astype(bool)]
        
        
        return pd.DataFrame(final_data, index=raw_data.index, columns=raw_data.columns)

# 定义生成器和判别器网络结构
class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=22):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim=22):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# 使用示例
if __name__ == "__main__":
    imputer = GANImputer(num_epochs=50)
    imputer.train(pd.read_csv("../../data_pool/model/data_lfill.csv", index_col=0).iloc[:, 13:])
    filled_data = imputer.impute_test("../../data_pool/model/data_lfill.csv")
    print(filled_data)