import pandas as pd
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS
from numpyro.infer.autoguide import AutoNormal
from sklearn.impute import SimpleImputer
import jax.numpy as jnp
import jax.random as random

class BMF_imputer():
    def __init__(self, rank=2, num_samples=1000):
        self.rank = rank  # 潜在因子数量
        self.num_samples = num_samples  # MCMC 样本数
        self.COLUMNS = []  # 数据列名
        self.pm_matrix = None  # 原始矩阵
        self.pm_matrix_filled = None  # 填充后的矩阵
        self.recon_mean = None  # 预测矩阵均值
        
    def _train_svi(self, data):
        """ 使用变分推断训练模型 """
        guide = AutoNormal(
            bayesian_matrix_factorization,
            create_plates=lambda *args, **kwargs: ()
        )
        optimizer = numpyro.optim.Adam(0.01)
        svi = SVI(bayesian_matrix_factorization, guide, optimizer, loss=Trace_ELBO())
        rng_key = random.PRNGKey(0)
        svi_result = svi.run(rng_key, num_steps=5000, data=data, rank=self.rank)
        return svi_result.params

    def impute_test(self, data_csv):
        """ 主填充方法，返回填补后的矩阵 """
        # 加载数据
        self.data = pd.read_csv(data_csv, index_col=0)
        self.COLUMNS = self.data.columns.tolist()
        
        # 选择需要填充的列（假设以 "station_" 开头）
        self.COLUMNS = [col for col in self.COLUMNS if 'station' in col.lower()]
        self.pm_matrix = self.data[self.COLUMNS].values
        
        # 处理缺失值掩码
        self.missing_mask = np.isnan(self.pm_matrix)
        observed_values = self.pm_matrix[~self.missing_mask]

        # 使用简单插补填充缺失值（仅用于初始化）
        imp = SimpleImputer(strategy="mean")
        self.pm_matrix_filled = imp.fit_transform(self.pm_matrix)

        # 使用变分推断初始化参数
        self.params = self._train_svi(self.pm_matrix_filled)

        # 使用 MCMC 进行推断
        nuts_kernel = NUTS(bayesian_matrix_factorization)
        rng_key = random.PRNGKey(1)
        mcmc = MCMC(nuts_kernel, num_samples=self.num_samples, num_warmup=500)
        mcmc.run(rng_key, self.pm_matrix_filled, self.rank)

        # 提取后验样本
        mcmc_samples = mcmc.get_samples()

        # 计算填补值的均值
        U_samples = mcmc_samples["U"]
        V_samples = mcmc_samples["V"]
        recon_samples = [jnp.dot(U, V) for U, V in zip(U_samples, V_samples)]
        self.recon_mean = np.nanmean(recon_samples, axis=0)

        # 更新原始数据
        data_imputed = self.data.copy()
        data_imputed[self.COLUMNS] = self.recon_mean
        
        
        
        return data_imputed

# 定义贝叶斯矩阵分解模型
def bayesian_matrix_factorization(data, rank):
    n_samples, n_features = data.shape
    U = numpyro.sample("U", dist.Normal(0, 1).expand([n_samples, rank]))
    V = numpyro.sample("V", dist.Normal(0, 1).expand([rank, n_features]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    recon = numpyro.deterministic("recon", jnp.dot(U, V))
    numpyro.sample("obs", dist.Normal(recon, sigma), obs=data)