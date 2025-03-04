
import pandas as pd
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS
from sklearn.impute import SimpleImputer
import jax.numpy as jnp
import jax.random as random

# 加载数据
data = pd.read_csv("../data_pool/data_raw.csv")
pm2_5_columns = [f"station_{i}" for i in range(1, 23)]
pm_matrix = data[pm2_5_columns].values

# 处理缺失值掩码
missing_mask = np.isnan(pm_matrix)
observed_values = pm_matrix[~missing_mask]

# 使用简单插补填充缺失值（仅用于初始化）
imp = SimpleImputer(strategy="mean")
pm_matrix_filled = imp.fit_transform(pm_matrix)

# 定义贝叶斯矩阵分解模型
def bayesian_matrix_factorization(data, rank):
    n_samples, n_features = data.shape
    U = numpyro.sample("U", dist.Normal(0, 1).expand([n_samples, rank]))
    V = numpyro.sample("V", dist.Normal(0, 1).expand([rank, n_features]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    recon = numpyro.deterministic("recon", jnp.dot(U, V))
    numpyro.sample("obs", dist.Normal(recon, sigma), obs=data)

# 使用变分推断
def run_svi(data, rank, num_steps=1000):
    guide = numpyro.infer.autoguide.AutoNormal(
        bayesian_matrix_factorization,
        create_plates=lambda *args, **kwargs: ()
    )
    optimizer = numpyro.optim.Adam(0.01)
    svi = SVI(bayesian_matrix_factorization, guide, optimizer, loss=Trace_ELBO())
    rng_key = random.PRNGKey(0)
    svi_result = svi.run(rng_key, num_steps, data, rank)
    params = svi_result.params
    return params

# 超参数设置
rank = 2  # 潜在因子数量
num_samples = 1000  # MCMC 样本数

# 初始化模型
params = run_svi(pm_matrix_filled, rank)

# 使用 MCMC 进行推断
nuts_kernel = NUTS(bayesian_matrix_factorization)
rng_key = random.PRNGKey(1)
mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=500)
mcmc.run(rng_key, pm_matrix_filled, rank)

# 提取后验样本
mcmc_samples = mcmc.get_samples()

# 计算填补值的均值
U_samples = mcmc_samples["U"]
V_samples = mcmc_samples["V"]
recon_samples = [jnp.dot(U, V) for U, V in zip(U_samples, V_samples)]
recon_mean = np.mean(recon_samples, axis=0)

# 更新原始数据
data[pm2_5_columns] = recon_mean

# 保存结果
data.to_csv("data_imputed.csv", index=False)