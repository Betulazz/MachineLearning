from markupsafe import escape_silent
from scipy.linalg.interpolative import estimate_rank
from sklearn.preprocessing import StandardScaler  # 特征处理
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.linear_model import LinearRegression  # 正规方程的回归类型
from sklearn.linear_model import SGDRegressor  # 梯度下降的回归模型
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error  # 均方误差估计
from sklearn.linear_model import Ridge, RidgeCV

import pandas as pd
import numpy as np

# 加载数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])  # hstack():水平拼接数组
target = raw_df.values[1::2, 2]
# print(f'特征：{data.shape}')  # (506, 13)
# print(f'标签：{target.shape}')  # (506,)
#
# print(f'特征数据：{data[:5]}')
# print(f'标签数据：{target[:5]}')

# 数据预处理
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=23)

# 特征工程
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 模型训练
estimator = SGDRegressor(fit_intercept=True, learning_rate='constant', eta0=0.01)
estimator.fit(x_train,y_train)

print(f'weight:{estimator.coef_}')
print(f'bias:{estimator.intercept_}')

# 模型预测
y_pre= estimator.predict(x_test)

# 模型评估
print(f'均方误差为：{mean_squared_error(y_test, y_pre)}')
print(f'均方根误差为：{root_mean_squared_error(y_test, y_pre)}')
print(f'平均绝对误差为：{mean_absolute_error(y_test, y_pre)}')

