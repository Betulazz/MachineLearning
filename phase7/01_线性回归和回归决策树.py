"""
    案例:
        演示线性回归和回归决策树(CART)对比.
    细节：
        CART分类回归决策树，既可以做分类，也可以做回归，一般做：分类。
        做分类是采用基尼值，做回归时采用平方损失（类似于最小二乘）
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor  # 回归决策树
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 准备数据
x_train = np.array(list(range(1, 11))).reshape(-1, 1)
y_train = np.array([5.56, 5.7, 5.91, 6.4, 6.8, 7.05, 8.9, 8.7, 9, 9.05])
print(x_train)
print(y_train)

estimator1 = LinearRegression()
estimator2 = DecisionTreeRegressor(max_depth=1)
estimator3 = DecisionTreeRegressor(max_depth=3)

estimator1.fit(x_train, y_train)
estimator2.fit(x_train, y_train)
estimator3.fit(x_train, y_train)

x_test = np.arange(0, 10, 0.1).reshape(-1, 1)
print(x_test)

y_pred1 = estimator1.predict(x_test)
y_pred2 = estimator2.predict(x_test)
y_pred3 = estimator3.predict(x_test)

print(f'预测结果（线性回归）：{y_pred1}')
print(f'预测结果（决策树深度为1）：{y_pred2}')
print(f'预测结果（决策树深度为3）：{y_pred3}')

plt.scatter(x_train, y_train)
plt.plot(x_test, y_pred1, label='linear regression')
plt.plot(x_test, y_pred2, label='max depth=1')
plt.plot(x_test, y_pred3, label='max depth=3')
plt.legend()
plt.xlabel('data')
plt.ylabel('target')
plt.show()
