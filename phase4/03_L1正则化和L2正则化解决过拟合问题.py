"""
案例：
    演示欠拟合，正好拟合，L1正则化，L2正则化的效果图
问题：
    欠拟合：模型在训练集和测试集的效果都不好
    正好拟合：模型在测试集和测试集表现效果更好
    过拟合：模型在训练集表现好，测试集表现不好
过拟合，欠拟合解释：
    产生原因：
        欠拟合：模型简单。
        过拟合：模型复杂。
    解决方案：
        欠拟合：增加特征，从而增加模型的复杂度。
        过拟合：减少模型复杂度，手动筛选（减少）特征，L1和L2正则化。
L1和L2正则化介绍：
    目的/思路：
        都是基于惩罚系数来修改（特征列的）权重的，惩罚系数越大，则修改力度就越大，对应的权重就越小。
    区别：
        L1正则化，可以实现让权重变为0，从而达到特征选择的目的。
        L2正则化，只能让权重无限趋近于0，但是不能为0.
    大白话：
        我要去爬山，带了个小包，装了：登山杖，水，面包，衣服，雨伞，鞋子...发现包装不下了。
        L1正则化：可以实现去掉一些不是必选的，例如：当天去，当前回，且天气晴朗→不带雨伞，鞋子，即：权重为0
        L2正则化：换一个非常非常大的包，还是那些物品，但是空间占用（权重）就变小了...
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.linear_model import LinearRegression, Lasso  # 正规方程的回归类型
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error  # 均方误差估计
from sklearn.linear_model import Ridge, RidgeCV


# 定义函数模拟欠拟合
def under_fitting():
    # 加载数据
    np.random.seed(23)
    x = np.random.uniform(-3, 3, 100)  # 参数：最小值 最大值 生成个数
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)  # 参数：平均值 标准值 生成个数
    # print(x)
    # print(y)

    # 数据预处理，把x转为1列
    X = x.reshape(-1, 1)
    # print(X)

    # 特征工程，此处不需要

    # 模型训练
    estimator = LinearRegression()  # 正规方程线性回归
    estimator.fit(X, y)

    # 模型预测
    y_pre = estimator.predict(X)

    # 模型评估
    print(f'均方误差：{mean_squared_error(y, y_pre)}')

    # 绘图
    plt.scatter(x, y)  # 散点图
    plt.plot(x, y_pre, color='red')  # 折线图
    plt.show()


# 正好拟合
def just_fitting():
    # 加载数据
    np.random.seed(23)
    x = np.random.uniform(-3, 3, 100)  # 参数：最小值 最大值 生成个数
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)  # 参数：平均值 标准值 生成个数
    # print(x)
    # print(y)

    # 数据预处理，把x转为1列
    X = x.reshape(-1, 1)
    # print(X)
    # 因为特征列只有1列，模型过于简单，欠拟合，增加1列特征列，增加模型的复杂度
    X2 = np.hstack([X, X ** 2])  # 该函数作用：横向拼接，拼接两个数组，拼接后行数不变

    # 特征工程，此处不需要

    # 模型训练
    estimator = LinearRegression()  # 正规方程线性回归
    estimator.fit(X2, y)

    # 模型预测
    y_pre = estimator.predict(X2)

    # 模型评估
    print(f'均方误差：{mean_squared_error(y, y_pre)}')

    # 绘图
    plt.scatter(x, y)  # 散点图
    # np.sort(x):对x轴特征进行排序，默认升序
    # np.argsort(x)：对x轴排序，返回排序后的索引
    plt.plot(np.sort(x), y_pre[np.argsort(x)], color='red')  # 折线图
    plt.show()


def over_fitting():
    # 加载数据
    np.random.seed(23)
    x = np.random.uniform(-3, 3, 100)  # 参数：最小值 最大值 生成个数
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)  # 参数：平均值 标准值 生成个数
    # print(x)
    # print(y)

    # 数据预处理，把x转为1列
    X = x.reshape(-1, 1)
    # print(X)
    # 因为特征列只有1列，模型过于简单，欠拟合，增加9列特征列，增加模型的复杂度
    # 该函数作用：横向拼接，拼接多个数组，拼接后行数不变
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])

    # 特征工程，此处不需要

    # 模型训练
    estimator = LinearRegression()  # 正规方程线性回归
    estimator.fit(X3, y)

    # 模型预测
    y_pre = estimator.predict(X3)

    # 模型评估
    print(f'均方误差：{mean_squared_error(y, y_pre)}')

    # 绘图
    plt.scatter(x, y)  # 散点图
    # np.sort(x):对x轴特征进行排序，默认升序
    # np.argsort(x)：对x轴排序，返回排序后的索引
    plt.plot(np.sort(x), y_pre[np.argsort(x)], color='red')  # 折线图
    plt.show()


def l1_regularization():
    # 加载数据
    np.random.seed(23)
    x = np.random.uniform(-3, 3, 100)  # 参数：最小值 最大值 生成个数
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)  # 参数：平均值 标准值 生成个数
    # print(x)
    # print(y)

    # 数据预处理，把x转为1列
    X = x.reshape(-1, 1)
    # print(X)
    # 因为特征列只有1列，模型过于简单，欠拟合，增加9列特征列，增加模型的复杂度
    # 该函数作用：横向拼接，拼接多个数组，拼接后行数不变
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])

    # 特征工程，此处不需要

    # 模型训练
    # estimator = LinearRegression()  # 正规方程线性回归
    # 创建L1正则化对象
    estimator = Lasso(alpha=0.1)  # alpha 正则化系数（惩罚系数） 默认是1
    estimator.fit(X3, y)

    # 模型预测
    y_pre = estimator.predict(X3)

    # 模型评估
    print(f'均方误差：{mean_squared_error(y, y_pre)}')

    # 绘图
    plt.scatter(x, y)  # 散点图
    # np.sort(x):对x轴特征进行排序，默认升序
    # np.argsort(x)：对x轴排序，返回排序后的索引
    plt.plot(np.sort(x), y_pre[np.argsort(x)], color='red')  # 折线图
    plt.show()


def l2_regularization():
    # 加载数据
    np.random.seed(23)
    x = np.random.uniform(-3, 3, 100)  # 参数：最小值 最大值 生成个数
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)  # 参数：平均值 标准值 生成个数
    # print(x)
    # print(y)

    # 数据预处理，把x转为1列
    X = x.reshape(-1, 1)
    # print(X)
    # 因为特征列只有1列，模型过于简单，欠拟合，增加9列特征列，增加模型的复杂度
    # 该函数作用：横向拼接，拼接多个数组，拼接后行数不变
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])

    # 特征工程，此处不需要

    # 模型训练
    # estimator = LinearRegression()  # 正规方程线性回归
    # 创建L1正则化对象
    # estimator = Lasso(alpha=0.1)  # alpha 正则化系数（惩罚系数） 默认是1
    # 创建L2正则化对象
    estimator = Ridge(alpha=10)
    estimator.fit(X3, y)

    # 模型预测
    y_pre = estimator.predict(X3)

    # 模型评估
    print(f'均方误差：{mean_squared_error(y, y_pre)}')

    # 绘图
    plt.scatter(x, y)  # 散点图
    # np.sort(x):对x轴特征进行排序，默认升序
    # np.argsort(x)：对x轴排序，返回排序后的索引
    plt.plot(np.sort(x), y_pre[np.argsort(x)], color='red')  # 折线图
    plt.show()


if __name__ == '__main__':
    # under_fitting()
    # just_fitting()
    # over_fitting()
    # l1_regularization()
    l2_regularization()