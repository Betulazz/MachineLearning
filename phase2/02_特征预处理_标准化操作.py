"""
案例：演示特征预处理之标准化操作

特征工程的目的和步骤
    目的：
        利用专业的背景只是和技巧处理数据，用于提升模型的性能
    步骤
        1. 特征提取
        2. 特征预处理（归一化，标准化）
        3. 特征降维
        4. 特征选择
        5. 特征组合
特征预处理之标准化操作：
    目的：
        防止因为量纲问题，导致特征列的方差值较大，影响模型的最终结果
        所以通过公式将把各列的值映射到均值为0，标准差为1的正态分布序列
    公式：
        x' = （当前值-该列平均值）/该列的标准差 σ
    应用场景：
        适用于大数据集的处理
结论：
    无论是归一化还是标准化，目的都是为了解决因为量纲问题，导致模型评估较低等问腿
"""
from sklearn.preprocessing import StandardScaler  # 标准化对象

x_train = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]

transfer = StandardScaler()

x_train_new=transfer.fit_transform(x_train)

print('标准化后的数据集为：\n')
print(x_train_new)

print(f'数据集的均值为：{transfer.mean_}')
print(f'数据集的方差为：{transfer.var_}')
print(f'数据集的标准差为：{transfer.scale_}')