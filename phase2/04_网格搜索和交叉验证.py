"""
交叉验证解释：
    原理：
        把数据分成n份，例如分成：4份
        第一次：把第一份数据作为验证集（测试集），其他作为训练集，训练模型，模型预测，获取：准确率1
        第二次：把第二份数据作为验证集（测试集），其他作为训练集，训练模型，模型预测，获取：准确率2
        第三次：把第三份数据作为验证集（测试集），其他作为训练集，训练模型，模型预测，获取：准确率3
        第四次：把第四份数据作为验证集（测试集），其他作为训练集，训练模型，模型预测，获取：准确率4

        假设第四次准确率最高，则用全部数据训练模型，再使用（第四次的）测试集对模型测试
    目的：
        为了让模型的最终验证结果更准确
网格搜索：
    目的：
        寻找最优超参
    原理：
        接受超参可能出现的值，然后针对于超参的每个值进行交叉验证，获取到最优超参组合
    超参数：
        需要用户手动录入的数据，不同的超参（组合），可能会影响模型的最终评测结果
解释：
    网格搜索+交叉验证，本质上指的是GridSearchCV()这个API，他会帮我们寻找超参（仅参考）
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV  # 分割训练集和测试集的，寻找最优超参的（网格搜索+交叉验证）
from sklearn.preprocessing import StandardScaler  # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier  # KNN算法 分类
from sklearn.metrics import accuracy_score  # 模型评估，计算模型预测的准确率

iris_data = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=22)

transfer = StandardScaler()

x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 创建KNN分类对象
estimator = KNeighborsClassifier()
# 定义字典，记录超参可能出现的情况（值）
param_dict = {'n_neighbors': [i for i in range(1, 11)]}
# 创建GridSearchCV对象
# 参1：要计算最优超参的模型对象
# 参2：该模型超参可能出现的值
# 参3：交叉验证的折数，4*10=40次
# 返回值estimator处理后的模型对象
estimator = GridSearchCV(estimator, param_dict, cv=4)

estimator.fit(x_train, y_train)

print(f'最优评分：{estimator.best_score_}') # 0.9666666666666668
print(f'最优超参组合：{estimator.best_params_}')
print(f'最优估计值对象：{estimator.best_estimator_}')
print(f'最优交叉验证结果：{estimator.cv_results_}')

# estimator=estimator.best_estimator_ 获取最优的模型对象
estimator = KNeighborsClassifier(n_neighbors=3)

estimator.fit(x_train,y_train)

y_pre=estimator.predict(x_test)

print(f'准确率：{accuracy_score(y_test,y_pre)}') # 0.9666666666666667