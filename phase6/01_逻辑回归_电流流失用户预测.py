"""
案例：
    通过逻辑回归算法，针对电信用户数据建模。进行流失观测分析
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    classification_report  # 准确率 精确率 召回率 F1值 分类评估报告


# 数据预处理
def data_preprocess():
    churn_df = pd.read_csv('./data/churn.csv')

    # churn_df.info()
    # print(churn_df.head(5))

    # Churn gender列是字符串，所以需要进行one-hot编码（热编码处理）
    churn_df = pd.get_dummies(churn_df, columns=['Churn', 'gender'])

    # churn_df.info()
    # print(churn_df.head(5))

    # 删除冗余的列 axis=1
    churn_df.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)

    # churn_df.info()
    # print(churn_df.head(5))

    # 修改列名，Churn_Yes -> flag,充当标签列
    churn_df.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    churn_df.info()
    print(churn_df.head(5))

    print(churn_df.flag.value_counts())  # False 5174   True 1869


def data_visualization():
    churn_df = pd.read_csv('./data/churn.csv')
    churn_df = pd.get_dummies(churn_df, columns=['Churn', 'gender'])
    churn_df.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)
    churn_df.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    """
    列名：
        ['Partner_att', 'Dependents_att', 'landline', 'internet_att',
       'internet_other', 'StreamingTV', 'StreamingMovies', 'Contract_Month',
       'Contract_1YR', 'PaymentBank', 'PaymentCreditcard', 'PaymentElectronic',
       'MonthlyCharges', 'TotalCharges', 'flag', 'gender_Female'] 
    """
    print(churn_df.columns)

    sns.countplot(data=churn_df, x='Contract_Month', hue='flag')
    plt.show()


def logistic_regression():
    churn_df = pd.read_csv('./data/churn.csv')
    churn_df = pd.get_dummies(churn_df, columns=['Churn', 'gender'])
    churn_df.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)
    churn_df.rename(columns={'Churn_Yes': 'flag'}, inplace=True)

    x = churn_df[['Contract_Month', 'internet_other', 'PaymentElectronic']]
    y = churn_df['flag']
    x_trian, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

    estimator = LogisticRegression()
    estimator.fit(x_trian, y_train)

    y_pred = estimator.predict(x_test)
    print(f'predict:{y_pred}')

    print(f'预测前准确率:{estimator.score(x_test, y_test)}')
    print(f'预测后准确率:{accuracy_score(y_test, y_pred)}')

    print(f'精确率:{precision_score(y_test, y_pred)}')
    print(f'召回率:{recall_score(y_test, y_pred)}')
    print(f'F1率:{f1_score(y_test, y_pred)}')

    # macro avg:宏平均，即不考虑样本权重，直接求平均，适用于数据均衡的情况
    # weighted avg:样本权重平均，即考虑样本权重，求平均，适用于数据不均衡的情况
    print(f'分类评估报告:\n{classification_report(y_test, y_pred)}')


if __name__ == '__main__':
    # data_preprocess()
    # data_visualization()
    logistic_regression()
