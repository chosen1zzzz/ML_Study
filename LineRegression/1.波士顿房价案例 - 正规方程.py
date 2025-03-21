#0.导包
# from sklearn.datasets import load_boston


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#1.读取数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
# x = data
# y = target



#2.数据集划分
data_train,data_test,target_train,target_test=train_test_split(data, target, test_size=0.2, random_state=22)

#3.标准化
process = StandardScaler()
data_train = process.fit_transform(data_train)
data_test = process.transform(data_test)

#4.模型训练
#4.1实例化
model = LinearRegression(fit_intercept=True)
#4.2fit训练
model.fit(data_train, target_train)
print(model.coef_)   #输出权重
print(model.intercept_)  #输出偏置
#5.预测
target_test_predict = model.predict(data_test)

print(target_test_predict)


#6.模型评估
print(mean_squared_error(target_test, target_test_predict)   )#把测试集的真实data的target，和预测的target