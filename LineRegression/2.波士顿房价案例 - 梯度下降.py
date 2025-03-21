#0.导包
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#1.数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

#2.分割数据
data_train,data_test,target_tarin,target_test = train_test_split(data,target,test_size=0.2,random_state=22)

#3.数据预处理
process = StandardScaler()
data_train = process.fit_transform(data_train)
data_test = process.transform(data_test)


#4.模型训练
model = SGDRegressor()
model.fit(data_train,target_tarin)

#5.模型预测
target_test_predict = model.predict(data_test)
# print(target_test_predict)

#6.模型评估
print(mean_squared_error(target_test,target_test_predict))