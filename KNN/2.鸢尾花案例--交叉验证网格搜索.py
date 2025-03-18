#0.导包
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#1.获取数据集
iris_data = datasets.load_iris()


#2.划分数据集
x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target,test_size=0.2,random_state=22)


#3.数据预处理 -- 数据标准化
pre = StandardScaler()
x_train = pre.fit_transform(x_train)
x_test = pre.transform(x_test)


#4.模型训练
#4.1模型实例化 + 交叉验证 + 网格搜索
model = KNeighborsClassifier()
params_grid = {'n_neighbors':[4,5,7,9,3,8]}   #定义网格搜索的超参数空间
estimator = GridSearchCV(estimator=model, param_grid=params_grid, cv=6)   #cv交叉验证（Cross-Validation）:将训练集划分为 4 个大小相似的子集（称为“折”）,每次使用 3 折作为训练数据，剩下的 1 折作为验证数据,重复 4 次（每次选择不同的验证折）

#4.2fit方法
estimator.fit(x_train, y_train)
# print(estimator.best_score_)
# print(estimator.best_params_)
# print(estimator.cv_results_)
best_model = estimator.best_estimator_
# model = KNeighborsClassifier(n_neighbors=7)
# model.fit(x_train, y_train)

#5.模型预测
x_new = [[5.1, 3.5, 1.4, 0.2]]
x_new_scaled = pre.transform(x_new)  # 标准化新数据
y_new_pred = best_model.predict(x_new_scaled)

#6.模型评估(对测试集进行模型评估)
y_test_pred = best_model.predict(x_test)
print(accuracy_score(y_test,y_test_pred))

