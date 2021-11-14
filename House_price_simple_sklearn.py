# %%
!pip install kaggle

# %% [markdown]
# 引入kaggle資料(ex: !kaggle competitions download -c machine-learningntut-2021-autumn-regression)

# %%
api_token = {"username":"kaggle_username","key":"kaggle_key"}
import json
import zipfile
import os
 
if not os.path.exists("/root/.kaggle"):
    os.makedirs("/root/.kaggle")
 
with open('/root/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)
!chmod 600 /root/.kaggle/kaggle.json
 
if not os.path.exists("/kaggle"):
    os.makedirs("/kaggle")
os.chdir('/kaggle')
!kaggle competitions download -c machine-learningntut-2021-autumn-regression
 
!ls /kaggle

# %% [markdown]
# 解壓縮train-v3.csv.zip

# %%
!unzip "/kaggle/train-v3.csv.zip" -d "/kaggle/"

# %%
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

# %%
#輸入資料並查看
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt

basepath = "/kaggle/"

train = pd.read_csv(basepath + "train-v3.csv")
valid = pd.read_csv(basepath + "valid-v3.csv")

# %%
# 分類
train_xx = train.drop(['sale_month', 'sale_day', 'sale_yr','id','price'], axis = 1)
valid_xx = valid.drop(['sale_month', 'sale_day', 'sale_yr','id','price'], axis = 1)
train_yy = train[['price']].values
valid_yy = valid[['price']].values

# %%
#表準化數據
from sklearn import preprocessing
set_x = preprocessing.StandardScaler()
train_x = set_x.fit_transform(train_xx)
valid_x = set_x.fit_transform(valid_xx)

set_y = preprocessing.StandardScaler()
train_y = set_y.fit_transform(train_yy.reshape(-1, 1))
valid_y = set_y.fit_transform(valid_yy.reshape(-1, 1))

# %%
# sklearn 多層回歸模型
model1 = MLPRegressor(solver='lbfgs',hidden_layer_sizes=(16,8,8),random_state=42,max_iter=5000)
model1.fit(train_x,train_y.ravel())
score = model1.score(valid_x,valid_y.ravel())
print('sklearn 多層回歸模型得分',score)

# %%
# sklearn 集成回歸模型
model2 = GradientBoostingRegressor()
model2.fit(train_x,train_y.ravel())
score = model2.score(valid_x,valid_y.ravel())
print('sklearn 集成回歸模型得分',score)

# %%
# # sklearn 集成回歸模型 參數調整
# model = GradientBoostingRegressor()
# # 設置參數
# # param = {'n_estimators'     : range(20,201,10), 200 
# #          'learning_rate'    : [0.2,0.1,0.05,0.02,0.01], 0.2 
# #          'max_depth'        : range(2,5,1), 4 
# #          'min_samples_leaf' : [3, 5, 8, 13], 3 
# #          'max_features'     : [0.8,0.5,0.3,0.1]} 0.3
# param = {'n_estimators'     : range(20,201,10)}
# # 跑參數
# from sklearn.model_selection import GridSearchCV
# est = GridSearchCV(model, param)
# est.fit(train_x,train_y.ravel())

# print("最佳參數: ",est.best_params_)

# %%
model3 = GradientBoostingRegressor(n_estimators = 200, learning_rate = 0.2, 
                                  max_depth = 4, min_samples_leaf = 3,  
                                  max_features = 0.3)
model3.fit(train_x,train_y.ravel())
score = model3.score(valid_x,valid_y.ravel())
print('sklearn 調參後 集成回歸模型得分',score)

# %%
# sklearn 多層回歸模型
model4 = MLPRegressor(solver='adam',hidden_layer_sizes=(16,8,8),random_state=42,max_iter=5000)
model4.fit(train_x,train_y.ravel())
score = model4.score(valid_x,valid_y.ravel())
print('sklearn 多層回歸模型得分',score)

# %%
model5 = MLPRegressor(solver='sgd',hidden_layer_sizes=(16,8,8),random_state=42,max_iter=5000)
model5.fit(train_x,train_y.ravel())
score = model5.score(valid_x,valid_y.ravel())
print('sklearn 多層回歸模型得分',score)

# %%
test = pd.read_csv(basepath + "test-v3.csv")
# 存下id
id = test['id'].values
# 取出垃圾
test = test.drop(['sale_month', 'sale_day', 'sale_yr','id'], axis = 1)
#表準化數據
test = preprocessing.StandardScaler().fit_transform(test)

# %%
pred1 = model1.predict(test)
# pred2 = model2.predict(test)
pred3 = model3.predict(test)
pred4 = model4.predict(test)
pred5 = model5.predict(test)

# %%
scaler = preprocessing.StandardScaler().fit(train_yy)
pred1 = scaler.inverse_transform(pred1)
# pred2 = scaler.inverse_transform(pred2)
pred3 = scaler.inverse_transform(pred3)
pred4 = scaler.inverse_transform(pred4)
pred5 = scaler.inverse_transform(pred5)
pred_avg = (pred1 + pred3 + pred4 + pred5) / 4

# %%
id = list(id)

y_predict = pred_avg

mem = {"id":  id, "price" : y_predict}
mem_df = pd.DataFrame(mem)
mem_df.head()
mem_df.to_csv("submission_sklearn.csv", index=False)

# %%
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(20,4))
# axes = fig.add_subplot(1,1,1)
# line1,=axes.plot(range(len(predv)), predv, 'y', label = 'real')
# line2,=axes.plot(range(len(pred_avg)), pred_avg, 'b', label = 'model_mix')
# axes.grid()
# fig.tight_layout()
# # plt.legend(handles = [line1, line2, line3, line4])
# plt.legend(handles = [line1, line2])
# plt.title('sklearn model')
# plt.show()


