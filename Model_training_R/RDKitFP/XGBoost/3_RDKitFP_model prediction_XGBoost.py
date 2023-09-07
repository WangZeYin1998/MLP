# -*- coding: UTF-8 -*-
'''
@Project ：MLP 
@File    ：3_RDKitFP_model prediction_XGBoost.py
@IDE     ：PyCharm 
@Author  ：bruce
@Date    ：2023/9/6 17:11 
'''
import csv
from itertools import islice

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor as XGBR
from xgboost import plot_importance, plot_tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def bit2attr(bitstr) -> list:
    return list(bitstr)


def read_bit(filepath):
    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        # num_attr = int()
        for row in islice(reader, 1, None):  # 不跳过第一行
            if len(row) == 0:
                continue
            num_attr = len(row[1])
            assert num_attr == 2048
            num_attr = len(row[2])
            assert num_attr == 2048
            temp = bit2attr(row[1])
            temp = temp + bit2attr(row[2])
            temp.append(float(row[0]))
            data.append(temp)

    data = np.array(data)
    data = pd.DataFrame(data)
    return data


train_filepath = "../../../data/RDKitFP_train&test/train.csv"
test_filepath = "../../../data/RDKitFP_train&test/test.csv"

'Train_dataset preprogress'
train_data = read_bit(train_filepath)

train_data_x_df = pd.DataFrame(train_data.iloc[:, :-1])
train_data_y_df = pd.DataFrame(train_data.iloc[:, -1])

# 使用 MinMaxScaler 进行归一化
min_max_scaler_X = MinMaxScaler()
min_max_scaler_X.fit(train_data_x_df)
x_trains = min_max_scaler_X.transform(train_data_x_df)

min_max_scaler_y = MinMaxScaler()
min_max_scaler_y.fit(train_data_y_df)
y_trains = min_max_scaler_y.transform(train_data_y_df)
'Test_dataset preprogress'
test_data = read_bit(test_filepath)

test_data_x_df = pd.DataFrame(test_data.iloc[:, :-1])
test_data_y_df = pd.DataFrame(test_data.iloc[:, -1])

x_test = min_max_scaler_X.transform(test_data_x_df)
y_test = min_max_scaler_y.transform(test_data_y_df)

train_dataset = xgb.DMatrix(x_trains, label=y_trains)
test_dataset = xgb.DMatrix(x_test)

# Set custom hyperparameters
params = {'booster': 'gbtree',
          'nthread': 12,
          'objective': 'rank:pairwise',
          'eval_metric': 'auc',  # optional:rmse、mae、logloss、error、merror、mlogloss、auc
          'seed': 0,
          'eta': 0.01,  # learning rate
          'gamma': 0.1,
          'min_child_weight': 1.1,  # Sum of minimum sample weights
          'max_depth': 5,
          'lambda': 10,  # Control the regularization part of XGBoost to reduce overfitting
          'subsample': 0.7,
          'colsample_bytree': 0.7,  # Control the proportion of randomly sampled columns per tree
          'colsample_bylevel': 0.7,
          # Control the proportion of sampling the number of columns for each level of splitting in the tree
          'tree_method': 'exact'
          }
watchlist = [(train_dataset, 'train')]
model = xgb.train(params, train_dataset, num_boost_round=1, evals=watchlist)

print("train finished!")
model.save_model('xgb_model')


result = model.predict(test_dataset)
result = result.reshape(result.size,1)
print("test finished!")

'Training'

# 计算平均绝对误差
def mean_relative_error(true, pred):
    return np.sum(np.abs((true - pred) / true))/len(true)
# 计算绝对误差
def mean_absolute_error(true, pred):
    return np.sum(np.abs(true - pred))/len(true)
# 计算R方
def R_Square(true, pred):
    residuals = true - pred
    TSS = np.sum((true - ture_av)**2)
    if TSS == 0:
        # 处理TSS为零的情况
        return np.nan  # 返回一个特定的值，如NaN
    else:
        RSS = np.sum(residuals**2)
        R_squared = 1 - (RSS / TSS)
        return R_squared

# callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
# model_mlp = buildModel()
# model_mlp.fit(x_trains, y_trains, epochs=200, verbose=1, callbacks=[callback])
#
#  # external validation
# result = model_mlp.predict(x_test)

y_test = np.reshape(y_test, (-1, 1))
y_test = min_max_scaler_y.inverse_transform(y_test)
result = result.reshape(-1, 1)

result = min_max_scaler_y.inverse_transform(result)

ture_av = np.mean(y_test)
mae = mean_absolute_error(y_test, result)
mre = mean_relative_error(y_test, result)
mse = mean_squared_error(y_test, result)
rmse = np.sqrt(mean_squared_error(y_test, result))
R2 = R_Square(y_test, result)

MAE = []
MRE = []
MSE = []
RMSE = []
R2 = []

X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
X_test = min_max_scaler_X.inverse_transform(X_test)

for idx in range(len(y_test)):
    MAE.append(mean_absolute_error(y_test[idx], result[idx]))
    MRE.append(mean_relative_error(y_test[idx], result[idx]))
    MSE.append(mean_squared_error(y_test[idx], result[idx]))
    RMSE.append(np.sqrt(mean_squared_error(y_test[idx], result[idx])))
    R2.append(R_Square(y_test[idx], result[idx]))

y_test = list(np.reshape(y_test, (-1,)))
y_pred = list(np.reshape(result, (-1,)))

temp = pd.DataFrame(X_test)
temp = pd.concat([temp, pd.DataFrame({'Real Value': y_test}),
                  pd.DataFrame({'Predicted Value': y_pred}),
                  pd.DataFrame({'MAE': MAE}),
                  pd.DataFrame({'MRE': MRE}),
                 pd.DataFrame({'MSE': MSE}),
                pd.DataFrame({'RMSE': RMSE}),
                pd.DataFrame({'R2': R2})], axis=1)

temp.to_csv("../../../data/RDKitFP_XGBoost_out.csv", encoding='gb18030', index=False)
