# -*- coding: UTF-8 -*-
'''
@Project ：MLP 
@File    ：3_AvalonFP_model prediction_RF.py
@IDE     ：PyCharm 
@Author  ：bruce
@Date    ：2023/9/12 8:54 
'''
import os
import pickle

import joblib
import shap as shap
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump,load
import pandas as pd
import numpy as np
import csv
from itertools import islice
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tqdm import tqdm

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
            assert num_attr == 512
            num_attr = len(row[2])
            assert num_attr == 512
            temp = bit2attr(row[1])
            temp = temp + bit2attr(row[2])
            temp.append(float(row[0]))
            data.append(temp)

    data = np.array(data)
    data = pd.DataFrame(data)
    return data
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

train_filepath = "../../../data/AvalonFP_train&test/train.csv"
test_filepath = "../../../data/AvalonFP_train&test/test.csv"

if os.path.isfile("../../../data/AvalonFP_train&test/x_trains_MinMaxScaler.pkl"):
    with open('../../../data/AvalonFP_train&test/x_trains_MinMaxScaler.pkl', 'rb') as f:
        x_trains = pickle.load(f)
    with open('../../../data/AvalonFP_train&test/y_trains_MinMaxScaler.pkl', 'rb') as f:
        y_trains = pickle.load(f)
    with open('../../../data/AvalonFP_train&test/x_test_trans.pkl', 'rb') as f:
        x_test = pickle.load(f)
    with open('../../../data/AvalonFP_train&test/y_test_trans.pkl', 'rb') as f:
        y_test = pickle.load(f)


else:
    print("数据未经处理！")
    train_data = read_bit(train_filepath)
    # train_data = shuffle(train_data) #随机打乱数据
    # 创建 data_x_df 和 data_y_df 两个变量，分别存储 train_data 中的特征数据（除最后一列）和标签数据（最后一列）。
    train_data_x_df = pd.DataFrame(train_data.iloc[:, :-1])  # 取分子fp和溶剂fp(共4096bit)
    train_data_y_df = pd.DataFrame(train_data.iloc[:, -1])  # 取label

    # 使用 MinMaxScaler 进行归一化
    min_max_scaler_X = MinMaxScaler()
    min_max_scaler_X.fit(train_data_x_df)
    x_trains = min_max_scaler_X.transform(train_data_x_df)
    # x_trains = pd.get_dummies(x_trains)

    min_max_scaler_y = MinMaxScaler()
    min_max_scaler_y.fit(train_data_y_df)
    y_trains = min_max_scaler_y.transform(train_data_y_df)
    y_trains = y_trains.reshape(-1)

    'test_dataset progress'
    test_data = read_bit(test_filepath)

    test_data_x_df = pd.DataFrame(test_data.iloc[:, :-1])
    test_data_y_df = pd.DataFrame(test_data.iloc[:, -1])

    x_test = min_max_scaler_X.transform(test_data_x_df)
    y_test = min_max_scaler_y.transform(test_data_y_df)
    y_test = y_test.reshape(-1)

    with open('../../../data/AvalonFP_train&test/x_trains_MinMaxScaler.pkl', 'wb') as f:
        pickle.dump(x_trains, f)
    with open('../../../data/AvalonFP_train&test/y_trains_MinMaxScaler.pkl', 'wb') as f:
        pickle.dump(y_trains, f)
    with open('../../../data/AvalonFP_train&test/x_test_trans.pkl', 'wb') as f:
        pickle.dump(x_test, f)
    with open('../../../data/AvalonFP_train&test/y_test_trans.pkl', 'wb') as f:
        pickle.dump(y_test, f)


x_trains = x_trains[0:501]
y_trains = y_trains[0:501]



'model training'
if os.path.isfile("train_model.m"):
    print("model exist！")
    rf = joblib.load("train_model.m")
    result = rf.predict(x_test)
    result = result.reshape(-1, 1)
else:
    print("model not exist!")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, verbose=1)

    rf.fit(x_trains, y_trains)

    print("train finished!")
    joblib.dump(rf, "train_model.m")

    result = rf.predict(x_test)
    result = result.reshape(-1, 1)

explainer = shap.Explainer(rf.fit(x_trains, y_trains))
shap_values = explainer(x_trains)
shap.summary_plot(shap_values, x_trains)

# 打印预测结果
# print(result)
# train_data = read_bit(train_filepath)
#
# train_data_x_df = pd.DataFrame(train_data.iloc[:, :-1])
# train_data_y_df = pd.DataFrame(train_data.iloc[:, -1])

# 使用 MinMaxScaler 进行归一化

test_data = read_bit(test_filepath)

test_data_x_df = pd.DataFrame(test_data.iloc[:, :-1])
test_data_y_df = pd.DataFrame(test_data.iloc[:, -1])

min_max_scaler_X = MinMaxScaler()
min_max_scaler_X.fit(test_data_x_df)

min_max_scaler_y = MinMaxScaler()
min_max_scaler_y.fit(test_data_y_df)

x_test = min_max_scaler_X.transform(test_data_x_df)
y_test = min_max_scaler_y.transform(test_data_y_df)

# result = min_max_scaler_y.inverse_transform(result) #此处不适用pytorch
#result = min_max_scaler_y.inverse_transform(result.detach().numpy())

y_test = min_max_scaler_y.inverse_transform(y_test)

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

temp.to_csv("../../../data/AvalonFP_RF_out.csv", encoding='gb18030', index=False)
