# -*- coding: UTF-8 -*-
'''
@Project ：MLP 
@File    ：3_RDKitFP_model prediction_FCNN.py
@IDE     ：PyCharm 
@Author  ：bruce
@Date    ：2023/9/6 10:18 
'''
import csv
import csv
from itertools import islice

import pandas as pd
import numpy as np
import torch.nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torch.optim.lr_scheduler import StepLR


# from pylab import *

def bit2attr(bitstr) -> list:
    return list(bitstr)

# 读取位字符串数据
# 检查行的长度是否为零，如果是则跳过此行。
# 确定列数，并与预期的 NUM_ATTR 进行比较。
# 将位字符串转换为特征向量使用 bit2attr 函数，并将其存储在临时变量 temp 中。
# 将第二个位字符串的特征向量与第一个连接起来，并添加第一列的值（假设为浮点数）到 temp 列表中。
# 将 temp 添加到 data 列表中。
# 错误：num_attr == 512，有可能是训练集或测试集中出现的列名'label','molecule_avalonFP','solvent_avalonFP'.
#NUM_ATTR = 512
def read_bit(filepath):
    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        #num_attr = int()
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

train_filepath = "../../../data/AvalonFP_train&test/train.csv"
test_filepath = "../../../data/AvalonFP_train&test/test.csv"

'训练集数据处理'
train_data = read_bit(train_filepath)
#train_data = shuffle(train_data) #随机打乱数据
#创建 data_x_df 和 data_y_df 两个变量，分别存储 train_data 中的特征数据（除最后一列）和标签数据（最后一列）。
train_data_x_df = pd.DataFrame(train_data.iloc[:, :-1])#取分子fp和溶剂fp(共4096bit)
train_data_y_df = pd.DataFrame(train_data.iloc[:, -1])#取label

# 使用 MinMaxScaler 进行归一化
min_max_scaler_X = MinMaxScaler()
min_max_scaler_X.fit(train_data_x_df)
x_trains = min_max_scaler_X.transform(train_data_x_df)

min_max_scaler_y = MinMaxScaler()
min_max_scaler_y.fit(train_data_y_df)
y_trains = min_max_scaler_y.transform(train_data_y_df)
'测试集数据处理'
test_data = read_bit(test_filepath)

test_data_x_df = pd.DataFrame(test_data.iloc[:, :-1])
test_data_y_df = pd.DataFrame(test_data.iloc[:, -1])

x_test = min_max_scaler_X.transform(test_data_x_df)
y_test = min_max_scaler_y.transform(test_data_y_df)

# train_data_input = pd.concat([pd.DataFrame(x_trains), pd.DataFrame(y_trains)])

batch_size = 64
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(x_trains, batch_size=batch_size, shuffle=True)

class Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.data)

torch.set_default_dtype(torch.float32)
train_dataset = Dataset(x_trains.astype(np.float32), y_trains.astype(np.float32))
test_dataset = Dataset(x_test.astype(np.float32), y_test.astype(np.float32))
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(1024, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 12)
        self.layer4 = nn.Linear(12, 1)

    def forward(self, x):
        # x = torch.from_numpy(x).float()
        # x = x.float()
        # x.reshape(-1, 4096)
        # x = torch.from_numpy(x)
        x = torch.relu(self.layer1(x))
        x = nn.Dropout(0.3)(x)
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        return x

model = Model()


optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
criterion = nn.MSELoss()

# 训练模型
def train():
    for epoch in range(2):

        print("epoch = ", epoch)
        temp_loss = 0.0
        model.train()

        for i, data in enumerate(trainloader, 0):
            inputs = data[0]
            labels = data[1]
            optimizer.zero_grad()
            output = model(inputs)
            # y_trains = torch.tensor(y_trains, dtype=torch.float32)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # 计算平均损失
            temp_loss += loss.item()
            if i % 100 == 99:  # 每100个小批次打印一次损失值
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, temp_loss / 100))
                running_loss = 0.0
        scheduler.step()  # 在每次迭代后更新学习率



    # if (epoch + 1) % 100 == 0:
    #     model.eval()
    #     with torch.no_grad():
    #         test_output = model(x_test)
    #         test_loss = criterion(test_output, y_test)
    #         test_acc = optimizer.state_dict()['param_groups'][0]['lr']
    #
    #     print('Epoch: {}, Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(epoch + 1, test_loss, test_acc))

print("Model_train finished!")
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
#  # 外部验证
# result = model_mlp.predict(x_test)
temp = []
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        output = model(inputs)
        temp.append(output.cpu().numpy())

result = np.concatenate(temp)
# result = model(x_test)
y_test = np.reshape(y_test, (-1, 1))
y_test = min_max_scaler_y.inverse_transform(y_test)
result = result.reshape(-1, 1)
result = min_max_scaler_y.inverse_transform(result)
# result = min_max_scaler_y.inverse_transform(result.detach().numpy())

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

temp.to_csv("../../../data/AvalonFP_FCNN_out.csv", encoding='gb18030', index=False)

