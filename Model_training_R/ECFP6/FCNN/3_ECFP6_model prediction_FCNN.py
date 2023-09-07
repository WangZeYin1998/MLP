import csv
from itertools import islice

import numpy as np
import torch.nn
from sklearn.preprocessing import MinMaxScaler
from time import sleep
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd

# from pylab import *

'(1) 数据预处理'

# 将位字符串转换为列表
# 如果 bitstr 是 '101010'，那么函数将返回一个包含 ['1', '0', '1', '0', '1', '0'] 的列表。
def bit2attr(bitstr) -> list:
    return list(bitstr)

# 读取位字符串数据
# 检查行的长度是否为零，如果是则跳过此行。
# 确定列数，并与预期的 NUM_ATTR 进行比较。
# 将位字符串转换为特征向量使用 bit2attr 函数，并将其存储在临时变量 temp 中。
# 将第二个位字符串的特征向量与第一个连接起来，并添加第一列的值（假设为浮点数）到 temp 列表中。
# 将 temp 添加到 data 列表中。
# 错误：num_attr == 2048，有可能是训练集或测试集中出现的列名'label','molecule_ecfp6','solvent_ecfp6'.
#NUM_ATTR = 2048
def read_bit(filepath):
    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        #num_attr = int()
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

#函数将 data 转换为NumPy数组，然后创建一个 DataFrame 对象，并将其返回
    data = np.array(data)
    data = pd.DataFrame(data)
    return data

train_filepath = "../../../data/ECFP6_train&test/train.csv"
test_filepath = "../../../data/ECFP6_train&test/test.csv"

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
#等待 1 秒
sleep(1)

'(2) MLP模型构建'

#l1 是一个包含512个神经元的全连接层，激活函数为 ReLU。
#l2 是一个 dropout 层，设置了丢弃率为0.5，用于防止过拟合。
#l3 是一个包含128个神经元的全连接层，激活函数为 ReLU。
#l4 是一个包含30个神经元的全连接层，激活函数为 ReLU。
#l5 是一个包含1个神经元的全连接层。
#使用 Adam 优化器和 logcosh 损失函数编译模型，并指定了要使用的评估指标为平均绝对误差（MAE）。

# def buildModel():
#     model = models.Sequential()
#
#     l1 = Dense(units=128, activation='relu')
#     l2 = Dropout(rate=0.3)
#     l3 = Dense(units=64, activation='relu')
#     l4 = Dense(units=12, activation='relu')
#     l5 = Dense(units=1)
#
#     layers = [l1, l2, l3, l4, l5]
#     for layer in layers:
#         model.add(layer)
#
#     adam = Adam(lr=1e-3)
#     #model.compile(optimizer=adam, loss='logcosh', metrics=['mae'])
#     model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])
#     return model





class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(4096, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 12)
        self.layer4 = nn.Linear(12, 1)

    def forward(self, x):
        x = torch.from_numpy(x).float()
        # x.reshape(-1, 4096)
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
for epoch in range(200):

    print("epoch = ", epoch)

    model.train()
    optimizer.zero_grad()
    output = model(x_trains)
    y_trains = torch.tensor(y_trains, dtype=torch.float32)
    loss = criterion(output, y_trains)
    loss.backward()
    optimizer.step()
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


#学习率调度函数scheduler(epoch, lr)。
#该函数用于自定义学习率的调整规则。在每个 epoch 的开头，调度函数将当前的 epoch 数和当前的学习率作为参数传入。
#调度函数的逻辑是，如果当前 epoch 大于0且可以被500整除，将学习率乘以0.1，否则保持不变。
#调度函数返回调整后的学习率，从而实现了学习率的动态调整策略。
# def scheduler(epoch, learning_rate):
#     if epoch > 0 and epoch % 50 == 0:
#         return learning_rate * 0.1
#     else:
#         return learning_rate

'(3)模型训练'

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
result = model(x_test)
y_test = np.reshape(y_test, (-1, 1))
y_test = min_max_scaler_y.inverse_transform(y_test)
result = result.reshape(-1, 1)
# result = min_max_scaler_y.inverse_transform(result) #此处不适用pytorch
result = min_max_scaler_y.inverse_transform(result.detach().numpy())

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

temp.to_csv("../../../data/ECFP6_FCNN_out.csv", encoding='gb18030', index=False)
