from sklearn.metrics import mean_squared_error
import pandas as pd
from pylab import *


#计算平均绝对误差
def mean_relative_error(true, pred):
    return np.sum(np.abs((true - pred) / true))/len(true)
#计算绝对误差
def mean_absolute_error(true, pred):
    return np.sum(np.abs(true - pred))/len(true)
#计算R方
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


df = pd.read_csv('data/out.csv')
y_test = df['Real Value']
result = df['Predicted Value']

ture_av = np.mean(y_test)
mae = mean_absolute_error(y_test, result)
mre = mean_relative_error(y_test, result)
mse = mean_squared_error(y_test, result)
rmse = np.sqrt(mean_squared_error(y_test, result))
R2 = R_Square(y_test, result)
# 创建数据帧
data = pd.DataFrame({'Metrics': ['Mean Absolute Error', 'Mean Relative Error', 'Mean Squared Error', 'Root Mean Squared Error', 'R-Squared'],
                     'Value': [mae, mre, mse, rmse, R2]})
# 将数据帧写入CSV文件
data.to_csv('data/data_analysis.csv', encoding='gb18030', index=False)