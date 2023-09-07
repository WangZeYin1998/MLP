import numpy as np
#利用Matplotlib绘制的带渐变颜色填充的密度图
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from KDEpy import FFTKDE
from colormaps import parula
# import matplotlib.pyplot as plt
# from matplotlib.axes import Axes
# from matplotlib.colors import LinearSegmentedColormap

'绘制误差分布密度图'
def an_error(true, pred):
    return true - pred
# 构建数据集
df = pd.read_csv('data/out.csv')
y_test = df['Real Value']
y_pred = df['Predicted Value']
AE = an_error(y_test, y_pred)
data_AE = AE.tolist()



plt.rcParams["font.family"] = "Times New Roman"  # 设置字体
plt.rcParams["axes.linewidth"] = 1  # 设置轴线宽度
plt.rcParams["axes.labelsize"] = 22  # 设置轴标签字体大小
plt.rcParams["xtick.minor.visible"] = True # 显示次要刻度线（x轴）
plt.rcParams["ytick.minor.visible"] = True # 显示次要刻度线（y轴）
plt.rcParams["xtick.direction"] = "in"  # 刻度线指向内部（x轴）
plt.rcParams["ytick.direction"] = "in"  # 刻度线指向内部（y轴）
plt.rcParams["xtick.labelsize"] = 18  # 设置刻度标签字体大小（x轴）
plt.rcParams["ytick.labelsize"] = 18  # 设置刻度标签字体大小（y轴）
plt.rcParams["xtick.top"] = False  # 不显示x轴的刻度线（顶部）
plt.rcParams["ytick.right"] = False  # 不显示y轴的刻度线（右侧）

# 使用FFTKDE对数据进行高斯核密度估计，得到x和y
x, y = FFTKDE(kernel="gaussian", bw=2).fit(data_AE).evaluate()
img_data = x.reshape(1, -1)  # 重新构造图像数据

#定义颜色列表和新的色彩映射
#clist = ['white', 'blue', 'black']
#cmap = LinearSegmentedColormap.from_list('chaos',clist)

cmap = parula  # 使用parula颜色映射
fig, ax = plt.subplots(figsize=(16, 4.5), dpi=300, facecolor="w")  # 创建画布和坐标轴
ax.plot(x, y, lw=1, color="k")  # 绘制曲线
ax.plot(data_AE, [0.005] * len(data_AE), '|', color='k', lw=1)  # 绘制数据点
ax.set_xlabel("Values")  # 设置x轴标签
ax.set_ylabel("Density")  # 设置y轴标签
ax.set_title("Error distribution density map", size=25)  # 设置图形标题

extent = [*ax.get_xlim(), *ax.get_ylim()]  # 获取坐标轴范围
im = Axes.imshow(ax, img_data, aspect='auto', cmap=cmap, extent=extent)  # 绘制带有图像数据的密度图
fill_line, = ax.fill(x, y, facecolor='none')  # 填充曲线下的区域
im.set_clip_path(fill_line) # 设置图像的剪裁路径
colorbar = fig.colorbar(im, ax=ax, aspect=12, label="Values") # 添加颜色条
fig.savefig('graph/AE.png', bbox_inches='tight', dpi=300)  # 保存图形到文件
plt.show()



'绘制误差分布直方图'
import numpy as np
import pandas as pd

# 图3-2-2 带统计信息的直方图绘制示例
from scipy.stats import norm
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"  # 设置字体
# 设置直方图的柱子数量
bins=1000
# 计算数据的中位数
Median = np.median(data_AE)
# 使用正态分布拟合数据，返回拟合的均值 mu 和标准差 std
mu, std = norm.fit(data_AE)
# 创建一个图形和坐标轴对象
fig,ax = plt.subplots(figsize=(16,4.5),dpi=100,facecolor="w")
# 绘制直方图
hist = ax.hist(x=data_AE, bins=bins,color="blue",lw=.5)
# 绘制拟合的正态分布曲线
xmin, xmax = min(data_AE),max(data_AE)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
N = len(data_AE)
bin_width = (x.max() - x.min()) / bins
ax.plot(x, p*N*bin_width,linewidth=1,color="r",label="Normal Distribution Curve")
# 添加中位数线
ax.axvline(x=Median,ls="--",lw=1,color="black",label="Median Line")
# 设置 x 轴和 y 轴的标签
ax.set_xlabel('Values')
ax.set_ylabel('Count')
ax.set_title("Error distribution histogram", size=18)  # 设置图形标题
# 在图中显示图例
ax.legend(loc='upper right', fontsize=10, frameon=False)
plt.savefig('graph\AE_1.png', bbox_inches='tight',dpi=300)
plt.show()



'绘制模型预测结果图'
# 构建数据集
df_result = pd.read_csv('data/data_analysis.csv')
MAE = df_result.iloc[0,1]
MRE = df_result.iloc[1,1]
MSE = df_result.iloc[2,1]
RMSE = df_result.iloc[3,1]
R2 = df_result.iloc[4,1]

#定义颜色列表和新的色彩映射
clist = ['white', 'blue', 'black']
cmap = LinearSegmentedColormap.from_list('chaos',clist)
#cmap = parula  # 使用parula颜色映射

#out_y_test中的最小值和最大值来确定x轴和y轴的范围
xmin = y_test.min()
xmax = y_test.max()

#创建一个14x10英寸大小的图形对象
fig = plt.figure(figsize=(16, 9))

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
#坐标轴的标签
plt.xlabel('Real values for lambda(nm)', fontsize=25)
plt.ylabel('Predicted values for lambda(nm)', fontsize=25)
# 设置图形标题
plt.title('MLP', fontsize=30)
#刻度的大小
plt.yticks(size=18)
plt.xticks(size=18)
#使用plot()函数绘制了一条红色的虚线，表示x轴和y轴相等的情况
plt.plot([xmin, xmax], [xmin, xmax], ':', linewidth=3, color='red')

#添加文本
MAE = 'MAE=%.2f' % MAE
MRE = 'MRE=%.2f%%' % (MRE * 100)
MSE = 'MSE=%.2f' % MSE
RMSE = 'RMSE=%.2f' % RMSE
R2 = 'R2=%.2f%%' % (R2 * 100)
#文本位置
plt.text(xmin + 10, xmax-30, MAE, fontsize=20, weight='bold')
plt.text(xmin + 10, xmax - 60, MRE, fontsize=20, weight='bold')
plt.text(xmin + 10, xmax - 90, MSE, fontsize=20, weight='bold')
plt.text(xmin + 10, xmax - 120, RMSE, fontsize=20, weight='bold')
plt.text(xmin + 10, xmax - 150, R2, fontsize=20, weight='bold')

#绘制散点图，camp进行着色。gridsize参数定义了网格单元格的大小，extent参数设置了x轴和y轴的范围。
hexf = plt.hexbin(y_test, y_pred, gridsize=20, extent=[xmin, xmax, xmin, xmax],
           cmap=cmap)
#图片显示横纵坐标的范围
plt.axis([xmin, xmax, xmin, xmax])
#gca()函数获取当前的坐标轴对象以进行一些自定义设置
ax = plt.gca()
#tick_params()函数设置了坐标轴刻度的位置（顶部和右侧）。
ax.tick_params(top=True, right=True)
#使用colorbar()函数添加一个颜色条，并对颜色条的标记进行一些设置。
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=18)

plt.savefig('graph/out.png', bbox_inches='tight', dpi=300)
plt.show()



'绘制平均相对误差分布密度图'
# 构建数据集
MRE = df['MRE']
data_MRE = MRE.tolist()

#利用Matplotlib绘制的带渐变颜色填充的密度图
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = "Times New Roman"  # 设置字体
plt.rcParams["axes.linewidth"] = 1  # 设置轴线宽度
plt.rcParams["axes.labelsize"] = 22  # 设置轴标签字体大小
plt.rcParams["xtick.minor.visible"] = True # 显示次要刻度线（x轴）
plt.rcParams["ytick.minor.visible"] = True # 显示次要刻度线（y轴）
plt.rcParams["xtick.direction"] = "in"  # 刻度线指向内部（x轴）
plt.rcParams["ytick.direction"] = "in"  # 刻度线指向内部（y轴）
plt.rcParams["xtick.labelsize"] = 18  # 设置刻度标签字体大小（x轴）
plt.rcParams["ytick.labelsize"] = 18  # 设置刻度标签字体大小（y轴）
plt.rcParams["xtick.top"] = False  # 不显示x轴的刻度线（顶部）
plt.rcParams["ytick.right"] = False  # 不显示y轴的刻度线（右侧）

# 使用FFTKDE对数据进行高斯核密度估计，得到x和y
x, y = FFTKDE(kernel="gaussian", bw=2).fit(data_MRE).evaluate()
img_data = x.reshape(1, -1)  # 重新构造图像数据

#定义颜色列表和新的色彩映射
#clist = ['white', 'blue', 'black']
#cmap = LinearSegmentedColormap.from_list('chaos',clist)

cmap = parula  # 使用parula颜色映射
fig, ax = plt.subplots(figsize=(16, 4.5), dpi=300, facecolor="w")  # 创建画布和坐标轴
ax.plot(x, y, lw=1, color="k")  # 绘制曲线
ax.plot(data_MRE, [0.005] * len(data_MRE), '|', color='k', lw=1)  # 绘制数据点
ax.set_xlabel("Values")  # 设置x轴标签
ax.set_ylabel("Density")  # 设置y轴标签
ax.set_title("Mean Relative Error Map", size=25)  # 设置图形标题

extent = [*ax.get_xlim(), *ax.get_ylim()]  # 获取坐标轴范围
im = Axes.imshow(ax, img_data, aspect='auto', cmap=cmap, extent=extent)  # 绘制带有图像数据的密度图
fill_line, = ax.fill(x, y, facecolor='none')  # 填充曲线下的区域
im.set_clip_path(fill_line) # 设置图像的剪裁路径
colorbar = fig.colorbar(im, ax=ax, aspect=12, label="Values") # 添加颜色条
fig.savefig('graph/MRE.png', bbox_inches='tight', dpi=300)  # 保存图形到文件
plt.show()



'绘制平均相对误差分布直方图'
import numpy as np
import pandas as pd

# 图3-2-2 带统计信息的直方图绘制示例
from scipy.stats import norm
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"  # 设置字体
# 设置直方图的柱子数量
bins=1000
# 计算数据的中位数
Median = np.median(data_MRE)
# 使用正态分布拟合数据，返回拟合的均值 mu 和标准差 std
mu, std = norm.fit(data_MRE)
# 创建一个图形和坐标轴对象
fig,ax = plt.subplots(figsize=(16,4.5),dpi=300,facecolor="w")
# 绘制直方图
hist = ax.hist(x=data_MRE, bins=bins,color="blue",lw=.5)
# 绘制拟合的正态分布曲线
xmin, xmax = min(data_MRE),max(data_MRE)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
N = len(data_MRE)
bin_width = (x.max() - x.min()) / bins
ax.plot(x, p*N*bin_width,linewidth=1,color="r",label="Normal Distribution Curve")
# 添加中位数线
ax.axvline(x=Median,ls="--",lw=1,color="black",label="Median Line")
# 设置 x 轴和 y 轴的标签
ax.set_xlabel('Values')
ax.set_ylabel('Count')
ax.set_title("Mean Relative Error distribution histogram", size=18)  # 设置图形标题
# 在图中显示图例
ax.legend(loc='upper right', fontsize=10, frameon=False)
plt.savefig('graph\MRE_1.png', bbox_inches='tight',dpi=300)
plt.show()



'绘制均方误差分布密度图'
# 构建数据集
MSE = df['MSE']
data_MSE = MSE.tolist()

#利用Matplotlib绘制的带渐变颜色填充的密度图
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = "Times New Roman"  # 设置字体
plt.rcParams["axes.linewidth"] = 1  # 设置轴线宽度
plt.rcParams["axes.labelsize"] = 22  # 设置轴标签字体大小
plt.rcParams["xtick.minor.visible"] = True # 显示次要刻度线（x轴）
plt.rcParams["ytick.minor.visible"] = True # 显示次要刻度线（y轴）
plt.rcParams["xtick.direction"] = "in"  # 刻度线指向内部（x轴）
plt.rcParams["ytick.direction"] = "in"  # 刻度线指向内部（y轴）
plt.rcParams["xtick.labelsize"] = 18  # 设置刻度标签字体大小（x轴）
plt.rcParams["ytick.labelsize"] = 18  # 设置刻度标签字体大小（y轴）
plt.rcParams["xtick.top"] = False  # 不显示x轴的刻度线（顶部）
plt.rcParams["ytick.right"] = False  # 不显示y轴的刻度线（右侧）

# 使用FFTKDE对数据进行高斯核密度估计，得到x和y
x, y = FFTKDE(kernel="gaussian", bw=2).fit(data_MSE).evaluate()
img_data = x.reshape(1, -1)  # 重新构造图像数据

#定义颜色列表和新的色彩映射
#clist = ['white', 'blue', 'black']
#cmap = LinearSegmentedColormap.from_list('chaos',clist)

cmap = parula  # 使用parula颜色映射
fig, ax = plt.subplots(figsize=(16, 4.5), dpi=300, facecolor="w")  # 创建画布和坐标轴
ax.plot(x, y, lw=1, color="k")  # 绘制曲线
ax.plot(data_MSE, [0.005] * len(data_MSE), '|', color='k', lw=1)  # 绘制数据点
ax.set_xlabel("Values")  # 设置x轴标签
ax.set_ylabel("Density")  # 设置y轴标签
ax.set_title("Mean Squared Error Map", size=25)  # 设置图形标题

extent = [*ax.get_xlim(), *ax.get_ylim()]  # 获取坐标轴范围
im = Axes.imshow(ax, img_data, aspect='auto', cmap=cmap, extent=extent)  # 绘制带有图像数据的密度图
fill_line, = ax.fill(x, y, facecolor='none')  # 填充曲线下的区域
im.set_clip_path(fill_line) # 设置图像的剪裁路径
colorbar = fig.colorbar(im, ax=ax, aspect=12, label="Values") # 添加颜色条
fig.savefig('graph/MSE.png', bbox_inches='tight', dpi=300)  # 保存图形到文件
plt.show()




'绘制均方误差分布直方图'
import numpy as np
import pandas as pd

# 图3-2-2 带统计信息的直方图绘制示例
from scipy.stats import norm
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"  # 设置字体
# 设置直方图的柱子数量
bins=1000
# 计算数据的中位数
Median = np.median(data_MSE)
# 使用正态分布拟合数据，返回拟合的均值 mu 和标准差 std
mu, std = norm.fit(data_MSE)
# 创建一个图形和坐标轴对象
fig,ax = plt.subplots(figsize=(16,4.5),dpi=300,facecolor="w")
# 绘制直方图
hist = ax.hist(x=data_MSE, bins=bins,color="blue",lw=.5)
# 绘制拟合的正态分布曲线
xmin, xmax = min(data_MSE),max(data_MSE)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
N = len(data_MSE)
bin_width = (x.max() - x.min()) / bins
ax.plot(x, p*N*bin_width,linewidth=1,color="r",label="Normal Distribution Curve")
# 添加中位数线
ax.axvline(x=Median,ls="--",lw=1,color="black",label="Median Line")
# 设置 x 轴和 y 轴的标签
ax.set_xlabel('Values')
ax.set_ylabel('Count')
ax.set_title("Mean Squared Error distribution histogram", size=18)  # 设置图形标题
# 在图中显示图例
ax.legend(loc='upper right', fontsize=10, frameon=False)
plt.savefig('graph\MSE_1.png', bbox_inches='tight',dpi=300)
plt.show()




'绘制均方根误差分布密度图'
# 构建数据集
RMSE = df['RMSE']
data_RMSE = RMSE.tolist()

#利用Matplotlib绘制的带渐变颜色填充的密度图
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = "Times New Roman"  # 设置字体
plt.rcParams["axes.linewidth"] = 1  # 设置轴线宽度
plt.rcParams["axes.labelsize"] = 22  # 设置轴标签字体大小
plt.rcParams["xtick.minor.visible"] = True # 显示次要刻度线（x轴）
plt.rcParams["ytick.minor.visible"] = True # 显示次要刻度线（y轴）
plt.rcParams["xtick.direction"] = "in"  # 刻度线指向内部（x轴）
plt.rcParams["ytick.direction"] = "in"  # 刻度线指向内部（y轴）
plt.rcParams["xtick.labelsize"] = 18  # 设置刻度标签字体大小（x轴）
plt.rcParams["ytick.labelsize"] = 18  # 设置刻度标签字体大小（y轴）
plt.rcParams["xtick.top"] = False  # 不显示x轴的刻度线（顶部）
plt.rcParams["ytick.right"] = False  # 不显示y轴的刻度线（右侧）

# 使用FFTKDE对数据进行高斯核密度估计，得到x和y
x, y = FFTKDE(kernel="gaussian", bw=2).fit(data_RMSE).evaluate()
img_data = x.reshape(1, -1)  # 重新构造图像数据

#定义颜色列表和新的色彩映射
#clist = ['white', 'blue', 'black']
#cmap = LinearSegmentedColormap.from_list('chaos',clist)

cmap = parula  # 使用parula颜色映射
fig, ax = plt.subplots(figsize=(16, 4.5), dpi=300, facecolor="w")  # 创建画布和坐标轴
ax.plot(x, y, lw=1, color="k")  # 绘制曲线
ax.plot(data_RMSE, [0.005] * len(data_RMSE), '|', color='k', lw=1)  # 绘制数据点
ax.set_xlabel("Values")  # 设置x轴标签
ax.set_ylabel("Density")  # 设置y轴标签
ax.set_title("Root Mean Squared Error Map", size=25)  # 设置图形标题

extent = [*ax.get_xlim(), *ax.get_ylim()]  # 获取坐标轴范围
im = Axes.imshow(ax, img_data, aspect='auto', cmap=cmap, extent=extent)  # 绘制带有图像数据的密度图
fill_line, = ax.fill(x, y, facecolor='none')  # 填充曲线下的区域
im.set_clip_path(fill_line) # 设置图像的剪裁路径
colorbar = fig.colorbar(im, ax=ax, aspect=12, label="Values") # 添加颜色条
fig.savefig('graph/RMSE.png', bbox_inches='tight', dpi=300)  # 保存图形到文件
plt.show()



'绘制均方根误差分布直方图'
import numpy as np
import pandas as pd

# 图3-2-2 带统计信息的直方图绘制示例
from scipy.stats import norm
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"  # 设置字体
# 设置直方图的柱子数量
bins=1000
# 计算数据的中位数
Median = np.median(data_RMSE)
# 使用正态分布拟合数据，返回拟合的均值 mu 和标准差 std
mu, std = norm.fit(data_RMSE)
# 创建一个图形和坐标轴对象
fig,ax = plt.subplots(figsize=(16,4.5),dpi=300,facecolor="w")
# 绘制直方图
hist = ax.hist(x=data_RMSE, bins=bins,color="blue",lw=.5)
# 绘制拟合的正态分布曲线
xmin, xmax = min(data_RMSE),max(data_RMSE)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
N = len(data_RMSE)
bin_width = (x.max() - x.min()) / bins
ax.plot(x, p*N*bin_width,linewidth=1,color="r",label="Normal Distribution Curve")
# 添加中位数线
ax.axvline(x=Median,ls="--",lw=1,color="black",label="Median Line")
# 设置 x 轴和 y 轴的标签
ax.set_xlabel('Values')
ax.set_ylabel('Count')
ax.set_title("Root Mean Squared Error distribution histogram", size=18)  # 设置图形标题
# 在图中显示图例
ax.legend(loc='upper right', fontsize=10, frameon=False)
plt.savefig('graph\RMSE_1.png', bbox_inches='tight',dpi=300)
plt.show()



'绘制R方分布密度图'
# 构建数据集
R2 = df['R2']
data_R2 = R2.tolist()

#利用Matplotlib绘制的带渐变颜色填充的密度图
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = "Times New Roman"  # 设置字体
plt.rcParams["axes.linewidth"] = 1  # 设置轴线宽度
plt.rcParams["axes.labelsize"] = 22  # 设置轴标签字体大小
plt.rcParams["xtick.minor.visible"] = True # 显示次要刻度线（x轴）
plt.rcParams["ytick.minor.visible"] = True # 显示次要刻度线（y轴）
plt.rcParams["xtick.direction"] = "in"  # 刻度线指向内部（x轴）
plt.rcParams["ytick.direction"] = "in"  # 刻度线指向内部（y轴）
plt.rcParams["xtick.labelsize"] = 18  # 设置刻度标签字体大小（x轴）
plt.rcParams["ytick.labelsize"] = 18  # 设置刻度标签字体大小（y轴）
plt.rcParams["xtick.top"] = False  # 不显示x轴的刻度线（顶部）
plt.rcParams["ytick.right"] = False  # 不显示y轴的刻度线（右侧）

# 使用FFTKDE对数据进行高斯核密度估计，得到x和y
x, y = FFTKDE(kernel="gaussian", bw=2).fit(data_R2).evaluate()
img_data = x.reshape(1, -1)  # 重新构造图像数据

#定义颜色列表和新的色彩映射
#clist = ['white', 'blue', 'black']
#cmap = LinearSegmentedColormap.from_list('chaos',clist)

cmap = parula  # 使用parula颜色映射
fig, ax = plt.subplots(figsize=(16, 4.5), dpi=300, facecolor="w")  # 创建画布和坐标轴
ax.plot(x, y, lw=1, color="k")  # 绘制曲线
ax.plot(data_R2, [0.005] * len(data_R2), '|', color='k', lw=1)  # 绘制数据点
ax.set_xlabel("Values")  # 设置x轴标签
ax.set_ylabel("Density")  # 设置y轴标签
ax.set_title("R-Square Map", size=25)  # 设置图形标题

extent = [*ax.get_xlim(), *ax.get_ylim()]  # 获取坐标轴范围
im = Axes.imshow(ax, img_data, aspect='auto', cmap=cmap, extent=extent)  # 绘制带有图像数据的密度图
fill_line, = ax.fill(x, y, facecolor='none')  # 填充曲线下的区域
im.set_clip_path(fill_line) # 设置图像的剪裁路径
colorbar = fig.colorbar(im, ax=ax, aspect=12, label="Values") # 添加颜色条
fig.savefig('graph/R2.png', bbox_inches='tight', dpi=300)  # 保存图形到文件
plt.show()



'绘制R方分布直方图'
import numpy as np
import pandas as pd

# 图3-2-2 带统计信息的直方图绘制示例
from scipy.stats import norm
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"  # 设置字体
# 设置直方图的柱子数量
bins=1000
# 计算数据的中位数
Median = np.median(data_R2)
# 使用正态分布拟合数据，返回拟合的均值 mu 和标准差 std
mu, std = norm.fit(data_R2)
# 创建一个图形和坐标轴对象
fig,ax = plt.subplots(figsize=(16,4.5),dpi=300,facecolor="w")
# 绘制直方图
hist = ax.hist(x=data_R2, bins=bins,color="blue",lw=.5)
# 绘制拟合的正态分布曲线
xmin, xmax = min(data_R2),max(data_R2)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
N = len(data_R2)
bin_width = (x.max() - x.min()) / bins
ax.plot(x, p*N*bin_width,linewidth=1,color="r",label="Normal Distribution Curve")
# 添加中位数线
ax.axvline(x=Median,ls="--",lw=1,color="black",label="Median Line")
# 设置 x 轴和 y 轴的标签
ax.set_xlabel('Values')
ax.set_ylabel('Count')
ax.set_title("R-Square distribution histogram", size=18)  # 设置图形标题
# 在图中显示图例
ax.legend(loc='upper right', fontsize=10, frameon=False)
plt.savefig('graph\R2_1.png', bbox_inches='tight',dpi=300)
plt.show()

