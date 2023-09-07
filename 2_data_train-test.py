from tkinter import filedialog

import pandas as pd
import tkinter as tk


window = tk.Tk()

window.withdraw()
data = filedialog.askopenfilename(
    title="Select csv data file (CSV format)", filetypes=[("CSV data files", "*.csv")]
)
output_folder = filedialog.askdirectory(
    title="Select output folder"
)


# 随机打乱数据
data = pd.read_csv(data)
data = data.sample(frac=1, random_state=42)

# 划分数据集
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# 保存为不同的 CSV 文件
train_data.to_csv(f'{output_folder}/train.csv', index=False)
test_data.to_csv(f'{output_folder}/test.csv', index=False)