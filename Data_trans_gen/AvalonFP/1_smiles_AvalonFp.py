# -*- coding: UTF-8 -*-
'''
@Project ：MLP 
@File    ：1_smiles_AvalonFp.py
@IDE     ：PyCharm 
@Author  ：bruce
@Date    ：2023/9/5 20:12 
'''
import pandas as pd
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem
# 读取 CSV 文件
df = pd.read_csv('../../data/data.csv')

# 定义 SMILES 列名和输出指纹列名
smiles_col = 'molecule'
solvent_col = 'solvent'
smiles_output_col = 'molecule_AvalonFP'
solvent_output_col = 'solvent_AvalonFP'

# 生成分子指纹
def gen_fingerprint_to_AvalonFp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    avalon_fp = pyAvalonTools.GetAvalonFP(mol)
    return avalon_fp

df[smiles_output_col] = df[smiles_col].apply(gen_fingerprint_to_AvalonFp)
df[solvent_output_col] = df[solvent_col].apply(gen_fingerprint_to_AvalonFp)
#AvalonFp长度512比特串

# 将 BitVector 转换为字符串并输出为 CSV 文件
df[smiles_output_col] = df[smiles_output_col].apply(lambda x: x.ToBitString())
df[solvent_output_col] = df[solvent_output_col].apply(lambda x: x.ToBitString())

df.to_csv('../../data/smiles_AvalonFP.csv', index=False)

df1= df.drop(labels=['molecule', 'solvent'], axis=1) #删除含有smiles的两列
df1.to_csv('../../data/AvalonFP.csv', index=False)

# print(len(df[smiles_output_col][0]))