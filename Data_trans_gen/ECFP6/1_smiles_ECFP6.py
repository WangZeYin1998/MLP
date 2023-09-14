import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# 读取 CSV 文件
df = pd.read_csv('../../data/data.csv')

# 定义 SMILES 列名和输出指纹列名
smiles_col = 'molecule'
output_col = 'molecule_ecfp6'

# 生成分子指纹
def gen_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
    return fingerprint

df[output_col] = df[smiles_col].apply(gen_fingerprint)

# 将 BitVector 转换为字符串并输出为 CSV 文件
df[output_col] = df[output_col].apply(lambda x: x.ToBitString())

# 定义 SMILES 列名和输出指纹列名
smiles_col = 'solvent'
output_col = 'solvent_ecfp6'

# 生成分子指纹
def gen_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
    return fingerprint

df[output_col] = df[smiles_col].apply(gen_fingerprint)

# 将 BitVector 转换为字符串并输出为 CSV 文件
df[output_col] = df[output_col].apply(lambda x: x.ToBitString())
df.to_csv('../../data/smiles_ecfp6.csv', index=False)

df1= df.drop(labels=['molecule', 'solvent'],axis=1) #删除含有smiles的两列
df1.to_csv('../../data/ECFP6.csv', index=False)

