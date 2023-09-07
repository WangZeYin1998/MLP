import pandas as pd
from rdkit import Chem
from rdkit.Chem import RDKFingerprint
# 读取 CSV 文件


df = pd.read_csv('../../data/data.csv')

# 定义 SMILES 列名和输出指纹列名
smiles_col = 'molecule'
solvent_col = 'solvent'
smiles_output_col = 'molecule_RDKitFP'
solvent_output_col = 'solvent_RDKitFP'

# 生成分子指纹
def gen_fingerprint_to_RDKitFP(smiles):
    mol = Chem.MolFromSmiles(smiles)
    rdkit_fp = RDKFingerprint(mol)
    print(rdkit_fp)
    return rdkit_fp

df[smiles_output_col] = df[smiles_col].apply(gen_fingerprint_to_RDKitFP)
df[solvent_output_col] = df[solvent_col].apply(gen_fingerprint_to_RDKitFP)

# 将 BitVector 转换为字符串并输出为 CSV 文件
df[smiles_output_col] = df[smiles_output_col].apply(lambda x: x.ToBitString())
df[solvent_output_col] = df[solvent_output_col].apply(lambda x: x.ToBitString())

df.to_csv('../../data/smiles_RDKitFP.csv', index=False)

df1= df.drop(labels=['molecule', 'solvent'], axis=1) #删除含有smiles的两列
df1.to_csv('../../data/RDKitFP.csv', index=False)



