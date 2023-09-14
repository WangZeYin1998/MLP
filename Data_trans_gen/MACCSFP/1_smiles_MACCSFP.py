import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# 读取 CSV 文件
df = pd.read_csv('../../data/data.csv')

# 定义 SMILES 列名和输出指纹列名
smiles_col = 'molecule'
solvent_col = 'solvent'
smiles_output_col = 'molecule_MACCSFP'
solvent_output_col = 'solvent_MACCSFP'
# 生成分子指纹
def gen_fingerprint_to_MACCSFP(smiles):
    mol = Chem.MolFromSmiles(smiles)
    MACCSFP = AllChem.GetMACCSKeysFingerprint(mol)
    return MACCSFP


df[smiles_output_col] = df[smiles_col].apply(gen_fingerprint_to_MACCSFP)
df[solvent_output_col] = df[solvent_col].apply(gen_fingerprint_to_MACCSFP)

df[smiles_output_col] = df[smiles_output_col].apply(lambda x: x.ToBitString())
df[solvent_output_col] = df[solvent_output_col].apply(lambda x: x.ToBitString())


df.to_csv('../../data/smiles_MACCSFP.csv', index=False)

df1= df.drop(labels=['molecule', 'solvent'],axis=1) #删除含有smiles的两列
df1.to_csv('../../data/MACCSFP.csv', index=False)

