
'''
@Project ：MLP 
@File    ：FP_View.py
@IDE     ：PyCharm 
@Author  ：bruce
@Date    ：2023/9/13 10:07 
'''
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

data = 'data/data.csv'
data = pd.read_csv(data)
mol = data['molecule'][0]
mol = Chem.MolFromSmiles(mol)
bi = {}
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2)
fp = Draw.MolsToGridImage([mol], subImgSize=(200, 300))
plt.imshow(fp)
plt.show()