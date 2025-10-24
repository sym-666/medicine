import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pickle
CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25


def seq2int(line):

    return [CHARPROTSET[s] for s in line.upper()]

def seq2embed(sequence):

        # sequence = self.data[:, 1]
    max_len = 1000
    
    embedding = seq2int(sequence)
    if len(embedding) < max_len:
        embedding = np.pad(embedding, (0, max_len - len(embedding)))
    else:
        embedding = embedding[:max_len]
    
    return embedding
def antigen_get(id):
    # antigen_list = []
    with open('dataset/Antigen_1280.pkl', 'rb') as f:
        proteinembs = pickle.load(f)
    # for id in adcid:
    proteinemb = proteinembs[id]
        # antigen_list.append(proteinemb)
    return proteinemb

def heavy_get(id):
    # heavy_list = []
    with open('dataset/Heavy_1280.pkl', 'rb') as f:
        proteinembs = pickle.load(f)
    # for id in adcid:
    proteinemb = proteinembs[id]
        # antigen_list.append(proteinemb)
    return proteinemb

def light_get(id):
    # light_list = []
    with open('dataset/Light_1280.pkl', 'rb') as f:
        proteinembs = pickle.load(f)
    # for id in adcid:
    proteinemb = proteinembs[id]
        # antigen_list.append(proteinemb)
    return proteinemb

def smi2morgan(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("无效的 SMILES 字符串")
    else:
        # 生成摩根指纹 (Morgan Fingerprint)，这里的半径为2，位数为2048
        # 你可以根据需要调整半径和位数
        morgan = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        return np.array(morgan)
    
def smiles_combine(smiles1, smiles2):
    # 将 SMILES 转换为分子对象
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # 获取两个分子的公共骨架
    bonded_mol = Chem.CombineMols(mol1, mol2)

    # 创建可修改的分子对象
    combined_mol = Chem.RWMol(bonded_mol)

    # 在分子的末端添加单键连接
    combined_mol.AddBond(mol1.GetNumAtoms() - 1, mol1.GetNumAtoms(), order=Chem.BondType.SINGLE)

    # 生成最终的分子
    final_mol = combined_mol.GetMol()

    # 打印结果的 SMILES 表达式
    # final_smiles = Chem.MolToSmiles(final_mol)

    return final_mol

def union2morgan(playload, linker, id, flag):
    if flag == 1:
        mol = Chem.MolFromMolFile('dataset/structure/' + id + '.sdf')
    else:
        mol = smiles_combine(linker, playload)
    if mol is None:
        print("无效的 SMILES 字符串")
    else:
        # 生成摩根指纹 (Morgan Fingerprint)，这里的半径为2，位数为2048
        # 你可以根据需要调整半径和位数
        morgan = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        morgan = list(morgan)
        return morgan

def ML_Dataset(TVdataset):
    feature = []
    # self.adcid = TVdataset[:, 0]
    # print(TVdataset.shape[0])
    # print(TVdataset[0, 0])
    for i in range(TVdataset.shape[0]):
        # f1 = seq2embed(TVdataset[i, 1])
        # f2 = seq2embed(TVdataset[i, 2])
        # f3 = seq2embed(TVdataset[i, 3])
        f1 = antigen_get(TVdataset[i, 0])
        f2 = heavy_get(TVdataset[i, 0])
        f3 = light_get(TVdataset[i, 0])
        f4 = smi2morgan(TVdataset[i, 4])
        f5 = smi2morgan(TVdataset[i, 5])
        f6=np.array([TVdataset[i, 6]])
        # f6 = union2morgan(TVdataset[i, 4], TVdataset[i, 5], TVdataset[i, 0], TVdataset[i, 8])
        # f.extend([f1, f2, f3, f4, f5, f6, TVdataset[i, 6]])
        f = np.concatenate((f1, f2, f3 ,f4, f5,f6))
        feature.append(f)
    return np.array(feature)
