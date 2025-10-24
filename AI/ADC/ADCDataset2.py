import dgl
import pandas as pd
import torch
import numpy as np
import pickle
import h5py
import joblib
# from adcutils import smiles2adjoin, molecular_fg
from dgl import load_graphs
from torch.utils.data import DataLoader, Dataset
from scipy import sparse as sp
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
# from drug_process import laplacian_positional_encoding
if torch.cuda.is_available():
    device = torch.device('cuda')

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64



# def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
#     X = np.zeros(MAX_SMI_LEN,dtype=np.int64())
#     for i, ch in enumerate(line[:MAX_SMI_LEN]):
#         X[i] = smi_ch_ind[ch]
#     return X


def seq2int(line):

    return [CHARPROTSET[s] for s in line.upper()]
def smiint(line):

    return [CHARISOSMISET[s] for s in line]
def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix(scipy_fmt='csr').astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    if EigVec.shape[1] < pos_enc_dim + 1:
        PadVec = np.zeros((EigVec.shape[0], pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
        EigVec = np.concatenate((EigVec, PadVec), 1)
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return g
class ADCDataset(Dataset):
    #蛋白质采用ADCnet的特征，药物采用vidta
    def __init__(self, dataset_fold=None):
        # self.dataset = dataset
        self.data = dataset_fold
        # self.compound_graph, _ = load_graphs(compound_graph)
        self.adcid=self.data[:, 0]
        self.playload = self.compound_graph_get(self.data[:, 4])
        self.linker = self.compound_graph_get(self.data[:, 5])
        # self.compound_graph = list(self.compound_graph)

        # self.protein_graph, _ = load_graphs(protein_graph)
        # self.protein_graph = list(self.protein_graph)
        # self.protein_embedding = np.load(protein_embedding, allow_pickle=True)
        # dataset是数据集路径，获取target嵌入
        self.heavy = self.heavy_get(self.adcid)
        self.light = self.light_get(self.adcid)
        self.antigen = self.antigen_get(self.adcid)
        # self.target_embed = target_embed_file
        # self.target = self.target_get(self.target_embed)

        # self.compound_id = np.load(compound_id, allow_pickle=True)
        # self.protein_id = np.load(protein_id, allow_pickle=True)
        # self.label = np.load(label, allow_pickle=True)
        
        self.dar = self.data[:, -2]
        self.label = self.data[:, -1]

        self.adc=self.vn_ADC_graph(self.adcid,self.dar)
        self.unicomp = self.unio_compound_graph_get(self.adcid)

    def __len__(self):
        return len(self.label)


    def __getitem__(self, idx):

        # compound_len = self.compound_graph[idx].num_nodes()
        # protein_len = self.protein_graph[idx].num_nodes()

        # target = self.data[idx, 1]
        # with h5py.File(self.target_embed, 'r') as h5f:
        #     target_embedding = h5f[target][...]
        # print(f"Current index: {idx}; Lengths -> heavy: {len(self.heavy)}, light: {len(self.light)}, "
        #       f"antigen: {len(self.antigen)}, playload: {len(self.playload)}, linker: {len(self.linker)}, "
        #       f"dar: {len(self.dar)}, label: {len(self.label)}, adc: {len(self.adc)}, unicomp: {len(self.unicomp)}")
        
        return self.heavy[idx], self.light[idx], self.antigen[idx], self.playload[idx], self.linker[idx], self.dar[idx], self.label[idx],self.adc[idx],self.unicomp[idx]

    ''' 整数编码 '''
    def protein_get(self, sequence):
        
        N = len(sequence)
        sequence_list = []
        # sequence = self.data[:, 1]
        max_len = 1000
        for content in sequence:
            embedding = seq2int(content)
            if len(embedding) < max_len:
                embedding = np.pad(embedding, (0, max_len - len(embedding)))
            else:
                embedding = embedding[:max_len]
            sequence_list.append(embedding)        
        return sequence_list
    def antigen_get(self, adcid):
        
        antigen_list = []
        
        # N = len(id_TVdataset)
        with open('dataset/Antigen_1280.pkl', 'rb') as f:  # 使用 pickle 从文件中加载字典对象
            proteinembs = pickle.load(f)
        for no, id in enumerate(adcid):

            
            # print('/'.join(map(str, [no + 1, N])))
            # compound_graph_TVdataset, _ = load_graphs('dataset/' + self.dataset + '/processed' + '/compound_graph/' + str(id) + '.bin')
            proteinemb = proteinembs[id]
            # print(id)
            # print(proteinemb)
            # return 1
            antigen_list.append(proteinemb)


        # return 1
        return antigen_list
    
    def heavy_get(self, adcid):
        
        heavy_list = []
        
        # N = len(id_TVdataset)
        with open('dataset/Heavy_1280.pkl', 'rb') as f:  # 使用 pickle 从文件中加载字典对象
            proteinembs = pickle.load(f)
        for no, id in enumerate(adcid):
            # print('/'.join(map(str, [no + 1, N])))
            # compound_graph_TVdataset, _ = load_graphs('dataset/' + self.dataset + '/processed' + '/compound_graph/' + str(id) + '.bin')
            proteinemb = proteinembs[id]
            heavy_list.append(proteinemb)
        return heavy_list
    def light_get(self, adcid):
        
        light_list = []
        
        # N = len(id_TVdataset)
        with open('dataset/Light_1280.pkl', 'rb') as f:  # 使用 pickle 从文件中加载字典对象
            proteinembs = pickle.load(f)
        for no, id in enumerate(adcid):
            # print('/'.join(map(str, [no + 1, N])))
            # compound_graph_TVdataset, _ = load_graphs('dataset/' + self.dataset + '/processed' + '/compound_graph/' + str(id) + '.bin')
            proteinemb = proteinembs[id]
            light_list.append(proteinemb)
        return light_list


    def compound_graph_get(self, smiles):
        # smiles_TVdataset = self.data[:, 0]
        compounds_graph = []
        # N = len(id_TVdataset)
        with open('dataset/processed/compound_graphs_vn.pkl', 'rb') as f:  # 使用 pickle 从文件中加载字典对象
            smiles2graph = pickle.load(f)
        for no, smile in enumerate(smiles):
            # print('/'.join(map(str, [no + 1, N])))
            # compound_graph_TVdataset, _ = load_graphs('dataset/' + self.dataset + '/processed' + '/compound_graph/' + str(id) + '.bin')
            compound_graph = smiles2graph[smile]
            compounds_graph.append(compound_graph[0])
        return compounds_graph
    def unio_compound_graph_get(self, adcid):
        # smiles_TVdataset = self.data[:, 0]
        compounds_graph = []
        # N = len(id_TVdataset)
        with open('dataset/processed/struc_graphs_vn.pkl', 'rb') as f:  # 使用 pickle 从文件中加载字典对象
            smiles2graph = pickle.load(f)
        for no, smile in enumerate(adcid):
            # print('/'.join(map(str, [no + 1, N])))
            # compound_graph_TVdataset, _ = load_graphs('dataset/' + self.dataset + '/processed' + '/compound_graph/' + str(id) + '.bin')
            compound_graph = smiles2graph[smile]
            compounds_graph.append(compound_graph[0])
        return compounds_graph


    def vn_ADC_graph(self,adcid,dars):
        adc_graph=[]
        for no, dar in enumerate(dars):
            g = dgl.DGLGraph()
            g.add_nodes(7)
            x = torch.randn(7, 128)
            
            src_list = [0,1,1,1,1,2,3,4,5,2,4,3]
            dst_list = [1,2,3,4,5,6,6,6,6,4,5,5]
            g.add_edges(src_list, dst_list)
            
            g = dgl.to_bidirected(g)

            g.ndata['atom'] = x
            g.edata['bond'] = torch.randn(24, 5)

            g.edata['bond'][g.edge_ids([0, 1], [1, 0])] = torch.Tensor([1,0,0,0,0]).repeat(2, 1)
            g.edata['bond'][g.edge_ids([1,1,1,1,2,3,4,5], [2,3,4,5,1,1,1,1])] =  torch.Tensor([0,1,0,0,dar]).repeat(8, 1)
            g.edata['bond'][g.edge_ids([2,3,4,4,5,5], [4,5,2,5,4,3])] =  torch.Tensor([0,0,1,0,0]).repeat(6, 1)
            # g.edata['edge'][g.edge_ids([1,1,1,1,2,3,4,5], [2,3,4,5,1,1,1,1])] =  torch.Tensor([0,1,0,0,0]).repeat(8, 1)
            g.edata['bond'][g.edge_ids([2,3,4,5,6,6,6,6], [6,6,6,6,2,3,4,5])] = torch.Tensor([0,0,0,1,0]).repeat(8, 1)
            # print(g.edata['bond'])

            #GIN不需要,模型的节点分配也要改，netw1 336
            g = dgl.add_nodes(g, 1)
            for i in range(g.num_nodes()-1):
                g = dgl.add_edges(g, i,g.num_nodes()-1) #i->vn
                g = dgl.add_edges(g, g.num_nodes()-1,i) #vn->i #增加无向图  指向i的边
            g = laplacian_positional_encoding(g, pos_enc_dim=8)
            # g = laplacian_positional_encoding(g, pos_enc_dim=8)
            # print(g.edges())

            adc_graph.append(g)
        return adc_graph



    def collate(self, sample):
        # batch_size = len(sample)

        heavy, light, antigen, playload, linker, dar, label,adc,unicomp= map(list, zip(*sample))
        # max_protein_len = max(protein_len)

        # for i in range(batch_size):
        #     if protein_embedding[i].shape[0] < max_protein_len:
        #         protein_embedding[i] = np.pad(protein_embedding[i], ((0, max_protein_len-protein_embedding[i].shape[0]), (0, 0)), mode='constant', constant_values = (0,0))

        playload = dgl.batch(playload)
        linker = dgl.batch(linker)
        adc = dgl.batch(adc)
        unicomp=dgl.batch(unicomp)
        # protein_graph = dgl.batch(protein_graph).to(device)
        # protein_embedding = torch.FloatTensor(protein_embedding).to(device)
        
        dar = torch.FloatTensor(dar)
        label = torch.LongTensor(label)
        return heavy, light, antigen, playload, linker,dar, label,adc,unicomp

class ADCDataset_all_emb(Dataset):
    #蛋白质、药物采用整数编码，COVAE
    def __init__(self, dataset_fold=None):
        # self.dataset = dataset
        self.data = dataset_fold

        self.adcid=self.data[:, 0]
        self.playload = self.smi_get(self.data[:, 4])
        self.linker = self.smi_get(self.data[:, 5])

        self.heavy = self.protein_get(self.data[:, 1])
        self.light = self.protein_get(self.data[:, 2])
        self.antigen = self.protein_get(self.data[:, 3])

        self.dar = self.data[:, -2]
        self.label = self.data[:, -1]


    def __len__(self):
        return len(self.label)


    def __getitem__(self, idx):

        # compound_len = self.compound_graph[idx].num_nodes()
        # protein_len = self.protein_graph[idx].num_nodes()

        # target = self.data[idx, 1]
        # with h5py.File(self.target_embed, 'r') as h5f:
        #     target_embedding = h5f[target][...]

        return self.heavy[idx], self.light[idx], self.antigen[idx], self.playload[idx], self.linker[idx], self.dar[idx], self.label[idx]

    ''' 整数编码 '''
    def protein_get(self, sequence):
        
        N = len(sequence)
        sequence_list = []
        # sequence = self.data[:, 1]
        max_len = 1000
        for content in sequence:
            embedding = seq2int(content)
            if len(embedding) < max_len:
                embedding = np.pad(embedding, (0, max_len - len(embedding)))
            else:
                embedding = embedding[:max_len]
            sequence_list.append(embedding)        
        return sequence_list
    def smi_get(self, smiles):
        
        smi_list = []
        max_len=100
        # N = len(id_TVdataset)
        for content in smiles:
            # print(content)
            embedding = smiint(content)
            if len(embedding) < max_len:
                embedding = np.pad(embedding, (0, max_len - len(embedding)))
            else:
                embedding = embedding[:max_len]
            smi_list.append(embedding)        
        return smi_list


    def collate(self, sample):
        # batch_size = len(sample)

        heavy, light, antigen, playload, linker, dar, label= map(list, zip(*sample))
        
        dar = torch.FloatTensor(dar)
        label = torch.LongTensor(label)
        return heavy, light, antigen, playload, linker, dar, label

class ADCDatasetmorgan(Dataset):
    def __init__(self, dataset_fold=None):
        self.data = dataset_fold
        self.adcid = self.data['id'].values  # 确保转换为 NumPy 数组
        self.playload_graph = compound_graph_get_graph(self.data['playload'].values)
        self.linker_graph = compound_graph_get_graph(self.data['linker'].values)
        self.playload = compound_graph_get(self.data['playload'].values)
        self.linker = compound_graph_get(self.data['linker'].values)

        self.heavy = heavy_get(self.adcid)
        self.light = light_get(self.adcid)
        self.antigen = antigen_get(self.adcid)

        self.dar = self.data['dar'].values
        self.label = self.data['label'].values
        self.adc = self.vn_ADC_graph(self.adcid, self.dar)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} out of range.")
        return (
            self.heavy[idx],
            self.light[idx],
            self.antigen[idx],
            self.playload[idx],
            self.linker[idx],
            self.dar[idx],
            self.label[idx],
            self.playload_graph[idx],
            self.linker_graph[idx],
            self.adc[idx]
        )

    def vn_ADC_graph(self, adcid, dars):
        adc_graph = []
        for no, dar in enumerate(dars):
            g = dgl.DGLGraph()
            g.add_nodes(7)
            x = torch.randn(7, 128)
            
            src_list = [0,1,1,1,1,2,3,4,5,2,4,3]
            dst_list = [1,2,3,4,5,6,6,6,6,4,5,5]
            g.add_edges(src_list, dst_list)
            
            g = dgl.to_bidirected(g)

            g.ndata['atom'] = x
            g.edata['bond'] = torch.randn(24, 5)

            g.edata['bond'][g.edge_ids([0, 1], [1, 0])] = torch.Tensor([1,0,0,0,0]).repeat(2, 1)
            g.edata['bond'][g.edge_ids([1,1,1,1,2,3,4,5], [2,3,4,5,1,1,1,1])] = torch.Tensor([0,1,0,0,dar]).repeat(8, 1)
            g.edata['bond'][g.edge_ids([2,3,4,4,5,5], [4,5,2,5,4,3])] = torch.Tensor([0,0,1,0,0]).repeat(6, 1)
            g.edata['bond'][g.edge_ids([2,3,4,5,6,6,6,6], [6,6,6,6,2,3,4,5])] = torch.Tensor([0,0,0,1,0]).repeat(8, 1)
            
            g = dgl.add_nodes(g, 1)
            for i in range(g.num_nodes()-1):
                g = dgl.add_edges(g, i, g.num_nodes()-1) #i->vn
                g = dgl.add_edges(g, g.num_nodes()-1, i) #vn->i
            
            g = laplacian_positional_encoding(g, pos_enc_dim=8)
            adc_graph.append(g)
        return adc_graph

    def collate(self, sample):
        heavy, light, antigen, playload, linker, dar, label, playload_graph, linker_graph, adc = map(list, zip(*sample))

        adc = dgl.batch(adc)
        playload_graph = dgl.batch(playload_graph)
        linker_graph = dgl.batch(linker_graph)        
        dar = torch.FloatTensor(dar)
        label = torch.LongTensor(label)
        return heavy, light, antigen, playload, linker, dar, label, playload_graph, linker_graph, adc

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def compound_graph_get( smiles_list):
    morgan_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("无效的 SMILES 字符串")
        else:
            morgan = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        morgan_list.append(morgan)
    return np.array(morgan_list)

def antigen_get( adcid):
    antigen_list = []
    with open('dataset/Antigen_1280.pkl', 'rb') as f:
        proteinembs = pickle.load(f)
    for id in adcid:
        proteinemb = proteinembs[id]
        antigen_list.append(proteinemb)
    return antigen_list

def heavy_get( adcid):
    heavy_list = []
    with open('dataset/Heavy_1280.pkl', 'rb') as f:
        proteinembs = pickle.load(f)
    for id in adcid:
        proteinemb = proteinembs[id]
        heavy_list.append(proteinemb)
    return heavy_list

def light_get( adcid):
    light_list = []
    with open('dataset/Light_1280.pkl', 'rb') as f:
        proteinembs = pickle.load(f)
    for id in adcid:
        proteinemb = proteinembs[id]
        light_list.append(proteinemb)
    return light_list
def compound_graph_get_graph( smiles):
    # smiles_TVdataset = self.data[:, 0]
    compounds_graph = []
    # N = len(id_TVdataset)
    with open('dataset/processed/compound_graphs_vn.pkl', 'rb') as f:  # 使用 pickle 从文件中加载字典对象
        smiles2graph = pickle.load(f)
    for no, smile in enumerate(smiles):
        # print('/'.join(map(str, [no + 1, N])))
        # compound_graph_TVdataset, _ = load_graphs('dataset/' + self.dataset + '/processed' + '/compound_graph/' + str(id) + '.bin')
        compound_graph = smiles2graph[smile]
        compounds_graph.append(compound_graph[0])
    return compounds_graph
class ADCPCA:
    def __init__(self, df, n_components=128):
        self.data = df
        self.n_components = n_components
        # self.data = dataset_fold
        self.adcid = self.data[:, 0]
        self.playload = compound_graph_get(self.data[:, 4])
        self.linker = compound_graph_get(self.data[:, 5])
        self.heavy = heavy_get(self.adcid)
        self.light = light_get(self.adcid)
        self.antigen = antigen_get(self.adcid)
        self.dar = self.data[:, -2]
        self.dar = np.expand_dims(self.dar, axis=1)
        # print(f"heavy shape: {np.array(self.heavy).shape}")
        # print(f"light shape: {np.array(self.light).shape}")
        # print(f"antigen shape: {np.array(self.antigen).shape}")
        # print(f"playload shape: {np.array(self.playload).shape}")
        # print(f"linker shape: {np.array(self.linker).shape}")
        # print(f"dar shape: {np.array(self.dar).shape}")
        # 提取特征并融合
        fused_vector = self.fuse_features()
        print(f"Fused vector shape before PCA: {fused_vector.shape}")
        
        # 标准化处理
        scaler = StandardScaler()
        fused_vector_scaled = scaler.fit_transform(fused_vector)
        
        # PCA 降维
        pca = PCA(n_components=min(self.n_components, fused_vector_scaled.shape[0], fused_vector_scaled.shape[1]))
        self.fused_vector_reduced = pca.fit_transform(fused_vector_scaled)
        print(f"Fused vector shape after PCA: {self.fused_vector_reduced.shape}")
        
        # 更新 DataFrame，加入降维后的特征
        self.data = np.hstack((self.data, self.fused_vector_reduced))
        # self.data = np.hstack((self.data, fused_vector))#不使用pca

        # self.data=fused_vector#不使用pca

    def fuse_features(self):
        # 模拟数据融合，返回一个形状为 (样本数, 特征数) 的 NumPy 数组
        fused_vector = np.concatenate([self.heavy, self.light, self.antigen, self.playload, self.linker, self.dar], axis=1)

        return fused_vector

    def get_data(self):
        return self.data



class ADCPCAn:
    def __init__(self, n_components=128):
        """
        PCA降维处理器，支持对训练集训练PCA模型并将其应用于测试集。
        """
        self.n_components = n_components
        self.scaler = None
        self.pca = None

    def fit_transform(self, train_data):
        """
        使用训练集数据训练PCA并降维。
        参数:
            train_data: 训练集的原始数据。
        返回:
            降维后的训练集数据。
        """
        # 提取并融合特征
        heavy = heavy_get(train_data['id'])
        light = light_get(train_data['id'])
        antigen = antigen_get(train_data['id'])
        playload = compound_graph_get(train_data['playload'])
        linker = compound_graph_get(train_data['linker'])
        dar = np.expand_dims(train_data['dar'], axis=1)
        fused_vector = np.concatenate([heavy, light, antigen, playload, linker, dar], axis=1)

        # 标准化
        self.scaler = StandardScaler()
        fused_vector_scaled = self.scaler.fit_transform(fused_vector)

        # PCA降维
        self.pca = PCA(n_components=128)
        fused_vector_reduced = self.pca.fit_transform(fused_vector_scaled)

        # 返回降维后的训练数据
        return np.hstack((train_data, fused_vector_reduced))

    def transform(self, test_data):
        """
        使用训练好的PCA模型对测试集降维。
        参数:
            test_data: 测试集的原始数据。
        返回:
            降维后的测试集数据。
        """
        if self.pca is None or self.scaler is None:
            raise ValueError("PCA model and scaler must be trained on the training data first!")

        # 提取并融合特征
        heavy = heavy_get(test_data['id'])
        light = light_get(test_data['id'])
        antigen = antigen_get(test_data['id'])
        playload = compound_graph_get(test_data['playload'])
        linker = compound_graph_get(test_data['linker'])
        dar = np.expand_dims(test_data['dar'], axis=1)
        fused_vector = np.concatenate([heavy, light, antigen, playload, linker, dar], axis=1)

        # 标准化和降维
        fused_vector_scaled = self.scaler.transform(fused_vector)
        fused_vector_reduced = self.pca.transform(fused_vector_scaled)

        # 返回降维后的测试数据
        return np.hstack((test_data, fused_vector_reduced))

    def save_model(self, filepath_scaler, filepath_pca):
        """
        保存训练好的Scaler和PCA模型到文件。
        参数:
            filepath_scaler: Scaler模型保存路径。
            filepath_pca: PCA模型保存路径。
        """
        joblib.dump(self.scaler, filepath_scaler)
        joblib.dump(self.pca, filepath_pca)

    def load_model(self, filepath_scaler, filepath_pca):
        """
        从文件加载Scaler和PCA模型。
        参数:
            filepath_scaler: Scaler模型加载路径。
            filepath_pca: PCA模型加载路径。
        """
        self.scaler = joblib.load(filepath_scaler)
        self.pca = joblib.load(filepath_pca)


class ADCDatasetmorganPCA(Dataset):
    def __init__(self, dataset_fold=None):
        self.data = dataset_fold
        print(np.array(self.data[0]).shape)
        self.adcid = self.data[:, 0]
        self.playload = compound_graph_get_graph(self.data[:, 4])
        self.linker = compound_graph_get_graph(self.data[:, 5])
        self.heavy = heavy_get(self.adcid)
        self.light = light_get(self.adcid)
        self.antigen = antigen_get(self.adcid)
        self.dar = self.data[:, 6]
        self.label = self.data[:,7]
        self.adc = self.vn_ADC_graph(self.adcid, self.dar)
        self.dar = np.expand_dims(self.dar, axis=1)
        # self.fused_vector_reduced =self.data[:, 9:]#cold
        self.fused_vector_reduced =self.data[:, 8:]#

        # print(f"heavy shape: {np.array(self.heavy).shape}")
        # print(f"light shape: {np.array(self.light).shape}")
        # print(f"antigen shape: {np.array(self.antigen).shape}")
        # print(f"playload shape: {np.array(self.playload).shape}")
        # print(f"linker shape: {np.array(self.linker).shape}")
        # print(f"dar shape: {np.array(self.dar).shape}")

        # 进行 PCA 降维
        # fused_vector = self.fuse_features()
        # scaler = StandardScaler()
        # fused_vector_scaled = scaler.fit_transform(fused_vector)
        # n_samples, n_features = fused_vector_scaled.shape
        # # n_components = min(128, n_samples, n_features)
        # pca = PCA(n_components=128)
        # self.fused_vector_reduced = pca.fit_transform(fused_vector_scaled)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (
            self.heavy[idx],
            self.light[idx],
            self.antigen[idx],
            self.playload[idx],
            self.linker[idx],
            self.dar[idx],
            self.label[idx],
            self.adc[idx],
            self.fused_vector_reduced[idx]
        )

    def fuse_features(self):
        # 融合所有特征
        heavyo = np.stack(self.heavy)
        lighto = np.stack(self.light)
        antigeno = np.stack(self.antigen)
        playload_graph = self.playload
        linker_graph = self.linker
        dar = np.expand_dims(self.dar, axis=1).repeat(128, axis=1)
        # fused_vector = np.concatenate([heavyo, lighto, antigeno, playload_graph, linker_graph, dar], axis=1)
        fused_vector = np.concatenate([self.heavy, self.light, self.antigen, self.playload, self.linker, self.dar], axis=1)
        # print(f"dar shape: {np.array(heavyo).shape}")
        # print(f"dar shape: {np.array(lighto).shape}")
        # print(f"dar shape: {np.array(antigeno).shape}")
        # print(f"dar shape: {np.array(playload_graph).shape}")
        # print(f"dar shape: {np.array(linker_graph).shape}")
        # print(f"dar shape: {np.array(dar).shape}")
        # print(f"dar shape: {np.array(fused_vector).shape}")
        return fused_vector


    def vn_ADC_graph(self, adcid, dars):
        adc_graph = []
        for dar in dars:
            g = dgl.DGLGraph()
            g.add_nodes(7)
            x = torch.randn(7, 128)
            src_list = [0,1,1,1,1,2,3,4,5,2,4,3]
            dst_list = [1,2,3,4,5,6,6,6,6,4,5,5]
            g.add_edges(src_list, dst_list)
            g = dgl.to_bidirected(g)
            g.ndata['atom'] = x
            g.edata['bond'] = torch.randn(24, 5)
            g.edata['bond'][g.edge_ids([0, 1], [1, 0])] = torch.Tensor([1,0,0,0,0]).repeat(2, 1)
            g.edata['bond'][g.edge_ids([1,1,1,1,2,3,4,5], [2,3,4,5,1,1,1,1])] =  torch.Tensor([0,1,0,0,dar]).repeat(8, 1)
            g.edata['bond'][g.edge_ids([2,3,4,4,5,5], [4,5,2,5,4,3])] =  torch.Tensor([0,0,1,0,0]).repeat(6, 1)
            g.edata['bond'][g.edge_ids([2,3,4,5,6,6,6,6], [6,6,6,6,2,3,4,5])] = torch.Tensor([0,0,0,1,0]).repeat(8, 1)
            g = dgl.add_nodes(g, 1)
            for i in range(g.num_nodes()-1):
                g = dgl.add_edges(g, i, g.num_nodes()-1)
                g = dgl.add_edges(g, g.num_nodes()-1, i)
            g = laplacian_positional_encoding(g, pos_enc_dim=8)
            adc_graph.append(g)
        return adc_graph

    def collate(self, sample):
        heavy, light, antigen, playload, linker, dar, label, adc, fused_vector_reduced = map(list, zip(*sample))
        adc = dgl.batch(adc)
        playload= dgl.batch(playload)
        linker= dgl.batch(linker)
        # dar = np.array(dar)
        dar = torch.FloatTensor(dar)
        # label = torch.LongTensor(label)
        label = torch.FloatTensor(label)#bce
        fused_vector_reduced = torch.FloatTensor(fused_vector_reduced)
        return heavy, light, antigen, playload, linker, dar, label, adc, fused_vector_reduced



str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'P':  9,
         'I': 10,'Na': 11,'B':12,'Se':13,'Si':14,'<unk>':15,'<mask>':16,'<global>':17}
num2str =  {i:j for j,i in str2num.items()}
