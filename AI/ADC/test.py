import pandas as pd
import time
import os
import random
import numpy as np
import math
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import roc_auc_score,confusion_matrix,precision_recall_curve,auc
# from metrics import *
from model.netw1 import UNADC_test
from sklearn.utils import resample
# import torch
import esm
# import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
# from model.covae import net
from loss import FocalLoss
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.model_selection import KFold
# from ADCDataset2 import compound_graph_get_graph
from drug_process import smiles_to_graph
import joblib
import dgl
from scipy import sparse as sp
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
device = torch.device("cpu")  # 强制 DGL 使用 CPU
def score(y_test, y_pred):
    if np.isnan(y_pred).any():
        return 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0,0
    auc_roc_score = roc_auc_score(y_test, y_pred)
    prec, recall, _ = precision_recall_curve(y_test, y_pred)
    prauc = auc(recall, prec)
    y_pred_print = [round(y, 0) for y in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_print).ravel()
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = (tp + tn) / (tp + fn + tn + fp)
    mcc = (tp * tn - fn * fp) / math.sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))
    P = tp / (tp + fp)
    F1 = (P * se * 2) / (P + se)
    BA = (se + sp) / 2
    PPV = tp / (tp + fp)
    NPV = tn / (fn + tn)
    return tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV

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
def vn_ADC_graph(dar):
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
    # adc_graph.append(g)
    return g


def get_esm(sequence, id='sequence'):
    """
    接收一个氨基酸序列并返回其ESM嵌入表示。

    参数:
        sequence (str): 氨基酸序列。
        id (str): 用于标识序列的ID，默认为'sequence'。

    返回:
        torch.Tensor: 序列的ESM嵌入表示。
    """

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data
    data = [('protein', sequence)]  # 构造一个包含ID和序列的元组列表

    # Convert the data into a format that can be fed into the model
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representation = token_representations[0, 1 : batch_lens[0] - 1].mean(0)

    return sequence_representation

def finger_get( smi):
    # morgan_list = []
    # for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print("无效的 SMILES 字符串")
    else:
        morgan = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        # morgan_list.append(morgan)
    return torch.tensor(np.array(morgan))


def pca_transform(input,filepath_scaler, filepath_pca):
    scaler = joblib.load(filepath_scaler)
    pca = joblib.load(filepath_pca)
    # 标准化和降维
    fused_vector_scaled = scaler.transform(input)
    fused_vector_reduced = pca.transform(fused_vector_scaled)

    # 返回降维后的测试数据
    return fused_vector_reduced


@app.route('/predict_adc', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        heavy_seq = data.get('heavy_seq')
        light_seq = data.get('light_seq')
        antigen_seq = data.get('antigen_seq')
        payload_s = data.get('payload_s')
        linker_s = data.get('linker_s')
        dar_str = data.get('dar_str')

        # 转换 DAR 值为浮动类型张量
        dar_value = torch.tensor([float(dar_str)], dtype=torch.float32)

        # 获取 ESM 嵌入表示
        heavy_emb = get_esm(heavy_seq)
        light_emb = get_esm(light_seq)
        antigen_emb = get_esm(antigen_seq)

        # 获取指纹表示
        payload = finger_get(payload_s)
        linker = finger_get(linker_s)

        # 合并输入数据
        input_data = torch.cat([heavy_emb.unsqueeze(0), light_emb.unsqueeze(0), antigen_emb.unsqueeze(0), payload.unsqueeze(0), linker.unsqueeze(0), dar_value.unsqueeze(0)], dim=1)

        # PCA 变换
        pca_f = pca_transform(input_data, 'scaler.joblib', 'pca.joblib')

        # 获取图结构
        pca_f = torch.FloatTensor(pca_f)
        payload_g = smiles_to_graph(payload_s)
        linker_g = smiles_to_graph(linker_s)

        # 构建 ADC 图
        adc_g = vn_ADC_graph(dar_value)

        # 加载模型
        model = UNADC_test(device=device, compound_dim=128, protein_dim=128, gt_layers=3, gt_heads=4, out_dim=1).to(device)
        state_dict = torch.load('UNADC_model.pth', map_location=device)
        model.load_state_dict(state_dict['model'])
        model.eval()

        with torch.no_grad():
            output = model(
                heavy=heavy_emb,
                light=light_emb,
                antigen=antigen_emb,
                playload_graph=payload_g,
                linker_graph=linker_g,
                dar=dar_value,
                adc=adc_g,
                pca=pca_f
            )

            # 处理预测结果
            preds_class = torch.where(torch.sigmoid(output) >= 0.5, 1, 0).cpu().numpy()
            preds_class = preds_class.squeeze()

        return jsonify({"prediction": str(preds_class)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5173, debug=True)