import torch
import numpy as np
from flask import Flask, request, jsonify
from drug_process import smiles_to_graph
from model.net import TGDTA
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

def seq2int(line):
    return [CHARPROTSET[s] for s in line.upper()]

def target_get(protein):
    target_len = 1000
    target = seq2int(protein)
    if len(target) < target_len:
        target = np.pad(target, (0, target_len - len(target)))
    else:
        target = target[:target_len]
    return target

# 加载模型
model = TGDTA(compound_dim=128, protein_dim=128, gt_layers=10, gt_heads=8, out_dim=1, device=device).to(device)
state_dict = torch.load('model_state_dict.pth', map_location=device)
model.load_state_dict(state_dict)
model.eval()  # 切换到评估模式

@app.route('/predict_dta', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        smiles = data["smiles"]
        protein = data["protein"]

        # 预处理输入
        compound_graph = smiles_to_graph(smiles=smiles).to(device)
        target = torch.tensor(np.array(target_get(protein))).to(device).unsqueeze(0)

        # 进行预测
        with torch.no_grad():
            output = model(compound_graph, target)

        return jsonify({"affinity": output.item()})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5173, debug=True)
