from flask import Flask, request, jsonify
import torch
import numpy as np
from tape import TAPETokenizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 模型地址
pretrain_model_path = './pretrain_bert.models'
model_path = './model1231_epoch30.pth'
aaindex_path = './aaindex_pca.csv'


def get_aaindex_feature(seqs, aaindex_path, device, max_len=256):
    df = pd.read_csv(aaindex_path, header=0)
    scaler = MinMaxScaler()
    columns_to_normalize = df.columns[1:]
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    column_names = df.columns[list(range(1, 21))]
    groups = df.groupby('AA')
    results_dict = {}
    for group_name, group in groups:
        num_rows = group.shape[0]
        values_list = []
        for i in range(num_rows):
            row = group.iloc[i]
            values_list.append(row[column_names].tolist())
        results_dict[group_name] = values_list
    aaindex_feature = []
    for seq in seqs:
        seq_feature = []
        for aa in seq:
            if aa in results_dict.keys():
                seq_feature.append(torch.tensor(results_dict[aa][0]))
        if len(seq_feature) < max_len:
            for i in range(max_len - len(seq_feature)):
                seq_feature.append(torch.zeros(20))
        else:
            seq_feature = seq_feature[:max_len]
        aaindex_feature.append(torch.stack(seq_feature).unsqueeze(0).to(device))
    return aaindex_feature


def get_bert_feature(seqs, pretrain_model_path, device):
    pretrain_model = torch.load(pretrain_model_path)
    for param in pretrain_model.parameters():
        param.requires_grad = False
    pretrain_model.to(device)
    pretrain_model.eval()
    seq_tokenizer = TAPETokenizer(vocab='iupac')
    seq_embeddings = []
    for seq in seqs:
        token_ids = torch.tensor(np.array([seq_tokenizer.encode(seq)]), dtype=torch.long).to(device)
        seq_embeddings.append(pretrain_model(token_ids)[1][0].unsqueeze(0))
    return seq_embeddings


# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path)
model.to(device)
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        seqs = [data["seq_light"], data["seq_heavy"], data["seq_antigen"]]

        bert_feature = get_bert_feature(seqs, pretrain_model_path, device)
        aaindex_feature = get_aaindex_feature(seqs, aaindex_path, device)

        ph = model.predict(
            aaindex_feature[0], aaindex_feature[1], aaindex_feature[2],
            bert_feature[0], bert_feature[1], bert_feature[2]
        )
        p_hat = (ph * (16.9138 - 5.0400)) + 5.0400
        return jsonify({"prediction": p_hat.item()})
    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5173, debug=True)
