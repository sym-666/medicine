import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from model import gt_net_compound


if torch.cuda.is_available():
    device = torch.device('cuda')


# class SelfAttention(nn.Module):
#     def __init__(self, hid_dim, n_heads, dropout):
#         super().__init__()

#         self.hid_dim = hid_dim
#         self.n_heads = n_heads
#         assert hid_dim % n_heads == 0

#         self.w_q = nn.Linear(hid_dim, hid_dim)
#         self.w_k = nn.Linear(hid_dim, hid_dim)
#         self.w_v = nn.Linear(hid_dim, hid_dim)
#         self.fc = nn.Linear(hid_dim, hid_dim)
#         self.do = nn.Dropout(dropout)
#         self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

#     def forward(self, query, key, value, mask=None):
#         bsz = query.shape[0]
#         # query = key = value [batch size, sent len, hid dim]
#         Q = self.w_q(query)
#         K = self.w_k(key)
#         V = self.w_v(value)

#         # Q, K, V = [batch size, sent len, hid dim]
#         Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
#         K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
#         V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
#         # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
#         # Q = [batch size, n heads, sent len_q, hid dim // n heads]
#         energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

#         # energy = [batch size, n heads, sent len_Q, sent len_K]
#         if mask is not None:
#             energy = energy.masked_fill(mask == 0, -1e10)

#         attention = self.do(F.softmax(energy, dim=-1))
#         # attention = [batch size, n heads, sent len_Q, sent len_K]
#         x = torch.matmul(attention, V)
#         # x = [batch size, n heads, sent len_Q, hid dim // n heads]
#         x = x.permute(0, 2, 1, 3).contiguous()
#         # x = [batch size, sent len_Q, n heads, hid dim // n heads]
#         x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
#         # x = [batch size, src sent len_Q, hid dim]
#         x = self.fc(x)
#         # x = [batch size, sent len_Q, hid dim]
#         return x

class DTF(nn.Module):
    def __init__(self, channels=128, r=4):
        super(DTF, self).__init__()
        inter_channels = int(channels // r)

        self.att1 = nn.Sequential(
            nn.Linear(channels, inter_channels),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, channels),
            nn.BatchNorm1d(channels)
        )

        self.att2 = nn.Sequential(
            nn.Linear(channels, inter_channels),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, channels),
            nn.BatchNorm1d(channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, fd, fp):
        w1 = self.sigmoid(self.att1(fd + fp))
        # print('w1:', w1.shape)
        fout1 = fd * w1 + fp * (1 - w1)

        w2 = self.sigmoid(self.att2(fout1))
        # print('w2', w2.shape)
        # fd = fd * w2
        # fp = fp * (1 - w2)
        fout2 = fd * w2 + fp * (1 - w2)
        
        w3 = self.sigmoid(fout2)
        fout = w3 * fout2 + (1 - w3) * (fd + fp)

        # fout = torch.cat([fout1, fout2], dim=1)
        return fout
    
class FF3(nn.Module):
    def __init__(self, in_size=128, hidden_size=64):
        super(FF3, self).__init__()

        self.project_h = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_l = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_g = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, h, l, g):
        h = self.project_h(h)
        l = self.project_l(l)
        g = self.project_g(g)
        weight = torch.cat((h, l, g), 1)
        weight = torch.softmax(weight, dim=1)
        return weight
    
class FF2(nn.Module):
    def __init__(self, channels=128, r=4):
        super(FF2, self).__init__()
        inter_channels = int(channels // r)

        self.att = nn.Sequential(
            nn.Linear(channels, inter_channels),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, channels),
            nn.BatchNorm1d(channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, p, l):
        w = self.sigmoid(self.att(p + l))
        p = p * w
        l = l * (1 - w) 
        out = torch.cat([p, l], dim=1)
        return out

class FF5(nn.Module):
    def __init__(self, in_size=128, hidden_size=64):
        super(FF5, self).__init__()

        self.project_h = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_light = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_g = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_p = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_linker = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, h, light, g, p, linker):
        h = self.project_h(h)
        light = self.project_light(light)
        g = self.project_g(g)
        p = self.project_p(p)
        linker = self.project_linker(linker)
        weight = torch.cat((h, light, g, p, linker), 1)
        weight = torch.softmax(weight, dim=1)
        return weight

class TGADC(nn.Module):
    def __init__(self, compound_dim=128, protein_dim=128, gt_layers=10, gt_heads=8, out_dim=1):
        super(TGADC, self).__init__()
        self.compound_dim = compound_dim
        self.protein_dim = protein_dim
        self.n_layers = gt_layers
        self.n_heads = gt_heads

        # self.crossAttention = SelfAttention(hid_dim=self.compound_dim, n_heads=1, dropout=0.2)

        self.compound_encoder = gt_net_compound.GraphTransformer(device, n_layers=gt_layers, node_dim=44, edge_dim=10, hidden_dim=compound_dim,
                                                        out_dim=compound_dim, n_heads=gt_heads, in_feat_dropout=0.0, dropout=0.2, pos_enc_dim=8)
        # self.protein_gt = gt_net_protein.GraphTransformer(device, n_layers=gt_layers, node_dim=41, edge_dim=5, hidden_dim=protein_dim,
                                                    #    out_dim=protein_dim, n_heads=gt_heads, in_feat_dropout=0.0, dropout=0.2, pos_enc_dim=8)
        self.protein_embed = nn.Embedding(26, self.protein_dim, padding_idx=0)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.protein_dim, out_channels=self.protein_dim, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.protein_dim, out_channels=self.protein_dim * 2, kernel_size=8),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.protein_dim * 2, out_channels=self.protein_dim, kernel_size=12),
            nn.ReLU(),
        )
        # self.Protein_CNNs = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=512,  kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=512, out_channels=256,  kernel_size=7, padding=3),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=256, out_channels=768,  kernel_size=11, padding=5),
        #     nn.ReLU(),
        # )
        self.protein_max_pool = nn.MaxPool1d(979)
        # self.protein_max_pool = nn.MaxPool1d(768)
        self.compound_max_pool = nn.AdaptiveMaxPool1d(1)
        # self.protein_embedding_fc = nn.Linear(320, self.protein_dim)
        # self.protein_fc = nn.Linear(self.protein_dim * 2, self.protein_dim)

        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()
        # self.joint_attn_prot, self.joint_attn_comp = nn.Linear(979, compound_dim), nn.Linear(compound_dim, compound_dim)
        # self.modal_fc = nn.Linear(protein_dim*2, protein_dim)

        # self.fc_out = nn.Linear(compound_dim, out_dim)

        # self.compound_fc = nn.Linear(compound_dim, 768)
        # self.protein_fc = nn.Linear(protein_dim, 768)
        self.dtf = DTF()
        self.FF3 = FF3()
        self.FF2 = FF2()
        self.FF5 = FF5()

        self.classifier = nn.Sequential(
            nn.Linear(641, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim)
        )


    def dgl_split(self, bg, feats):
        max_num_nodes = int(bg.batch_num_nodes().max())
        batch = torch.cat([torch.full((1, x.type(torch.int)), y) for x, y in zip(bg.batch_num_nodes(), range(bg.batch_size))],
                       dim=1).reshape(-1).type(torch.long).to(bg.device)
        cum_nodes = torch.cat([batch.new_zeros(1), bg.batch_num_nodes().cumsum(dim=0)])
        idx = torch.arange(bg.num_nodes(), dtype=torch.long, device=bg.device)
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
        size = [bg.batch_size * max_num_nodes] + list(feats.size())[1:]
        out = feats.new_full(size, fill_value=0)
        out[idx] = feats
        out = out.view([bg.batch_size, max_num_nodes] + list(feats.size())[1:])
        return out

    def get_fn_feature(self, bg, feats):
        num_nodes = bg.batch_num_nodes()
        
        # num_nodes_list = num_nodes.numpy().tolist()
        # out=torch.tensor([])
        out=[]
        s=num_nodes[0]
        # print(num_nodes.sum())
        # out=feats[s-1]
        s=0
        for i in range(len(num_nodes)):
            s=s+num_nodes[i].item()
            out.append(feats[s-1])
            
        out = torch.stack(out)        
        return out

    def get_embed(self, sequence):
        embed = self.protein_embed(sequence) # [128,1000,128]
        embed = embed.permute(0, 2, 1) 
        embed_feats = self.Protein_CNNs(embed) # [128,128,979]
        feature = self.protein_max_pool(embed_feats).squeeze(2)
        return feature

    def forward(self, heavy, light, antigen, playload_graph, linker_graph, dar):
        playload_feat = self.compound_encoder(playload_graph)  # 3981*128
        linker_feat = self.compound_encoder(linker_graph)
        # compound_feat_x = self.dgl_split(compound_graph, compound_feat) # [128,40,128]
        # compound_feats = compound_feat_x.permute(0, 2, 1)
        # compound = self.compound_max_pool(compound_feats).squeeze(2)
        playload = self.get_fn_feature(playload_graph, playload_feat)
        linker = self.get_fn_feature(linker_graph, linker_feat)
        heavy = self.get_embed(heavy)
        light = self.get_embed(light)
        antigen = self.get_embed(antigen)
        dar = dar.reshape(-1, 1)
        # target = self.Protein_CNNs(target)
        # protein = self.protein_max_pool(target).squeeze(2)
        # heavyembed = self.protein_embed(heavy) # [128,1000,128]
        # heavyembed = proteinembed.permute(0, 2, 1) 
        # protein_feats = self.Protein_CNNs(proteinembed) # [128,128,979]
        # protein = self.protein_max_pool(protein_feats).squeeze(2)
        # protein_feat_x = self.dgl_split(protein_graph, protein_feat)
        # protein_embedding = self.protein_embedding_fc(protein_embedding)
        # protein_feats = self.crossAttention(protein_embedding, protein_feat_x, protein_feat_x)

        # compound-protein interaction
        # inter_comp_prot = self.sigmoid(torch.einsum('bij,bkj->bik', self.joint_attn_prot(self.relu(protein_feats)), self.joint_attn_comp(self.relu(compound_feats))))
        # inter_comp_prot_sum = torch.einsum('bij->b', inter_comp_prot)
        # inter_comp_prot = torch.einsum('bij,b->bij', inter_comp_prot, 1/inter_comp_prot_sum)

        # # compound-protein joint embedding
        # cp_embedding = self.tanh(torch.einsum('bij,bkj->bikj', protein_feats, compound_feats))
        # cp_embedding = torch.einsum('bijk,bij->bk', cp_embedding, inter_comp_prot)

        # 改变输入维度为768维
        # compound = self.compound_fc(compound)
        # protein = self.protein_fc(protein)
        # fused_vector = self.dtf(compound, protein)

        # fused_vector = torch.cat([compound, protein], dim=1)
        '''cat'''
        # fused_vector = torch.cat([heavy, light, antigen, playload, linker, dar], dim=1)

        '''FF3'''
        # # 重链、轻链、抗原的融合
        # weight = self.FF3(heavy, light ,antigen)
        # emb1 = torch.stack([heavy, light ,antigen], dim=1)
        # weight= weight.unsqueeze(dim =2)
        # emb1 = (weight * emb1).reshape(-1, 3* 128)
        # # playload、linker融合
        # emb2 = self.FF2(playload, linker)
        # fused_vector = torch.cat([emb1, emb2, dar], dim=1)
        # x = self.classifier(fused_vector)

        '''FF5'''
        weight = self.FF5(heavy, light ,antigen, playload, linker)
        fused_vector = torch.stack([heavy, light ,antigen, playload, linker], dim=1)
        weight= weight.unsqueeze(dim =2)
        fused_vector= (weight * fused_vector).reshape(-1, 5* 128)
        fused_vector = torch.cat([fused_vector, dar], dim=1)
        
        x = self.classifier(fused_vector)
        return x
    
# class FullyConnected(nn.Module):
#     def __init__(self, MDGTDTInet):
#         super(FullyConnected, self).__init__()
#         self.DTencoder = MDGTDTInet
#         self.classifier = nn.Sequential(
#             nn.Linear(768, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(1024, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(1024, 256),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(256, 1)
#         )

#     def forward(self, compound_graph, target):
#         fused_feature = self.DTencoder(compound_graph, target)        
#         predict = self.classifier(fused_feature)
#         return predict

