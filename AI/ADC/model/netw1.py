import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from model import gt_net_compound
from model import gin
from model import covae

from torch_geometric.nn.conv import GATConv,GATv2Conv,TransformerConv


if torch.cuda.is_available():
    device = torch.device('cuda')

class VAE(nn.Module):

    # def __init__(self, input_dim=641, h_dim=256, z_dim=128):
    def __init__(self, input_dim=5889, h_dim=1024, z_dim=128):
        # 调用父类方法初始化模块的state
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # 编码器 ： [b, input_dim] => [b, z_dim]
        self.fc1 = nn.Linear(input_dim, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, z_dim)  # mu
        self.fc3 = nn.Linear(h_dim, z_dim)  # log_var

        # 解码器 ： [b, z_dim] => [b, input_dim]
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)

    def forward(self, x):
        """
        向前传播部分, 在model_name(inputs)时自动调用
        :param x: the input of our training model [b, batch_size, 1, 28, 28]
        :return: the result of our training model
        """
        batch_size = x.shape[0]  # 每一批含有的样本的个数
        # flatten  [b, batch_size, 1, 28, 28] => [b, batch_size, 784]
        # tensor.view()方法可以调整tensor的形状，但必须保证调整前后元素总数一致。view不会修改自身的数据，
        # 返回的新tensor与原tensor共享内存，即更改一个，另一个也随之改变。
        x = x.view(batch_size, self.input_dim)  # 一行代表一个样本

        # encoder
        mu, log_var = self.encode(x)
        return mu, log_var
        # # reparameterization trick
        # sampled_z = self.reparameterization(mu, log_var)
        # # decoder
        # x_hat = self.decode(sampled_z)
        # # reshape
        # x_hat = x_hat.view(batch_size, 1, 28, 28)
        # return x_hat, mu, log_var

    def encode(self, x):
        """
        encoding part
        :param x: input image
        :return: mu and log_var
        """
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)

        return mu, log_var

    def reparameterization(self, mu, log_var):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu:
        :param log_var:
        :return: sampled z
        """
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps  # 这里的“*”是点乘的意思

    def decode(self, z):
        """
        Given a sampled z, decode it back to image
        :param z:
        :return:
        """
        h = F.relu(self.fc4(z))
        x_hat = torch.sigmoid(self.fc5(h))  # 图片数值取值为[0,1]，不宜用ReLU
        return x_hat
# class GateLinearUnit(nn.Module):
#     def __init__(self, input_size, num_filers, kernel_size, vocab_size=None, bias=True, batch_norm=True, activation=nn.Tanh()):
#         super(GateLinearUnit, self).__init__()
#         self.batch_norm = batch_norm
#         self.activation = activation
#         self.conv_layer1 = nn.Conv2d(1, num_filers, (kernel_size, input_size), bias=bias)
#         self.conv_layer2 = nn.Conv2d(1, num_filers, (kernel_size, input_size), bias=bias)
#         self.batch_norm = nn.BatchNorm2d(num_filers)
#         self.sigmoid = nn.Sigmoid()

#         nn.init.kaiming_uniform_(self.conv_layer1.weight)
#         nn.init.kaiming_uniform_(self.conv_layer2.weight)

#     def gate(self, inputs):
#         """门控机制"""
#         return self.sigmoid(inputs)

#     def forward(self, inputs):
#         inputs = inputs.unsqueeze(1)
#         output = self.conv_layer1(inputs)
#         gate_output = self.conv_layer2(inputs)
#         # Gate Operation
#         if self.activation is not None:
#             # GTU
#             output = self.activation(output) * self.gate(gate_output)
#         else:
#             # GLU
#             output = output * self.gate(gate_output)
#         if self.batch_norm:
#             output = self.batch_norm(output)
#             output = output.squeeze()
#             return output
#         else:
#             return output.squeeze()

class GateLinearUnit(nn.Module):
    def __init__(self, input_size, output_size, activation=nn.Tanh()):
        super(GateLinearUnit, self).__init__()
        # self.batch_norm = batch_norm
        self.activation = activation
        # self.conv_layer1 = nn.Conv2d(1, num_filers, (kernel_size, input_size), bias=bias)
        # self.conv_layer2 = nn.Conv2d(1, num_filers, (kernel_size, input_size), bias=bias)
        self.layer1=nn.Linear(in_features=input_size,out_features=output_size,bias=False)
        # self.batch_norm = nn.BatchNorm2d(num_filers)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_uniform_(self.layer1.weight)
        # nn.init.kaiming_uniform_(self.conv_layer2.weight)

    def gate(self, inputs):
        """门控机制"""
        return self.sigmoid(inputs)

    def forward(self, inputs):
        # inputs = inputs
        output = self.layer1(inputs)
        # gate_output = self.conv_layer2(inputs)
        # Gate Operation

        output = inputs * self.gate(output)

        return output


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
    def __init__(self, in_size=1280, hidden_size=128,ssize=64):
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
            nn.Linear(hidden_size, ssize),
            nn.Tanh(),
            nn.Linear(ssize, 1, bias=False)
        )
        self.project_linker = nn.Sequential(
            nn.Linear(hidden_size, ssize),
            nn.Tanh(),
            nn.Linear(ssize, 1, bias=False)
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

class UNADC_model(nn.Module):
    def __init__(self, device,compound_dim=128, protein_dim=128, gt_layers=3, gt_heads=4, out_dim=2):
        super(UNADC_model, self).__init__()
        self.compound_dim = compound_dim
        self.protein_dim = protein_dim
        self.n_layers = gt_layers
        self.n_heads = gt_heads

        self.compound_encoder = gt_net_compound.GraphTransformer(device, n_layers=gt_layers, node_dim=44, edge_dim=10, hidden_dim=compound_dim,
                                                        out_dim=compound_dim, n_heads=gt_heads, in_feat_dropout=0.1, dropout=0.1, pos_enc_dim=8)
  
        self.adc_encoder = gt_net_compound.GraphTransformer(device, n_layers=gt_layers, node_dim=128, edge_dim=5, hidden_dim=compound_dim,
                                                        out_dim=compound_dim, n_heads=gt_heads, in_feat_dropout=0.1, dropout=0.1, pos_enc_dim=8)

        self.pcalinear = nn.Sequential(
            nn.Linear(128, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 128),

        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 1024),
            # nn.Linear(128, 1024),#ablation 1/2
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim)
        )

        self.protein_linear = nn.Linear(1280, 128)

    def get_vn_feature(self, bg, feats):
        num_nodes = bg.batch_num_nodes()
        
        out=[]
        s=num_nodes[0]
        s=0
        for i in range(len(num_nodes)):
            s=s+num_nodes[i].item()
            out.append(feats[s-1])
            
        out = torch.stack(out)        
        return out

    def get_graph_feature(self, bg, feats):
        #对节点sum/mean作为图表示
        # 获取每个子图中节点的数量
        num_nodes = bg.batch_num_nodes()
        
        # 初始化列表用于存储每个子图的表示
        out = []
        start_idx = 0
        
        # 遍历每个子图，计算它的节点特征之和作为图的表示
        for n_nodes in num_nodes:
            # 取出当前子图的所有节点特征
            subgraph_feats = feats[start_idx:start_idx + n_nodes]
            # 对当前子图的所有节点特征进行求和，得到子图的表示
            # subgraph = subgraph_feats.sum(dim=0)
            subgraph, _ = torch.max(subgraph_feats, dim=0)
            # subgraph= torch.mean(subgraph_feats,dim=0)
            # 添加到输出列表中
            out.append(subgraph)
            # 更新下一个子图的起始索引
            start_idx += n_nodes
        
        # 将所有子图的表示堆叠成一个张量
        out = torch.stack(out)
        return out

    def forward(self, heavy, light, antigen, playload_graph, linker_graph, dar,adc,pca):

        playload_feat = self.compound_encoder(playload_graph)  # 3981*128
        linker_feat = self.compound_encoder(linker_graph)

        playload = self.get_vn_feature(playload_graph, playload_feat)
        linker = self.get_vn_feature(linker_graph, linker_feat)

        heavyo=heavy
        lighto=light
        antigeno=antigen
        heavy=self.protein_linear(heavy)
        light=self.protein_linear(light)
        antigen=self.protein_linear(antigen)
        dar = dar.reshape(-1, 1)
        nodenum=8
        for i in range(playload.size(0)):
            # 分配 playload 到特定节点
            adc.nodes[i * nodenum].data['atom'] = playload[i].unsqueeze(0)

            # 分配其他特征到特定节点
            adc.nodes[i * nodenum + 1].data['atom'] = linker[i].unsqueeze(0)
            adc.nodes[i * nodenum + 2].data['atom'] = light[i].unsqueeze(0)
            adc.nodes[i * nodenum + 3].data['atom'] = light[i].unsqueeze(0)
            adc.nodes[i * nodenum + 4].data['atom'] = heavy[i].unsqueeze(0)
            adc.nodes[i * nodenum + 5].data['atom'] = heavy[i].unsqueeze(0)
            adc.nodes[i * nodenum + 6].data['atom'] = antigen[i].unsqueeze(0)
            # adc.nodes[i * 8 + 7].data['node'] = #vn node

        # print(pca.shape)
        fused_vector = pca

        fuse_feat= fused_vector
        fuse_feat= self.pcalinear(fused_vector)
        fuse_feat+=fused_vector

        
        h = adc.ndata['atom'].float()
        # src, dst = adc.edges()
        # edge_index = torch.stack([src, dst], dim=0)
        e=adc.edata['bond']
        # adc_feat=self.adc_encoder(h,edge_index,e)#GAT/TransformerConv等等，得到一个图节点特征集合，节点特征为heat数量*128
        adc_feat=self.adc_encoder(adc)#GAT/TransformerConv等等，得到一个图节点特征集合，节点特征为heat数量*128
        adc_feat = self.get_vn_feature(adc,adc_feat)#vn

        all=torch.cat([fuse_feat,adc_feat], dim=1)

        x = self.classifier(all)
        return x

class UNADC_test(nn.Module):
    def __init__(self, device,compound_dim=128, protein_dim=128, gt_layers=3, gt_heads=4, out_dim=2):
        super(UNADC_test, self).__init__()
        self.compound_dim = compound_dim
        self.protein_dim = protein_dim
        self.n_layers = gt_layers
        self.n_heads = gt_heads

        self.compound_encoder = gt_net_compound.GraphTransformer(device, n_layers=gt_layers, node_dim=44, edge_dim=10, hidden_dim=compound_dim,
                                                        out_dim=compound_dim, n_heads=gt_heads, in_feat_dropout=0.1, dropout=0.1, pos_enc_dim=8)
  
        self.adc_encoder = gt_net_compound.GraphTransformer(device, n_layers=gt_layers, node_dim=128, edge_dim=5, hidden_dim=compound_dim,
                                                        out_dim=compound_dim, n_heads=gt_heads, in_feat_dropout=0.1, dropout=0.1, pos_enc_dim=8)

        self.pcalinear = nn.Sequential(
            nn.Linear(128, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 128),

        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 1024),
            # nn.Linear(128, 1024),#ablation 1/2
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim)
        )

        self.protein_linear = nn.Linear(1280, 128)

    def get_vn_feature(self, bg, feats):
        num_nodes = bg.batch_num_nodes()
        
        out=[]
        s=num_nodes[0]
        s=0
        for i in range(len(num_nodes)):
            s=s+num_nodes[i].item()
            out.append(feats[s-1])
            
        out = torch.stack(out)        
        return out

    def get_graph_feature(self, bg, feats):
        #对节点sum/mean作为图表示
        # 获取每个子图中节点的数量
        num_nodes = bg.batch_num_nodes()
        
        # 初始化列表用于存储每个子图的表示
        out = []
        start_idx = 0
        
        # 遍历每个子图，计算它的节点特征之和作为图的表示
        for n_nodes in num_nodes:
            # 取出当前子图的所有节点特征
            subgraph_feats = feats[start_idx:start_idx + n_nodes]
            # 对当前子图的所有节点特征进行求和，得到子图的表示
            # subgraph = subgraph_feats.sum(dim=0)
            subgraph, _ = torch.max(subgraph_feats, dim=0)
            # subgraph= torch.mean(subgraph_feats,dim=0)
            # 添加到输出列表中
            out.append(subgraph)
            # 更新下一个子图的起始索引
            start_idx += n_nodes
        
        # 将所有子图的表示堆叠成一个张量
        out = torch.stack(out)
        return out

    def forward(self, heavy, light, antigen, playload_graph, linker_graph, dar,adc,pca):

        playload_feat = self.compound_encoder(playload_graph)  # 3981*128
        linker_feat = self.compound_encoder(linker_graph)

        playload = self.get_vn_feature(playload_graph, playload_feat)
        linker = self.get_vn_feature(linker_graph, linker_feat)

        heavyo=heavy
        lighto=light
        antigeno=antigen
        heavy=self.protein_linear(heavy)
        light=self.protein_linear(light)
        antigen=self.protein_linear(antigen)
        dar = dar.reshape(-1, 1)
        nodenum=8
        # print(light.shape)
        # # for i in range(playload.size(0)):
        #     # 分配 playload 到特定节点
        # print(playload[0].shape)
        
        # print(adc.nodes[1].data['atom'].shape)
        # 分配其他特征到特定节点
        adc.nodes[0].data['atom'] = playload
        adc.nodes[1].data['atom'] = linker
        adc.nodes[2].data['atom'] = light.unsqueeze(0)
        adc.nodes[3].data['atom'] = light.unsqueeze(0)
        adc.nodes[4].data['atom'] = heavy.unsqueeze(0)
        adc.nodes[5].data['atom'] = heavy.unsqueeze(0)
        adc.nodes[6].data['atom'] = antigen.unsqueeze(0)
            # adc.nodes[i * 8 + 7].data['node'] = #vn node

        # print(pca.shape)
        fused_vector = pca

        fuse_feat= fused_vector
        fuse_feat= self.pcalinear(fused_vector)
        fuse_feat+=fused_vector

        
        h = adc.ndata['atom'].float()
        # src, dst = adc.edges()
        # edge_index = torch.stack([src, dst], dim=0)
        e=adc.edata['bond']
        # adc_feat=self.adc_encoder(h,edge_index,e)#GAT/TransformerConv等等，得到一个图节点特征集合，节点特征为heat数量*128
        adc_feat=self.adc_encoder(adc)#GAT/TransformerConv等等，得到一个图节点特征集合，节点特征为heat数量*128
        adc_feat = self.get_vn_feature(adc,adc_feat)#vn

        all=torch.cat([fuse_feat,adc_feat], dim=1)

        x = self.classifier(all)
        return x
    

class TGADC_ab(nn.Module):
    def __init__(self, compound_dim=128, protein_dim=128, gt_layers=3, gt_heads=4, out_dim=2):
        super(TGADC_ab, self).__init__()
        self.compound_dim = compound_dim
        self.protein_dim = protein_dim
        self.n_layers = gt_layers
        self.n_heads = gt_heads

        self.compound_encoder = gt_net_compound.GraphTransformer(device, n_layers=gt_layers, node_dim=44, edge_dim=10, hidden_dim=compound_dim,
                                                        out_dim=compound_dim, n_heads=gt_heads, in_feat_dropout=0.1, dropout=0.1, pos_enc_dim=8)
  
        # self.adc_encoder = GATv2Conv(in_channels=128, out_channels=16, heads=8, concat=True, negative_slope=0.2, dropout=0.1,edge_dim = 5)
        self.adc_encoder = gt_net_compound.GraphTransformer(device, n_layers=gt_layers, node_dim=128, edge_dim=5, hidden_dim=compound_dim,
                                                        out_dim=compound_dim, n_heads=gt_heads, in_feat_dropout=0.1, dropout=0.1, pos_enc_dim=8)
        self.protein_embed = nn.Embedding(26, self.protein_dim, padding_idx=0)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.protein_dim, out_channels=self.protein_dim, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.protein_dim, out_channels=self.protein_dim * 2, kernel_size=8),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.protein_dim * 2, out_channels=self.protein_dim, kernel_size=12),
            nn.ReLU(),
        )

        self.protein_max_pool = nn.MaxPool1d(979)
        self.compound_max_pool = nn.AdaptiveMaxPool1d(1)

        self.fuse_linear=nn.Linear(4097,128)
        self.pcalinear = nn.Sequential(
            nn.Linear(128, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 128),
            # nn.LeakyReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(128, 128)
        )
        self.glu = GateLinearUnit(input_size=256, output_size=256)

        self.classifier = nn.Sequential(
            nn.Linear(256, 1024),
            # nn.Linear(128, 1024),#ablation 1/2
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim)
        )
        self.fuse_encoder = VAE()

        self.protein_linear = nn.Linear(1280, 128)
        self.dtf = DTF()

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

    def get_vn_feature(self, bg, feats):
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

    def get_graph_feature(self, bg, feats):
        #对节点sum/mean作为图表示
        # 获取每个子图中节点的数量
        num_nodes = bg.batch_num_nodes()
        
        # 初始化列表用于存储每个子图的表示
        out = []
        start_idx = 0
        
        # 遍历每个子图，计算它的节点特征之和作为图的表示
        for n_nodes in num_nodes:
            # 取出当前子图的所有节点特征
            subgraph_feats = feats[start_idx:start_idx + n_nodes]
            # 对当前子图的所有节点特征进行求和，得到子图的表示
            subgraph_sum = subgraph_feats.sum(dim=0)
            # 添加到输出列表中
            out.append(subgraph_sum)
            # 更新下一个子图的起始索引
            start_idx += n_nodes
        
        # 将所有子图的表示堆叠成一个张量
        out = torch.stack(out)
        return out
    def get_embed(self, sequence):
        embed = self.protein_embed(sequence) # [128,1000,128]
        embed = embed.permute(0, 2, 1) 
        embed_feats = self.Protein_CNNs(embed) # [128,128,979]
        feature = self.protein_max_pool(embed_feats).squeeze(2)
        return feature
    def forward(self, heavy, light, antigen, playload_graph, linker_graph, dar,adc,playload,linker):
        # print(playload_graph.shape)
        # print(playload_graph)
        playload_feat = self.compound_encoder(playload_graph)  # 3981*128
        # print(playload_graph)
        # print(adc)
        linker_feat = self.compound_encoder(linker_graph)

        playload = self.get_vn_feature(playload_graph, playload_feat)
        linker = self.get_vn_feature(linker_graph, linker_feat)
        heavyo=heavy
        lighto=light
        antigeno=antigen
        heavy=self.protein_linear(heavy)
        light=self.protein_linear(light)
        antigen=self.protein_linear(antigen)
        # heavy = self.get_embed(heavy)
        # light = self.get_embed(light)
        # antigen = self.get_embed(antigen)
        dar = dar.reshape(-1, 1)


        nodenum=8
        for i in range(playload.size(0)):
            # 分配 playload 到特定节点
            adc.nodes[i * nodenum].data['atom'] = playload[i].unsqueeze(0)

            # 分配其他特征到特定节点
            adc.nodes[i * nodenum + 1].data['atom'] = linker[i].unsqueeze(0)
            adc.nodes[i * nodenum + 2].data['atom'] = light[i].unsqueeze(0)
            adc.nodes[i * nodenum + 3].data['atom'] = light[i].unsqueeze(0)
            adc.nodes[i * nodenum + 4].data['atom'] = heavy[i].unsqueeze(0)
            adc.nodes[i * nodenum + 5].data['atom'] = heavy[i].unsqueeze(0)
            adc.nodes[i * nodenum + 6].data['atom'] = antigen[i].unsqueeze(0)
            # adc.nodes[i * 8 + 7].data['node'] = #vn node

        # print(dar.shape,playload.shape)
        fused_vector = torch.cat((heavyo, lighto ,antigeno, playload, linker,dar),dim=1)

        fused_vector= self.fuse_linear(fused_vector)
        fuse_feat= self.pcalinear(fused_vector)
        fuse_feat+=fused_vector

        
        h = adc.ndata['atom'].float()
        # adc_feat=self.adc_encoder(adc,h)#GIN,直接得出图表示

        src, dst = adc.edges()
        # print(src)
        # print(dst)
        edge_index = torch.stack([src, dst], dim=0)
        # print(edge_index.shape)
        e=adc.edata['bond']
        # adc_feat=self.adc_encoder(h,edge_index,e)#GAT/TransformerConv等等，得到一个图节点特征集合，节点特征为heat数量*128
        adc_feat=self.adc_encoder(adc)#GAT/TransformerConv等等，得到一个图节点特征集合，节点特征为heat数量*128
        # # print(adc_feat.shape)#32 128
        # adc_feat = self.get_graph_feature(adc,adc_feat)#sum/mean
        adc_feat = self.get_vn_feature(adc,adc_feat)#vn


        # all=fuse_feat+adc_feat
        # all=self.dtf(fuse_feat, adc_feat)
        # print(fuse_feat)
        all=torch.cat([fuse_feat,adc_feat], dim=1)

        # all=adc_feat
        # all=fuse_feat
        # all=self.glu(all)
        x = self.classifier(all)
        return x
    


# 蛋白质 vae