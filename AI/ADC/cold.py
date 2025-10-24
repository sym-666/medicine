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
from ADCDataset2 import ADCDataset,ADCDatasetmorgan,ADCDatasetmorganPCA,ADCPCAn
from model.netw1 import TGADC,TGADC_m
from loss import DiceLoss,BCEFocalLoss
from sklearn.utils import resample

# from model.covae import net
from loss import FocalLoss
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

from sklearn.model_selection import KFold
from glob import glob


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

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    sample_num = 0

    # 初始化计数器

    total_preds = []
    total_labels = []
    for batch_idx, data in enumerate(train_loader):
        heavy, light, antigen, playload_graph, linker_graph, dar, label ,adc_graph,pca= data[:]
        heavy = torch.tensor(np.array(heavy)).to(device)
        light = torch.tensor(np.array(light)).to(device)
        antigen = torch.tensor(np.array(antigen)).to(device)
        # playload_graph = torch.FloatTensor(np.array(playload_graph)).to(device)
        # linker_graph = torch.FloatTensor(np.array(linker_graph)).to(device)
        playload_graph = playload_graph.to(device)
        linker_graph = linker_graph.to(device)
        dar = dar.to(device)
        label = label.to(device)
        label = label.unsqueeze(1)#BCE

        pca = pca.to(device)
        adc_graph=adc_graph.to(device)
        # union_graph=union_graph.to(device)
        # print(adc_graph.x, adc_graph.edge_index, adc_graph.edge_attr)
        output = model(heavy=heavy, light=light, antigen=antigen,
                       playload_graph=playload_graph, linker_graph=linker_graph, dar=dar,adc=adc_graph,pca=pca)
        # return
        
        batch_loss = criterion(output, label)
        # c1=criterion[0],c2=criterion[1]
        # for c1,c2 in criterion:
        # batch_loss=0
        # batch_loss=criterion[0](output,label)*0.65+criterion[1](output,label)*0.35
        # batch_loss+=criterion[1](output,label)

        total_loss += batch_loss.item() * label.size(0)
        sample_num += label.size(0)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # 计算预测概率
        # preds_score = F.softmax(output, 1).to('cpu').data.numpy()
        # preds_score = preds_score[:, 1]
        preds_score = torch.sigmoid(output).to('cpu').data.numpy() #BCE
        # print(preds_score)
        preds_score = preds_score.flatten()  # 将输出从 [64, 1] 转换为 [64]

        total_preds.extend(preds_score)

        total_labels.extend(label.cpu().numpy())

    train_loss = total_loss / sample_num

    # 输出结果

    tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV=score(total_labels, total_preds)
    print('traindataset {},{},{},{}'.format(tp, tn, fn, fp))
    return train_loss,mcc
def test(model, device, test_loader):
    model.eval()
    total_preds = []
    total_labels = []
    sample_num = 0
    total_loss=0
    with torch.no_grad():
        for data in test_loader:
            heavy, light, antigen, playload_graph, linker_graph, dar, label ,adc_graph,pca= data[:]
            heavy = torch.tensor(np.array(heavy)).to(device)
            light = torch.tensor(np.array(light)).to(device)
            antigen = torch.tensor(np.array(antigen)).to(device)
            # playload_graph = torch.FloatTensor(np.array(playload_graph)).to(device)
            # linker_graph = torch.FloatTensor(np.array(linker_graph)).to(device)
            playload_graph = playload_graph.to(device)
            linker_graph = linker_graph.to(device)
            pca = pca.to(device)
            dar = dar.to(device)
            label = label.to(device)
            label = label.unsqueeze(1)#BCE

            adc_graph=adc_graph.to(device)
            # union_graph=union_graph.to(device)
            # target = torch.tensor(np.array(target))
            # target = target.unsqueeze(1)
            # target = target.to(device).float()
            output = model(heavy=heavy, light=light, antigen=antigen,
                       playload_graph=playload_graph, linker_graph=linker_graph, dar=dar,adc=adc_graph,pca=pca)
            batch_loss = criterion(output, label)
            # c1=criterion[0],c2=criterion[1]
            # # for c1,c2 in criterion:
            # batch_loss=c1(output,label)
            # batch_loss+=c2(output,label)
            total_loss += batch_loss.item() * output.size(0)
            sample_num += output.size(0)
            # preds_score = F.softmax(output, 1).to('cpu').data.numpy()
            # preds_score = preds_score[:, 1]
            # print(output)
            preds_score = torch.sigmoid(output).to('cpu').data.numpy() #BCE
            preds_score = preds_score.flatten()  # 将输出从 [64, 1] 转换为 [64]

            
            total_preds.extend(preds_score)

            total_labels.extend(label.cpu().numpy())

    # total_preds = torch.sigmoid(total_preds).numpy()
    # total_preds = np.array(total_preds)
    # total_labels = np.array(total_labels)
    # total_preds = torch.sigmoid(torch.tensor(total_preds)).numpy()
    tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV = score(total_labels, total_preds)
    test_loss = total_loss / sample_num
    print("test {} {} {} {}".format(tp, tn, fn, fp))
    
    return  test_loss,tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV



if __name__ == '__main__':
    """select seed"""
    SEED = 10
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True

    # file_path = 'dataset/processed/'
    # protein_embed = 'dataset/' + dataset + '/proteinbert_embedding_seq.h5'
    log_file = 'logs/cold/' + str(time.strftime("%m%d-%H%M", time.localtime())) + '.txt'
    results_file = 'results/cold/' + str(time.strftime("%m%d-%H%M", time.localtime())) + '.txt'
    os.makedirs('results/cold/', exist_ok=True)
    os.makedirs('logs/cold/', exist_ok=True)

    # batch = 32
    batch = 64
    lr = 0.00008
    # batch = 32
    # lr = 0.00002
    weight_ce = torch.FloatTensor([1.9, 1]).to(device)

    notes = '特征融合,无weight，200epoch'
    with open(log_file, 'a') as f:
            f.write(notes + '\n')
            f.write('cold linker:' +'\n')
            f.write('batch_size:' + str(batch) + '\n')
            f.write('lr:' + str(lr) + '\n')
            f.write('seed:' + str(SEED) + '\n')

    k_fold = 5
    # num_iter = 3000
    
    Patience = 20
    se_list = []
    sp_list = []
    mcc_list = []
    acc_list = []
    auc_list = []
    F1_list = []
    BA_list = []
    prauc_list = []
    PPV_list = []
    NPV_list = []

    # df = pd.read_csv('dataset/dataset.csv')

    # kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    pca_processor = ADCPCAn(n_components=128)

    # dar_data = df.iloc[:, -2].values.reshape(-1, 1)
    # scaler = StandardScaler()
    # dar_data_standardized = scaler.fit_transform(dar_data)
    # dar_data_normalized = normalize(dar_data_standardized, axis=0).flatten()

    # # 更新 DataFrame 中的特定列
    # df.iloc[:, -2] = dar_data_normalized

    # 打印检查数据集长度
    # print('df:', len(df))
    # i_fold=0
    folder_path = 'dataset/cold start/cold_playload'  # 将此路径替换为实际文件夹路径
    # folder_path = 'dataset/cold start/cold_antigen'  # 将此路径替换为实际文件夹路径
    # folder_path = 'dataset/cold start/cold_antibody'  # 将此路径替换为实际文件夹路径
    # folder_path = 'dataset/cold start/cold_linker'  # 将此路径替换为实际文件夹路径
    # 获取所有fold_*.csv文件
    file_paths = sorted(glob(os.path.join(folder_path, 'fold*.csv')))
    num_folds = len(file_paths)

    for i_fold in range(num_folds):
        # 构造验证集文件名
        val_file = file_paths[i_fold]
        
        # 读取当前折叠的数据作为验证集
        test_fold = pd.read_csv(val_file)
        
        # 构造训练集，合并除了当前折叠外的所有数据
        train_files = [file for j, file in enumerate(file_paths) if j != i_fold]
        train_fold = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)

        # 分离训练集和验证集
        # train_fold = df.iloc[train_index]
        # test_fold = df.iloc[val_index]
        # train_fold, val_fold = train_test_split(train_fold, test_size=0.09)
        val_fold = test_fold
        # 计算每个类别的数量
        if i_fold>-1:
            label_counts = train_fold['label'].value_counts()

            # 判断哪个类别是少数类
            if label_counts[0] < label_counts[1]:
                minority = train_fold[train_fold['label'] == 0]
                majority = train_fold[train_fold['label'] == 1]
            else:
                minority = train_fold[train_fold['label'] == 1]
                majority = train_fold[train_fold['label'] == 0]

            # 上采样少数类
            minority_upsampled = resample(minority,
                                        replace=True,     # 允许重复采样
                                        n_samples=len(majority),  # 匹配多数类数量
                                        random_state=42)  # 随机种子

            # 合并多数类和上采样后的少数类
            train_fold_upsampled = pd.concat([majority, minority_upsampled], ignore_index=True)

            
            train_fold_balanced = pd.concat([majority, minority_upsampled])
            train_fold_balanced = train_fold_balanced.sample(frac=1, random_state=SEED).reset_index(drop=True)
            train_fold_processed = pca_processor.fit_transform(train_fold_balanced)

            test_fold_processed = pca_processor.transform(test_fold)
            val_fold_processed = pca_processor.transform(val_fold)

            train_set = ADCDatasetmorganPCA(train_fold_processed)
            test_set = ADCDatasetmorganPCA(test_fold_processed)
            val_set = ADCDatasetmorganPCA(val_fold_processed)

            train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, collate_fn=train_set.collate, drop_last=False)
            test_loader = DataLoader(test_set, batch_size=batch, shuffle=False, collate_fn=test_set.collate, drop_last=False)
            val_loader = DataLoader(val_set, batch_size=batch, shuffle=False, collate_fn=test_set.collate, drop_last=False)

            model = TGADC_m(compound_dim=128, protein_dim=128, gt_layers=3, gt_heads=4, out_dim=1)
            model.to(device)
    
            best_ci =0
            best_auc = 0
            best_rm2 = 0
            best_acc = 0
            best_epoch = -1
            patience = 0
            epochs = 300
            
            metric_dict = {'se':0, 'sp':0, 'mcc':0, 'acc':0, "auc":0, 'F1':0, 'BA':0, 'prauc':0, 'PPV':0, 'NPV':0}

            optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=0.01)
            # optimizer = optim.Adam(model.parameters(), lr=lr)
            # optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, lr_decay=0.01, weight_decay=0.01, initial_accumulator_value=0.1, eps=1e-07)
            
            # optimizer = torch.optim.SGD(
            #     model.parameters(), 
            #     lr=lr, 
            #     momentum=0.85, 
            #     weight_decay=1e-3, 
            #     nesterov=True
            # )

            # optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, lr_decay=0.1, weight_decay=0.1, initial_accumulator_value=0.1, eps=1e-07)
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.9, patience=30, verbose=True, min_lr=1e-5)
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.8, patience=5, verbose=True, min_lr=1e-5)
            # scheduler=optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[5,10,30,50], gamma=0.6, last_epoch=-1)
            # criterion = nn.CrossEntropyLoss(weight=weight_ce)
            # criterion = nn.BCEWithLogitsLoss()
            # criterion1 = DiceLoss(smooth=1e-6)
            criterion = BCEFocalLoss(gamma=3, alpha=0.25)
            # criterion = BCEFocalLoss(gamma=2, alpha=0.35)
            # criterion=[criterion1,criterion2]
            # criterion = nn.CrossEntropyLoss()
            # criterion = FocalLoss( num_class=2, alpha=[0.6,0.4], gamma=2, balance_index=-1, smooth=None, size_average=True )
            print('Start raining.')
            for epoch in range(epochs):

                train_loss,train_mcc=train(model, device, train_loader, optimizer,criterion)
                valloss,vtp, vtn, vfn, vfp, vse, vsp, vmcc, vacc, vauc_roc_score, vF1, vBA, vprauc, vPPV, vNPV = test(model, device, val_loader)

                testloss,tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV = test(model, device, test_loader)
                
                with open(log_file, 'a') as f:
                    f.write(str(time.strftime("%m-%d %H:%M:%S", time.localtime())) + ' epoch:' + str(epoch+1) + ' test_loss' + str(testloss) + ' se:' + str(round(se,4)) +' '+ 'sp:' + str(
                        round(sp,4)) + ' ' + 'mcc:' + str(round(mcc,4)) +' acc:'+str(round(acc,4)) +' auc:'+str(round(auc_roc_score,4)) + ' F1:'+str(round(F1,4)) + ' BA:'+str(round(BA,4))+ ' prauc:'+str(round(prauc,4))+
                        ' PPV:'+str(round(PPV,4))+ ' NPV:'+str(round(NPV,4))+ '\n')
                
                # scheduler.step()
                #上传时改成监控VAL
                
                if vacc > best_acc:
                    # if mse_test < 0.600:
                    #     torch.save(model.state_dict(), file_model + 'Epoch:' + str(epoch + 1) + '.pt')
                    #     print("model has been saved")
                    best_epoch = epoch + 1
                    # best_auc = auc_roc_score
                    best_acc=vacc
                    # metric_dict['se'] = se
                    # metric_dict['sp'] = sp
                    # metric_dict['mcc'] = mcc
                    # metric_dict['acc'] = acc
                    # metric_dict['auc'] = auc_roc_score
                    # metric_dict['F1'] = F1
                    # metric_dict['BA'] = BA
                    # metric_dict['prauc'] = prauc
                    # metric_dict['PPV'] = PPV
                    # metric_dict['NPV'] = NPV
                    patience = 0
                    with open(log_file, 'a') as f:
                        f.write('Training ACC improved at epoch ' + str(best_epoch) + ';\tbest_mcc:' + str(round(best_acc,4)) + '\n')
                    print('fold ',i_fold,' Acc improved at epoch ', best_epoch, ';\tbest_mcc:', best_acc)
                else:
                    patience += 1
                # epoch += 1

                if patience == Patience:
                #     # break_flag = True
                    #连续Patience后，auc_roc_score还没有提升，break
                    metric_dict['se'] = se
                    metric_dict['sp'] = sp
                    metric_dict['mcc'] = mcc
                    metric_dict['acc'] = acc
                    metric_dict['auc'] = auc_roc_score
                    metric_dict['F1'] = F1
                    metric_dict['BA'] = BA
                    metric_dict['prauc'] = prauc
                    metric_dict['PPV'] = PPV
                    metric_dict['NPV'] = NPV
                    break
                            
            se_list.append(metric_dict['se'])
            sp_list.append(metric_dict['sp'])
            mcc_list.append(metric_dict['mcc'])
            acc_list.append(metric_dict['acc'])
            auc_list.append(metric_dict['auc'])
            F1_list.append(metric_dict['F1'])
            BA_list.append(metric_dict['BA'])
            prauc_list.append(metric_dict['prauc'])
            PPV_list.append(metric_dict['PPV'])
            NPV_list.append(metric_dict['NPV'])

            with open(log_file, 'a') as f:
                f.write('第' + str(i_fold ) + '折---' + 'se:' + str(round(metric_dict['se'],4)) +' '+ 'sp:' + str(
                            round(metric_dict['sp'],4)) + ' ' + 'mcc:' + str(round(metric_dict['mcc'],4)) +' acc:'+str(round(metric_dict['acc'],4))+
                            ' auc:'+str(round(metric_dict['auc'],4)) + ' F1:'+str(round(metric_dict['F1'],4)) +' BA:'+str(round(metric_dict['BA'],4)) +
                            ' prauc:'+str(round(metric_dict['prauc'],4)) +' PPV:'+str(round(metric_dict['PPV'],4)) +' NPV:'+str(round(metric_dict['NPV'],4)) +'\n')
            with open(results_file, 'a') as f:
                f.write('第' + str(i_fold ) + '折---' + 'se:' + str(round(metric_dict['se'],4)) +' '+ 'sp:' + str(
                            round(metric_dict['sp'],4)) + ' ' + 'mcc:' + str(round(metric_dict['mcc'],4)) +' acc:'+str(round(metric_dict['acc'],4))+
                            ' auc:'+str(round(metric_dict['auc'],4)) + ' F1:'+str(round(metric_dict['F1'],4)) +' BA:'+str(round(metric_dict['BA'],4)) +
                            ' prauc:'+str(round(metric_dict['prauc'],4)) +' PPV:'+str(round(metric_dict['PPV'],4)) +' NPV:'+str(round(metric_dict['NPV'],4)) +'\n')

    se_mean, se_var = np.mean(se_list), np.sqrt(np.var(se_list))
    sp_mean, sp_var = np.mean(sp_list), np.sqrt(np.var(sp_list))
    mcc_mean, mcc_var = np.mean(mcc_list), np.sqrt(np.var(mcc_list))
    acc_mean, acc_var = np.mean(acc_list), np.sqrt(np.var(acc_list))
    auc_mean, auc_var = np.mean(auc_list), np.sqrt(np.var(auc_list))
    F1_mean, F1_var = np.mean(F1_list), np.sqrt(np.var(F1_list))
    BA_mean, BA_var = np.mean(BA_list), np.sqrt(np.var(BA_list))
    prauc_mean, prauc_var = np.mean(prauc_list), np.sqrt(np.var(prauc_list))
    PPV_mean, PPV_var = np.mean(PPV_list), np.sqrt(np.var(PPV_list))
    NPV_mean, NPV_var = np.mean(NPV_list), np.sqrt(np.var(NPV_list))
    
    with open(results_file, 'a') as f:

        f.write(f'mean results: se:{se_mean:.4f}({se_var:.4f}) sp:{sp_mean:.4f}({sp_var:.4f}) mcc:{mcc_mean:.4f}({mcc_var:.4f}) acc:{acc_mean:.4f}({acc_var:.4f}) auc:{auc_mean:.4f}({auc_var:.4f}) F1:{F1_mean:.4f}({F1_var:.4f}) BA:{BA_mean:.4f}({BA_var:.4f}) prauc:{prauc_mean:.4f}({prauc_var:.4f}) PPV:{PPV_mean:.4f}({PPV_var:.4f}) NPV:{NPV_mean:.4f}({NPV_var:.4f})')
        f.write(f'''seed36 \n损失crossentropy
                \n模型1：节点特征128d：小分子morgan+linear。大分子采用linear，边特征5d采用标准化+归一化dar+onehot、模型采用GT修改了layer和head,图表示采用VN  
\n模型2：没有用dar，fused_vector = torch.cat([heavyo, lighto ,antigeno, playload, linker])=5889维,采用未污染的PCA,给pca之后添加了线性层 lk ,2层 128-1024-128,残差
\n特征融合：,cat dar,leakrelu,分类器256-1024-128-128
                标准化+归一化
                改了学习率调整策略和factor=0.8, patience=5 lr=6e-5
                没有glu，patience：MCC
                batch=32
                Adagrad FOCAL 0.01
                linker
                ''')
            
#  \n模型2：没有用dar，fused_vector = torch.cat([heavyo, lighto ,antigeno, playload, linker])=5889维,模型采用PCA,给pca之后添加了线性层 lk ,2层
# \n模型1：节点特征128d：union+小分子都是vngt，大分子采用linear，边特征5d采用只归一化dar+onehot、模型采用GAT2,输出由128维改成16维，图表示采用VN
# \n模型2：没有用dar，fused_vector = torch.cat([heavyo, lighto ,antigeno,union], dim=1)=3968维,模型采用Linear
# \n特征融合：dtf,cat dar''')
        # \n无模型2''')
        
# 模型1：节点特征128d：union+小分子都是vngt，大分子采用linear，边特征5d采用只归一化dar+onehot、模型采用GAT2,输出由128维改成16维，图表示采用VN
        #模型2：没有用dar，fused_vector = torch.cat([heavyo, lighto ,antigeno, playload, linker])=4096维,模型采用VAE
        #直接cat,包括dar