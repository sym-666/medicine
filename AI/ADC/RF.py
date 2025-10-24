import math
import os
import time

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from ML_Dataset import ML_Dataset

import torch
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, auc, confusion_matrix, classification_report, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def score(y_test, y_pred):
    auc_roc_score = roc_auc_score(y_test, y_pred)#, multi_class='ovo')
    prec, recall, _ = precision_recall_curve(y_test, y_pred)
    prauc = auc(recall, prec)
    y_pred_print = [round(y, 0) for y in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_print).ravel()
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = (tp + tn) / (tp + fn + tn + fp)
    mcc = (tp * tn - fn * fp) / np.sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))
    P = tp / (tp + fp)
    F1 = (P * se * 2) / (P + se)
    BA = (se + sp) / 2
    PPV = tp / (tp + fp)
    NPV = tn / (fn + tn)
    return se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV

if __name__ == '__main__':
    # SEED = 29
    SEED = 10
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    results_file = 'ablation/results/' + str(time.strftime("%m%d-%H%M", time.localtime())) + '.txt'
    os.makedirs('ablation/results/', exist_ok=True)

    TVdataset = pd.read_csv('dataset/dataset.csv')
    TVdataset = TVdataset.values

    X = ML_Dataset(TVdataset)  # 蛋白质和药物的拼接特征
    y = TVdataset[:, -1]         # 亲和力标签（例如，0或1）

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

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for method in ['RF', 'LR', 'SVM', 'XGB']:
        # , 'LR', 'SVM', 'XGB'
        i_fold = -1     
        with open(results_file, 'a') as f:
            f.write('*' * 25 + str(method) + '*' * 25 + '\n')
        for train_index, val_index in kf.split(X):
            i_fold += 1
            X_train = X[train_index]
            X_test = X[val_index]
            y_train = y[train_index].astype('int')
            y_test = y[val_index].astype('int')

            print(f'Train DataFrame has {X_train.shape[0]} rows.')
            print(f'Test DataFrame has {X_test.shape[0]} rows.')

            if method == 'RF':
                rf_model = RandomForestClassifier(n_estimators=100, random_state=SEED)
                rf_model.fit(X_train, y_train)
                y_pred = rf_model.predict_proba(X_test)[:, 1]
            elif method == 'LR':
                lr_model = LogisticRegression(random_state=SEED)
                lr_model.fit(X_train, y_train)
                y_pred = lr_model.predict_proba(X_test)[:, 1]
            elif method == 'SVM':
                svm_model = SVC(random_state=SEED, probability=True)
                svm_model.fit(X_train, y_train)
                y_pred = svm_model.predict_proba(X_test)[:, 1]
            elif method == 'XGB':
                xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=SEED)
                xgb_model.fit(X_train, y_train)
                y_pred = xgb_model.predict_proba(X_test)[:, 1]

            # print(y_test)
            # print("y_test shape:", y_test.shape)
            # print("y_pred shape before predict:", y_pred.shape if 'y_pred' in locals() else 'not defined yet')
            # print("y_pred shape after predict:", y_pred.shape)

            se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV = score(y_test, y_pred)
            se_list.append(se)
            sp_list.append(sp)
            mcc_list.append(mcc)
            acc_list.append(acc)
            auc_list.append(auc_roc_score)
            F1_list.append(F1)
            BA_list.append(BA)
            prauc_list.append(prauc)
            PPV_list.append(PPV)
            NPV_list.append(NPV)
            with open(results_file, 'a') as f:
                f.write('第' + str(i_fold + 1) + '折---' + 'se:' + str(round(se,4)) +' '+ 'sp:' + str(
                            round(sp,4)) + ' ' + 'mcc:' + str(round(mcc,4)) +' acc:'+str(round(acc,4))+
                            ' auc:'+str(round(auc_roc_score,4)) + ' F1:'+str(round(F1,4)) +' BA:'+str(round(BA,4)) +
                            ' prauc:'+str(round(prauc,4)) +' PPV:'+str(round(PPV,4)) +' NPV:'+str(round(NPV,4)) +'\n')
                
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
            f.write(f'{SEED}\n')
            # f.write('\n')
            f.write(f'mean results: se:{se_mean:.4f}({se_var:.4f}) sp:{sp_mean:.4f}({sp_var:.4f}) mcc:{mcc_mean:.4f}({mcc_var:.4f}) acc:{acc_mean:.4f}({acc_var:.4f}) auc:{auc_mean:.4f}({auc_var:.4f}) F1:{F1_mean:.4f}({F1_var:.4f}) BA:{BA_mean:.4f}({BA_var:.4f}) prauc:{prauc_mean:.4f}({prauc_var:.4f}) PPV:{PPV_mean:.4f}({PPV_var:.4f}) NPV:{NPV_mean:.4f}({NPV_var:.4f})\n')
            f.write('\n')