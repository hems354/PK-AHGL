# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import os
import time
import copy
import torch
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
import scipy.io as scio
import random
import itertools
import torch
from tqdm import tqdm
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import pprint as pp
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score,confusion_matrix
from tqdm import tqdm
import torch
import dhg
from torch.utils.data import TensorDataset, DataLoader, random_split
from nilearn import datasets, image, plotting
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import index_img, math_img
from nilearn.plotting import find_parcellation_cut_coords
from scipy.spatial.distance import cdist

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
work_dir = '/data/wjc/Mamba_hyper_brainnet'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)
models_dir = "saved_models"
os.makedirs(models_dir, exist_ok=True)


import numpy as np
def Eu_dis(x):
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def importance_loss_pairwise_adaptive(H: torch.Tensor, edge_weight: torch.Tensor, important_roi: torch.Tensor, alpha=1.0):

    B, N, E = H.shape
    device = H.device

    imp_roi = important_roi.view(1, N, 1).to(device)  # (1, N, 1)
    important_counts = torch.sum(H * imp_roi, dim=1)  # (B, E)
    total_counts = torch.sum(H, dim=1)                # (B, E)
    imp_ratio = important_counts / (total_counts + 1e-8)  # (B, E)
    edge_weights = torch.sum(edge_weight * H, dim=1)  # (B, E)
    r_i = imp_ratio.unsqueeze(2)  # (B, E, 1)
    r_j = imp_ratio.unsqueeze(1)  # (B, 1, E)
    w_i = edge_weights.unsqueeze(2)
    w_j = edge_weights.unsqueeze(1)

    diff_r = r_i - r_j  # (B, E, E)
    diff_w = w_i - w_j  # (B, E, E)

    mask = (diff_r > 0).float()  # (B, E, E)

    adaptive_margin = alpha * diff_r * mask  # (B, E, E)

    loss_matrix = F.relu(adaptive_margin - diff_w)  # (B, E, E)

    valid_pairs = mask.sum() + 1e-8
    loss = loss_matrix.sum() / valid_pairs

    return loss



def laplacian_constraint_batch(w, delta_batch):

    eps = 1e-8
    w_diag = torch.diagonal(w)

    D_sigma = torch.sum(delta_batch, dim=2) + eps

    w_norm = w_diag.unsqueeze(0).expand(delta_batch.size(0), -1) / torch.sqrt(D_sigma)

    w_i = w_norm.unsqueeze(2)  
    w_j = w_norm.unsqueeze(1)  
    diff = w_i - w_j
    diff_squared = diff * diff
    costs = 0.5 * torch.sum(delta_batch * diff_squared, dim=[1, 2])
    return costs.mean()

def build_prior_matrix(brain_importance, H, x=0.8):
    if len(brain_importance.shape) == 1:
        brain_importance = brain_importance.reshape(-1, 1)

    brain_weights = np.where(brain_importance == 1, x, 1 - x)

    H_score = np.zeros_like(H)  # [200,200]

    mask = (H != 0)  # [200,200]

    weights_expanded = np.broadcast_to(brain_weights, (200, 200))  # [200,200]

    H_score = np.where(mask, weights_expanded, 0)
    delta = np.matmul(H_score.T, H_score)
    max_value = np.max(delta)
    if max_value > 0:
        delta = delta / max_value
    delta = delta + np.eye(200) * 1e-8

    return delta
def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):

    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H
def build_hg_for_intra_subj(feature):
    subj_num, roi_num, fea_num = feature.shape
    hg_list = []
    for idx in range(subj_num):
        temp_feature = torch.tensor(feature[idx], dtype=torch.float32)
        hg_list.append(dhg.Hypergraph.from_feature_kNN(temp_feature, k=3))  

    return hg_list


def batch_incidence2G(H, symmetric=False):

    batch_size = H.shape[0]
    if symmetric:
        Dv = H.sum(dim=2, keepdim=True)  # 每个节点的连接边数
        De = H.sum(dim=1, keepdim=True)  # 每条边的连接节点数

        Dv_12 = Dv.pow(-0.5)
        Dv_12[torch.isinf(Dv_12)] = 0
        De_1 = De.pow(-1)
        De_1[torch.isinf(De_1)] = 0

        x1 = Dv_12 * H  # D_v^(-1/2) H
        x2 = (De_1 * H * Dv_12).transpose(-2, -1)  # (D_e^(-1) H^T D_v^(-1/2))^T
    else:
        norm_r = 1 / H.sum(dim=2, keepdim=True)
        norm_r[torch.isinf(norm_r)] = 0
        norm_c = 1 / H.sum(dim=1, keepdim=True)
        norm_c[torch.isinf(norm_c)] = 0

        x1 = norm_r * H
        x2 = (norm_c * H).transpose(-2, -1)

    return x1, x2

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):

        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x



class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5,threshold = 0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.threshold = threshold
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)
        self.fc = nn.Linear(in_ch*2, 2)
        self.edge_weight = Parameter(torch.ones(1, in_ch, in_ch))

    def forward(self, x,H):

        batch_size, num_nodes, num_edges = H.shape
        x1, x2 = batch_incidence2G(H)
        G = torch.matmul(x1, self.edge_weight * x2)

        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x,self.edge_weight


cc200_img = nib.load(os.path.join(work_dir, f"data/ABIDE/abide_rois/CC200.nii.gz"))
cc200_data = cc200_img.get_fdata()
labels = np.unique(cc200_data)
labels = labels[labels != 0]  
affine = cc200_img.affine
regions_info = []

for label in labels:
    roi_img = math_img("img == {}".format(label), img=cc200_img)
    roi_data = roi_img.get_fdata()
    voxel_indices = np.array(np.nonzero(roi_data)).T  # Shape: (n_voxels, 3)
    voxel_coords = nib.affines.apply_affine(affine, voxel_indices)
    center_coord = voxel_coords.mean(axis=0)
    region_info = {
        'Region': int(label),
        'X': center_coord[0],
        'Y': center_coord[1],
        'Z': center_coord[2]
    }
    regions_info.append(region_info)

cc200_coords = pd.DataFrame(regions_info)

corticostriatal_regions = [
    {'Region': 'Caudate_L', 'Network': 'Corticostriatal', 'X': -12, 'Y': 10, 'Z': 9},
    {'Region': 'Caudate_R', 'Network': 'Corticostriatal', 'X': 12, 'Y': 10, 'Z': 9},
    {'Region': 'Putamen_L', 'Network': 'Corticostriatal', 'X': -28, 'Y': 1, 'Z': 3},
    {'Region': 'Putamen_R', 'Network': 'Corticostriatal', 'X': 28, 'Y': 1, 'Z': 3},
    {'Region': 'Pallidum_L', 'Network': 'Corticostriatal', 'X': -20, 'Y': -5, 'Z': 0},
    {'Region': 'Pallidum_R', 'Network': 'Corticostriatal', 'X': 20, 'Y': -5, 'Z': 0},
]

frontoparietal_regions = [
    {'Region': 'Frontal_Mid_L', 'Network': 'Frontoparietal', 'X': -42, 'Y': 36, 'Z': 20},
    {'Region': 'Frontal_Mid_R', 'Network': 'Frontoparietal', 'X': 42, 'Y': 36, 'Z': 20},
    {'Region': 'Parietal_Sup_L', 'Network': 'Frontoparietal', 'X': -26, 'Y': -64, 'Z': 54},
    {'Region': 'Parietal_Sup_R', 'Network': 'Frontoparietal', 'X': 26, 'Y': -64, 'Z': 54},
    {'Region': 'Inferior_Parietal_L', 'Network': 'Frontoparietal', 'X': -42, 'Y': -52, 'Z': 46},
    {'Region': 'Inferior_Parietal_R', 'Network': 'Frontoparietal', 'X': 42, 'Y': -52, 'Z': 46},
]

visual_regions = [
    {'Region': 'Calcarine_L', 'Network': 'Visual', 'X': -14, 'Y': -92, 'Z': -2},
    {'Region': 'Calcarine_R', 'Network': 'Visual', 'X': 14, 'Y': -92, 'Z': -2},
    {'Region': 'Occipital_Sup_L', 'Network': 'Visual', 'X': -24, 'Y': -76, 'Z': 38},
    {'Region': 'Occipital_Sup_R', 'Network': 'Visual', 'X': 24, 'Y': -76, 'Z': 38},
    {'Region': 'Occipital_Mid_L', 'Network': 'Visual', 'X': -30, 'Y': -88, 'Z': 2},
    {'Region': 'Occipital_Mid_R', 'Network': 'Visual', 'X': 30, 'Y': -88, 'Z': 2},
    {'Region': 'Fusiform_L', 'Network': 'Visual', 'X': -40, 'Y': -50, 'Z': -20},
    {'Region': 'Fusiform_R', 'Network': 'Visual', 'X': 40, 'Y': -50, 'Z': -20},
]

salience_regions = [
    {'Region': 'Insula_L', 'Network': 'Salience', 'X': -32, 'Y': 22, 'Z': 2},
    {'Region': 'Insula_R', 'Network': 'Salience', 'X': 32, 'Y': 22, 'Z': 2},
    {'Region': 'ACC', 'Network': 'Salience', 'X': 0, 'Y': 32, 'Z': 26},
    {'Region': 'Amygdala_L', 'Network': 'Salience', 'X': -24, 'Y': -2, 'Z': -16},
    {'Region': 'Amygdala_R', 'Network': 'Salience', 'X': 24, 'Y': -2, 'Z': -16},
]

standard_regions = (
        corticostriatal_regions +
        frontoparietal_regions +
        visual_regions +
        salience_regions
)

standard_regions = pd.DataFrame(standard_regions)
cc200_xyz = cc200_coords[['X', 'Y', 'Z']].values
standard_xyz = standard_regions[['X', 'Y', 'Z']].values
distance_matrix = cdist(cc200_xyz, standard_xyz)
closest_indices = np.argmin(distance_matrix, axis=1)
closest_distances = np.min(distance_matrix, axis=1)

distance_threshold = 10
cc200_network_labels = ['Unknown'] * len(cc200_coords)

for i, (closest_idx, dist) in enumerate(zip(closest_indices, closest_distances)):
    if dist <= distance_threshold:
        cc200_network_labels[i] = standard_regions.iloc[closest_idx]['Network']
    else:
        pass

cc200_coords['Network'] = cc200_network_labels
weights = np.zeros(len(cc200_coords))
network_weights = {
    'Corticostriatal': 1,
    'Frontoparietal': 1,
    'Visual': 1,
    'Salience': 1
}

for idx, network_label in enumerate(cc200_coords['Network']):
    if network_label in network_weights:
        weights[idx] = network_weights[network_label]
    else:
        weights[idx] = 0  
important_roi_array = weights  
important_roi = torch.tensor([important_roi_array], dtype=torch.float32).to(device)  # shape: (num_nodes,)
weights = weights[:, np.newaxis]



import yaml
import itertools
with open(os.path.join(work_dir, f"config_bio2009_0749_base_add_ver2.yaml"), 'r', encoding='utf-8') as f:
    params = yaml.safe_load(f)
keys = list(params.keys())
values = [params[key] for key in keys]

for combination in itertools.product(*values):
    current_params = {}
    for key, value in zip(keys, combination):
        current_params[key] = value
    batch_size = int(current_params['batch_size'])
    n_hid = int(current_params['n_hid'])
    dropout = float(current_params['dropout'])
    lr = float(current_params['lr'])
    threshold = float(current_params['threshold'])
    L = int(current_params['L'])
    epoch_size = int(current_params['epoch_size'])
    lambda_lap = float(current_params['lambda_lap'])
    lambda_imp = float(current_params['lambda_imp'])
    alpha_v = float(current_params['alpha_v'])
    x_score = float(current_params['x_score'])

    print(f"接收到的参数组合：batch_size={batch_size}, n_hid={n_hid}, dropout={dropout},lambda_lap={lambda_lap},lambda_imp={lambda_imp},alpha_v={alpha_v},x_score={x_score}")

    cut_len = L
    atlas = 'cc200'
    data_array = np.load(os.path.join(work_dir, f"data/ABIDE/871_FC_lable/{atlas}_fc_871.npy"))
    labels_array = np.load(os.path.join(work_dir, f"data/ABIDE/871_FC_lable/{atlas}_871_labels.npy"))

    H_matrices = np.zeros((data_array.shape[0], data_array.shape[1], data_array.shape[2]))
    delta_matrices = np.zeros((data_array.shape[0], data_array.shape[1], data_array.shape[2]))

    k_neig = 3  
    is_probH = False  
    m_prob = 1  
    X_score = x_score

    for i in range(data_array.shape[0]):
        dist_mat = Eu_dis(data_array[i])
        H = construct_H_with_KNN_from_distance(dist_mat, k_neig, is_probH, m_prob)
        H_matrices[i, :, :] = H
        delta_matrices[i, :, :] = build_prior_matrix(weights,H, X_score)

    data_tensor = torch.Tensor(data_array)  
    H_tensor = torch.Tensor(H_matrices)  
    labels_tensor = torch.Tensor(labels_array).long() 
    delta_tensor = torch.Tensor(delta_matrices).long() 


    kf = KFold(n_splits=10)
    acc_scores = []  
    auc_scores = []  
    f1_scores = []  
    sen_scores = []  

    train_acc_scores = []  
    train_auc_scores = []  
    train_f1_scores = []  
    train_sen_scores = []  

    acc_scores_early_stops = []
    auc_scores_early_stops = []
    F1_scores_early_stops = []
    recall_scores_early_stops = []
    precision_scores_early_stops = []

    in_ch = data_array.shape[1]
    print(in_ch)

    for fold, (train_idx, test_idx) in enumerate(kf.split(data_array)):
        setup_seed(20)
        print(f"Fold {fold+1}")
        acc = 0
        epoch_best = 0
        X_train, X_test = data_tensor[train_idx].to(device), data_tensor[test_idx].to(device)
        H_train, H_test = H_tensor[train_idx].to(device), H_tensor[test_idx].to(device)
        delta_train, delta_test = delta_tensor[train_idx].to(device), delta_tensor[test_idx].to(device)
        y_train, y_test = labels_tensor[train_idx].to(device), labels_tensor[test_idx].to(device)

        train_dataset = TensorDataset(X_train, H_train,delta_train,y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = HGNN(in_ch=in_ch, n_class=2, n_hid=n_hid, dropout=dropout,threshold = threshold).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        with tqdm(total=epoch_size, desc="Training Progress") as pbar:
            for epoch in range(epoch_size):  
                for batch_X1, batch_H,batch_delta,batch_y in train_loader:
                    optimizer.zero_grad()

                    batch_X1 = batch_X1.to(device)
                    batch_H = batch_H.to(device)
                    batch_delta = batch_delta.to(device)

                    batch_y = batch_y.to(device)

                    outputs,edge_weights  = model(batch_X1, batch_H)
                    laplacian_loss = laplacian_constraint_batch(edge_weights[0], batch_delta)
                    imp_loss = importance_loss_pairwise_adaptive(batch_H, edge_weights, important_roi, alpha=alpha_v)
                    classification_loss = criterion(outputs, batch_y)

                    loss = classification_loss + lambda_lap * laplacian_loss + lambda_imp * imp_loss
                    loss.backward()
                    optimizer.step()

                model.eval()  
                with torch.no_grad():
                    test_preds,_  = model(X_test.to(device),H_test.to(device))
                    test_preds_prob = torch.softmax(test_preds, dim=1)  # 转换为概率

                    test_preds_label = torch.argmax(test_preds, dim=1)
                    y_test = y_test.cpu()
                    test_preds_label = test_preds_label.cpu()
                    test_preds = test_preds.cpu()
                    test_preds_prob = test_preds_prob.cpu()
                    acc_test = accuracy_score(y_test.numpy(), test_preds_label.numpy())
                    auc_test = roc_auc_score(y_test.numpy(), test_preds_prob[:, 1].numpy())
                    F1_test = f1_score(y_test.numpy(), test_preds_label.numpy())
                    recall_test = recall_score(y_test.numpy(), test_preds_label.numpy())
                    precision_test = precision_score(y_test.numpy(), test_preds_label.numpy())

                    if acc_test > acc and epoch > 5:
                        acc = acc_test
                        auc = auc_test
                        f1 = F1_test
                        recall = recall_test
                        precision = precision_test
                        epoch_best = epoch
                pbar.set_postfix(Loss=f"{loss.item():.4f}", acc_best=f"{acc:.4f}",epoch_best=f"{epoch_best:.4f}",refresh=True)
                pbar.update(1)  
            acc_scores_early_stops.append(acc)
            auc_scores_early_stops.append(auc)
            F1_scores_early_stops.append(f1)
            recall_scores_early_stops.append(recall)
            precision_scores_early_stops.append(precision)

        torch.save(model.state_dict(), os.path.join(models_dir, f"model_fold_{fold + 1}.pth"))

        model.eval()
        with torch.no_grad():
            train_preds,_  = model(X_train.to(device),H_train.to(device))
            train_preds_label = torch.argmax(train_preds, dim=1)

            y_train = y_train.cpu()
            train_preds_label = train_preds_label.cpu()
            train_preds = train_preds.cpu()

            train_acc = accuracy_score(y_train.numpy(), train_preds_label.numpy())
            train_auc = roc_auc_score(y_train.numpy(), torch.softmax(train_preds, dim=1)[:, 1].numpy())
            train_f1 = f1_score(y_train.numpy(), train_preds_label.numpy())
            train_sen = recall_score(y_train.numpy(), train_preds_label.numpy())

            train_acc_scores.append(train_acc)
            train_auc_scores.append(train_auc)
            train_f1_scores.append(train_f1)
            train_sen_scores.append(train_sen)

            print(f"Train ACC: {train_acc}, train_AUC: {train_auc},Train F1: {train_f1}, Train_Sensitivity: {train_sen}\n")
    avg_acc = np.mean(acc_scores)
    std_acc = np.std(acc_scores)
    avg_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)

    avg_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    avg_sen = np.mean(sen_scores)
    std_sen = np.std(sen_scores)

    acc_scores_early_stop = np.mean(acc_scores_early_stops)
    std_acc_scores_early_stop = np.std(acc_scores_early_stops)

    auc_scores_early_stop = np.mean(auc_scores_early_stops)
    std_auc_scores_early_stop = np.std(auc_scores_early_stops)

    F1_scores_early_stop = np.mean(F1_scores_early_stops)
    std_F1_scores_early_stop = np.std(F1_scores_early_stops)

    recall_scores_early_stop = np.mean(recall_scores_early_stops)
    std_recall_scores_early_stop = np.std(recall_scores_early_stops)

    precision_scores_early_stop = np.mean(precision_scores_early_stops)
    std_precision_scores_early_stop = np.std(precision_scores_early_stops)

    print(f"Average ACC: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"Average AUC: {avg_auc:.4f} ± {std_auc:.4f}")
    print(f"Average SEN: {avg_sen:.4f} ± {std_sen:.4f}")
    print(f"Average F1: {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"atlas = {atlas},dropout:{dropout},Average acc best: {acc_scores_early_stop:.4f} ± {std_acc_scores_early_stop:.4f}")
    print(f"atlas = {atlas},dropout:{dropout},Average auc best: {auc_scores_early_stop:.4f} ± {std_auc_scores_early_stop:.4f}")
    print(f"atlas = {atlas},dropout:{dropout},Average F1 best: {F1_scores_early_stop:.4f} ± {std_F1_scores_early_stop:.4f}")
    print(f"atlas = {atlas},dropout:{dropout},Average recall best: {recall_scores_early_stop:.4f} ± {std_recall_scores_early_stop:.4f}")
    print(f"atlas = {atlas},dropout:{dropout},Average precision best: {precision_scores_early_stop:.4f} ± {std_precision_scores_early_stop:.4f}")
