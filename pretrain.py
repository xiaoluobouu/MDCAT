import torch
import numpy as np
import os
from dataset import collate_fn_get_params, Dataset_get_params
from utils.utils import compute_auc, compute_accuracy, data_split, batch_accuracy, data_split_by_log
from model_cd import MAMLModel
from policy import PPO, Memory, StraightThrough
from copy import deepcopy
from utils.configuration_2 import create_parser, initialize_seeds
import time
import os
import json
from datetime import datetime
import pickle
import random
from sklearn.metrics import confusion_matrix
from torch import nn
from tqdm import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_val_score = 0
best_epoch = -1
best_model = {}
best_policy = {}
best_metrics = {}


    
def train_model():
    global best_val_score, best_epoch, best_model, best_policy,best_val_auc,best_metrics,best_meta_params
    model.train()
    config['mode'] = 'train'
    total_loss = 0
    idx = 0
    for batch in train_loader:
        stu_ids = batch['real_user_ids']
        stu_params_batch = stu_params[stu_ids]
        stu_optimizer.zero_grad()
        exer_optimizer.zero_grad()
        res = model._forward_get_params(batch, stu_params_batch)
        train_loss = res['loss']
        total_loss += train_loss.item()
        train_loss.backward()
        exer_optimizer.step()
        stu_optimizer.step()
        idx += 1
    model.eval()
    avg_loss = round(total_loss / idx, 4)
    all_preds, all_targets = [], []
    for batch in valid_loader:
        stu_ids = batch['real_user_ids']
        stu_params_batch = stu_params[stu_ids]
        res = model._forward_get_params(batch, stu_params_batch)
        output = res['output']
        target = batch['labels'].float().numpy()
        mask = batch['mask'].numpy() == 1
        all_preds.append(output[mask])
        all_targets.append(target[mask])
    all_pred = np.concatenate(all_preds, axis=0)
    all_target = np.concatenate(all_targets, axis=0)
    auc = round(compute_auc(all_target, all_pred), 4)
    accuracy = round(compute_accuracy(all_target, all_pred), 4)
    
    if best_val_score < accuracy:
        best_val_score = accuracy
        best_val_auc = auc
        best_epoch = epoch
        best_model = model.state_dict()
        best_metrics = {'val_accuracy': accuracy, 'val_auc': auc}
    print(f'Epoch {epoch} train_loss: {avg_loss}  val_accuracy: {accuracy} val_auc: {auc} best_epoch: {best_epoch} best_val_score: {best_val_score} best_val_auc: {best_val_auc} ')
    return {'train_loss': train_loss, 'val_auc': auc, 'val_accuracy': accuracy}


if __name__ == "__main__":
    params = create_parser()
    print(params)
    
    
    config = {}
    initialize_seeds(params.seed)
    
    # build model
    base, sampling = params.model.split('-')[0], params.model.split('-')[-1]
    concept_name = './data/'+params.dataset +'_concept_map.json'
    with open(concept_name, 'r') as file:
        concepts = json.load(file)
    num_concepts = params.question_dim
    concepts_emb = [[0.] * num_concepts for i in range(params.n_question)]
    for i in range(params.n_question):
        for concept in concepts[str(i)]:
            concepts_emb[i][concept] = 1.0
    concepts_emb = torch.tensor(concepts_emb, dtype=torch.float32).to(device)
    model = MAMLModel(sampling=sampling, n_query=params.n_query,
                        n_question=params.n_question, question_dim=params.question_dim, low_dim = params.low_dim,know_emb=concepts_emb,cd_type=base).to(device)
    if(base == 'bikancd'):
        meta_params = [torch.Tensor(
            1, params.low_dim).normal_(-1., 1.).to(device).requires_grad_()]
    elif(base == 'biirt'):
        meta_params = [torch.Tensor(
            1,1).normal_(-1., 1.).to(device).requires_grad_()]
    else:
        meta_params = [torch.Tensor(
            1, params.question_dim).normal_(-1., 1.).to(device).requires_grad_()]
    
    data_path = os.path.normpath('data/train_task_'+params.dataset+'.json')
    raw_train_data, raw_valid_data, raw_test_data = data_split(
        data_path, params.fold,  params.seed)
                    
    train_total = []
    for i in range(len(raw_train_data)):
        train_total.append(raw_train_data[i])
    for i in range(len(raw_valid_data)):
        train_total.append(raw_valid_data[i])
    stu_num = len(train_total)

    train_data, valid_data = data_split_by_log(train_total)
    train_dataset, valid_dataset = Dataset_get_params(train_data), Dataset_get_params(valid_data)
    collate_fn = collate_fn_get_params(params.n_question)
    num_workers = 3
    train_loader = torch.utils.data.DataLoader(
            train_dataset, collate_fn=collate_fn, batch_size=params.train_batch_size, num_workers=num_workers, shuffle=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(
            valid_dataset, collate_fn=collate_fn, batch_size=params.train_batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
    

    stu_params_list = [meta_params[0].clone().to(device) for _ in range(stu_num)]
    stu_params = nn.Parameter(torch.cat(stu_params_list, dim=0), requires_grad=True)
    nn.init.xavier_normal_(stu_params)
    stu_optimizer = torch.optim.Adam([stu_params], lr=params.lr, weight_decay=1e-8)
    exer_optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=1e-8)
    
    best_meta_params = meta_params[0].detach().cpu().tolist()
    for epoch in tqdm(range(params.n_epoch)):
        results  = train_model()
        torch.cuda.empty_cache()
        if epoch >= (best_epoch+params.wait):
            break
        
    timestamp = datetime.now().strftime(f'%Y%m%d_%H%M%S')
    r_folder_name = f"./history/{params.dataset}/{params.model}/pretrain/{params.seed}_{timestamp}"
    print(f'{r_folder_name}')
    print(f'best_epoch:{best_epoch}')
    os.makedirs(r_folder_name, exist_ok=True)

    best_model_path = os.path.join(r_folder_name, f"best_model.pth")
    torch.save(best_model, best_model_path)
    with open(f'{r_folder_name}/final_metrics.json', 'w') as f:
        json.dump(best_metrics, f, indent=2)
    print('pretrain completed!')
    