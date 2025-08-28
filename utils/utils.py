import json
import torch
import numpy as np
from sklearn import metrics
import os
import random
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(current_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def open_json(path_):
    with open(path_) as fh:
        data = json.load(fh)
    return data


def dump_json(path_, data):
    with open(path_, 'w') as fh:
        json.dump(data, fh, indent=2)
    return data


def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def list_to_string(socres):
    str_scores = [str(s) for s in socres]
    return ','.join(str_scores)


def data_split(datapath, fold, seed):
    data = open_json(datapath)
    random.Random(seed).shuffle(data)
    fields = ['user_id','q_ids',  'labels']  # 'ans', 'correct_ans',
    del_fields = []
    for f in data[0]:
        if f not in fields:
            del_fields.append(f)
    for d in data:
        for f in fields:
            d[f] = np.array(d[f])
        for f in del_fields:
            if f not in fields:
                del d[f]
    N = len(data)//5
    test_fold, valid_fold = fold-1, fold % 5
    test_data = data[test_fold*N: (test_fold+1)*N]
    valid_data = data[valid_fold*N: (valid_fold+1)*N]
    train_indices = [idx for idx in range(len(data))]
    train_indices = [idx for idx in train_indices if idx //
                     N != test_fold and idx//N != valid_fold]
    train_data = [data[idx] for idx in train_indices]

    return train_data, valid_data, test_data

def data_split_by_log(train_total, seed=None):
    stu_data = train_total
    train_data, valid_data = [], []
    for i, data in enumerate(stu_data):
        observed_index = np.array([idx for idx in range(len(data['q_ids']))])
        if not seed:
            np.random.shuffle(observed_index)
        else:
            random.Random(i+seed).shuffle(observed_index)
        N = len(observed_index)
        target_index = observed_index[:N//5]
        trainable_index = observed_index[N//5:]

        input_label = data['labels'][trainable_index]
        input_question = data['q_ids'][trainable_index]
        real_user_id = data['user_id']
        output_label = data['labels'][target_index]
        output_question = data['q_ids'][target_index]
        
        new_user_id = np.array(i)
        input_data = {'labels': input_label, 'q_ids': input_question, 'real_user_id': new_user_id}
        output_data = {'labels': output_label, 'q_ids': output_question, 'real_user_id': new_user_id}
        train_data.append(input_data)
        valid_data.append(output_data)
    return train_data, valid_data


def batch_accuracy(output, batch):
    output[output > 0.5] = 1
    output[output <= 0.5] = 0
    target = batch['output_labels'].float().numpy()
    mask = batch['output_mask'].numpy() == 1
    accuracy = torch.from_numpy(np.sum((target == output) * mask, axis=-1) /
                                np.sum(mask, axis=-1)).float()  # B,
    return accuracy


def try_makedirs(path_):
    if not os.path.isdir(path_):
        try:
            os.makedirs(path_)
        except FileExistsError:
            pass

def get_cd_pretrain(dataset, base,seed):
    data = open_json(f'utils/model_cfg.json') 
    seed = str(seed)
    return data[dataset][base][seed]

def get_data(dataset, version):
    train_data_path = os.path.normpath(f'data/iid/{version}/{dataset}/train_task_{dataset}_train.json')
    valid_data_path = os.path.normpath(f'data/iid/{version}/{dataset}/train_task_{dataset}_valid.json')
    test_data_path = os.path.normpath(f'data/iid/{version}/{dataset}/train_task_{dataset}_test.json')
    test_data_ood_path = os.path.normpath(f'data/ood/{version}/{dataset}/train_task_{dataset}_test.json')
    train_data = open_json(train_data_path)
    valid_data = open_json(valid_data_path)
    test_data = open_json(test_data_path)
    test_data_ood = open_json(test_data_ood_path)
    fields = ['user_id','q_ids',  'labels']
    for d in [train_data, valid_data, test_data, test_data_ood]:
        for entry in d:
            for f in fields:
                entry[f] = np.array(entry[f])
    return train_data, valid_data, test_data, test_data_ood

def get_group_data(group_types, output_labels, output, mask):
    group_preds = {0: {'label_0': [], 'label_1': []},
                   1: {'label_0': [], 'label_1': []},
                   2: {'label_0': [], 'label_1': []}}
    group_targets = {0: {'label_0': [], 'label_1': []},
                     1: {'label_0': [], 'label_1': []},
                     2: {'label_0': [], 'label_1': []}}
    indices = {
        0: np.where(group_types == 0)[0],
        1: np.where(group_types == 1)[0],
        2: np.where(group_types == 2)[0]
    }
    for group in [0, 1, 2]:
        idx = indices[group]
        mask_group = (mask[idx].numpy() == 1)
        preds = output[idx][mask_group]
        targets = output_labels[idx][mask_group]
        label_0_mask = targets.numpy() == 0
        label_1_mask = targets.numpy() == 1
        group_preds[group]['label_0'].extend(preds[label_0_mask])
        group_targets[group]['label_0'].extend(targets[label_0_mask])
        group_preds[group]['label_1'].extend(preds[label_1_mask])
        group_targets[group]['label_1'].extend(targets[label_1_mask])

    return group_targets, group_preds

def cal_group_accs(group_targets, group_preds):
    group_label_accs = {}
    min_acc = 1.0
    
    for group in [0, 1, 2]:
        label_0_preds = np.array(group_preds[group]['label_0'])
        label_0_targets = np.array(group_targets[group]['label_0'])
        label_1_preds = np.array(group_preds[group]['label_1'])
        label_1_targets = np.array(group_targets[group]['label_1'])
        label_0_acc = compute_accuracy(label_0_targets, label_0_preds) if len(label_0_targets) > 0 else 0
        label_1_acc = compute_accuracy(label_1_targets, label_1_preds) if len(label_1_targets) > 0 else 0
        
        min_acc = min(min_acc, label_0_acc, label_1_acc)
        group_label_accs[group] = {
            '0': round(label_0_acc, 4),
            '1': round(label_1_acc, 4)
        }
    
    return group_label_accs, min_acc

def filter_ood_data(test_data):
    test_ood_data = []
    for data in test_data:
        out_num = len(data['q_ids']) // 5
        all_pos_idx = [idx for idx in range(len(data['labels'])) if data['labels'][idx] == 1]
        all_neg_idx = [idx for idx in range(len(data['labels'])) if data['labels'][idx] == 0]
        pos_num = out_num // 2
        neg_num = pos_num
        if pos_num > len(all_pos_idx) or neg_num > len(all_neg_idx):
            continue  
        test_ood_data.append(data)
    return test_ood_data