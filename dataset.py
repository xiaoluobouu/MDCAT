import torch
import numpy as np
from torch.utils import data

import random
from utils.utils import open_json, dump_json

class collate_fn(object):
    def __init__(self, n_question):
        self.n_question = n_question

    def __call__(self, batch):
        B = len(batch)
        input_labels = torch.zeros(B, self.n_question).long()
        output_labels = torch.zeros(B, self.n_question).long()
        input_mask = torch.zeros(B, self.n_question).long()
        output_mask = torch.zeros(B, self.n_question).long()

        real_user_ids = torch.zeros(B).long()
        for b_idx in range(B):
            input_labels[b_idx, batch[b_idx]['input_question'].long(
            )] = batch[b_idx]['input_label'].long()
            input_mask[b_idx, batch[b_idx]['input_question'].long()] = 1
            output_labels[b_idx, batch[b_idx]['output_question'].long(
            )] = batch[b_idx]['output_label'].long()
            output_mask[b_idx, batch[b_idx]['output_question'].long()] = 1
            real_user_ids[b_idx] = batch[b_idx]['real_user_id'].long()
        output = {'input_labels': input_labels,  'input_mask': input_mask,
                  'output_labels': output_labels, 'output_mask': output_mask,
                  'real_user_ids':real_user_ids}            
        return output

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data, seed=None):
        self.data = data
        self.seed = seed
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.data[index]
        observed_index = np.array([idx for idx in range(len(data['q_ids']))])
        if not self.seed:
            np.random.shuffle(observed_index)
        else:
            random.Random(index+self.seed).shuffle(observed_index)
        N = len(observed_index)
        target_index = observed_index[-N//5:]
        trainable_index = observed_index[:-N//5]
        input_label = data['labels'][trainable_index]
        input_question = data['q_ids'][trainable_index]
        output_label = data['labels'][target_index]
        output_question = data['q_ids'][target_index]
        label_1_ratio = sum(np.array(output_label))/len(output_label)
        if label_1_ratio < 0.4:
            group = np.array(0)
        elif label_1_ratio < 0.6:
            group = np.array(1)
        else:
            group = np.array(2)
        real_user_id = data['user_id']

        output = {'input_label': torch.FloatTensor(input_label), 'input_question': torch.FloatTensor(input_question),
                  'output_question': torch.FloatTensor(output_question), 'output_label': torch.FloatTensor(output_label),
                  'real_user_id':torch.from_numpy(real_user_id), 'group_type':torch.from_numpy(group)}
        # 'input_ans': torch.FloatTensor(input_ans)
        return output

class Dataset_ood(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data, seed=None):
        self.data = data
        self.seed = seed
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.data[index]
        total_num = len(data['q_ids'])
        out_num = total_num // 5  # output 占 20%
        in_num = total_num - out_num  # input 占 80%

        all_pos_idx = [idx for idx in range(total_num) if data['labels'][idx] == 1]
        all_neg_idx = [idx for idx in range(total_num) if data['labels'][idx] == 0]

        pos_num = out_num // 2
        neg_num = pos_num
        if pos_num > len(all_pos_idx) or neg_num > len(all_neg_idx):
            raise ValueError(f"Not enough positive or negative samples for user {data['user_id']}")

        random.seed(self.seed + index if self.seed else None)
        pos_ids = random.sample(all_pos_idx, pos_num)
        neg_ids = random.sample(all_neg_idx, neg_num)

        output_sample_ids = pos_ids + neg_ids
        random.shuffle(output_sample_ids)

        output_q_ids = [data['q_ids'][idx] for idx in output_sample_ids]
        output_labels = [data['labels'][idx] for idx in output_sample_ids]
        input_sample_ids = [idx for idx in range(total_num) if idx not in output_sample_ids]
        input_q_ids = [data['q_ids'][idx] for idx in input_sample_ids]
        input_labels = [data['labels'][idx] for idx in input_sample_ids]

        label_1_ratio = sum(input_labels) / len(input_labels)
        if label_1_ratio < 0.4:
            group = np.array(0)
        elif label_1_ratio < 0.6:
            group = np.array(1)
        else:
            group = np.array(2)

        real_user_id = data['user_id']

        output = {
            'input_label': torch.FloatTensor(input_labels),
            'input_question': torch.FloatTensor(input_q_ids),
            'output_question': torch.FloatTensor(output_q_ids),
            'output_label': torch.FloatTensor(output_labels),
            'real_user_id': torch.tensor(real_user_id, dtype=torch.long),
            'group_type': torch.tensor(group, dtype=torch.long)
        }
        return output
    
class collate_group_fn(object):
    def __init__(self, n_question):
        self.n_question = n_question

    def __call__(self, batch):
        B = len(batch)
        input_labels = torch.zeros(B, self.n_question).long()
        output_labels = torch.zeros(B, self.n_question).long()
        input_mask = torch.zeros(B, self.n_question).long()
        output_mask = torch.zeros(B, self.n_question).long()

        real_user_ids = torch.zeros(B).long()
        group_types = torch.zeros(B).long()
        for b_idx in range(B): 
            input_labels[b_idx, batch[b_idx]['input_question'].long(
            )] = batch[b_idx]['input_label'].long()
            input_mask[b_idx, batch[b_idx]['input_question'].long()] = 1
            output_labels[b_idx, batch[b_idx]['output_question'].long(
            )] = batch[b_idx]['output_label'].long()
            output_mask[b_idx, batch[b_idx]['output_question'].long()] = 1
            real_user_ids[b_idx] = batch[b_idx]['real_user_id'].long()
            group_types[b_idx] = batch[b_idx]['group_type'].long()
        output = {'input_labels': input_labels,  'input_mask': input_mask,
                  'output_labels': output_labels, 'output_mask': output_mask,
                  'real_user_ids':real_user_ids, 'group_types':group_types}            
        return output
    

    
class Dataset_get_params(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data):
        'Initialization'
        self.data = data
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        output = {
            'labels': torch.FloatTensor(data['labels']),
            'q_ids': torch.FloatTensor(data['q_ids']),
            'real_user_id': torch.from_numpy(data['real_user_id'])
        }
        return output
    
    
class collate_fn_get_params(object):
    def __init__(self, n_question):
        self.n_question = n_question
        
    def __call__(self, batch):
        B = len(batch)
        labels = torch.zeros(B, self.n_question).long()
        mask = torch.zeros(B, self.n_question).long()

        real_user_ids = torch.zeros(B).long()
        
        for b_idx in range(B):
            labels[b_idx, batch[b_idx]['q_ids'].long(
            )] = batch[b_idx]['labels'].long()
            mask[b_idx, batch[b_idx]['q_ids'].long()] = 1
            real_user_ids[b_idx] = batch[b_idx]['real_user_id'].long()
        output = {'labels': labels,  'mask': mask,
                  'real_user_ids':real_user_ids}            
        return output