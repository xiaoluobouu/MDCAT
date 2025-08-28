from collections import namedtuple
from torch.distributions import Categorical
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch.utils.checkpoint import checkpoint

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def pick_random_sample(input_mask,n_query,n_question):
    if n_query==-1:
        return input_mask.detach().clone()
    train_mask = torch.zeros(input_mask.shape[0], n_question).long().to(device)
    actions = torch.multinomial(input_mask.float(), n_query, replacement=False)
    train_mask = train_mask.scatter(dim=1, index=actions, value=1)
    return train_mask

def get_inputs(batch):
    input_labels = batch['input_labels'].to(device).float()
    input_mask = batch['input_mask'].to(device)
    #input_ans = batch['input_ans'].to(device)-1
    input_ans = None
    return input_labels, input_ans, input_mask

def get_outputs(batch):
    output_labels, output_mask = batch['output_labels'].to(
        device).float(), batch['output_mask'].to(device)  # B,948
    return output_labels, output_mask, 


def compute_loss(output, labels, mask, reduction= True):
    # calculate input_loss
    loss_function = nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_function(output, labels) * mask
    if reduction:
        return loss.sum()/mask.sum()
    else:
        return loss.sum()

def compute_group_loss(output, labels, mask, group_types, reduction= True):
    loss_function = nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_function(output, labels) * mask
    if reduction:
        return loss.sum()/mask.sum()
    else:
        # calculate group loss
        indices_0 = torch.where(group_types == 0)[0]
        indices_1 = torch.where(group_types == 1)[0]
        indices_2 = torch.where(group_types == 2)[0]
        loss_0 = loss[indices_0].sum()/len(indices_0)
        loss_0 = round(loss_0.detach().cpu().item(), 2)
        loss_1 = loss[indices_1].sum()/len(indices_1)
        loss_1 = round(loss_1.detach().cpu().item(), 2)
        loss_2 = loss[indices_2].sum()/len(indices_2)
        loss_2 = round(loss_2.detach().cpu().item(), 2)
        # 组装loss字典
        group_loss = [loss_0, loss_1, loss_2]
        return group_loss, loss.sum()
    
def compute_mixup_loss(output, labels, reduction= False):
    loss_function = nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_function(output, labels)
    if reduction:
        return loss.sum()/labels.sum()
    else:
        return loss.sum()
    

def normalize_loss(output, labels, mask):
    # normalize loss
    loss_function = nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_function(output, labels) * mask
    count = mask.sum(dim =-1)+1e-8 #N,1
    loss = 10. * torch.sum(loss, dim =-1)/count
    return loss.sum()

def env_sample(group_types,env_num=3):
    # sample three numbers from [0, 1] with sum = 1 as environment
    indices = {}
    alpha = [0.5, 0.5, 0.5]
    samples = np.random.dirichlet(alpha,  size=env_num)
    indices_0 = torch.where(group_types == 0)[0]
    indices_1 = torch.where(group_types == 1)[0]
    indices_2 = torch.where(group_types == 2)[0]
    all_num = int(len(group_types) * 0.8)
    for i in range(env_num):
        sample_list = samples[i]
        num_0 = int(all_num * sample_list[0])
        num_1 = int(all_num * sample_list[1])
        num_2 = int(all_num * sample_list[2])
        # sample certain number of students from each group
        sample_0 = indices_0[torch.randperm(len(indices_0))[:num_0]]
        sample_1 = indices_1[torch.randperm(len(indices_1))[:num_1]]
        sample_2 = indices_2[torch.randperm(len(indices_2))[:num_2]]
        sample_all = torch.cat((sample_0, sample_1, sample_2))
        indices['env_'+str(i)] = sample_all
    return indices

def cal_env_loss(output, labels, mask, group_types, env_num):
    # calculate env loss
    envs = env_sample(group_types, env_num)
    all_erm_loss = []
    for key in envs:
        env = envs[key]
        if(len(env)==0):
            continue
        erm_loss = compute_loss(output[env], labels[env], mask[env], reduction=False)/len(env)
        all_erm_loss.append(erm_loss)
    
    var_erm = torch.var(torch.stack(all_erm_loss))
    mean_erm = torch.mean(torch.stack(all_erm_loss))
    return torch.stack(all_erm_loss), mean_erm, var_erm

class IRT(nn.Module):
    def __init__(self, n_question):
        super().__init__()
        self.e_diff = nn.Parameter(torch.zeros(1,n_question))
    
    def forward(self, student_embed):
        output = student_embed - self.e_diff
        return output
    
    def get_e_diff(self, sel_e_ids):
        sel_e_diff = self.e_diff[:,sel_e_ids].T
        return sel_e_diff.detach().cpu().numpy()
    
    def cal_s_e(self, s_emb):
        s_e = s_emb - self.e_diff
        abs_s_e = abs(s_e)
        return abs_s_e
    
    def get_mix_output(self, stu_emb, exer_pairs, lam):
        exer_pairs = exer_pairs.t()
        exer_emb = self.e_diff[:,exer_pairs].reshape(2, -1)
        e_diff = lam * exer_emb[0] + (1 - lam) * exer_emb[1]
        output = stu_emb - e_diff
        return output

class NCDM(nn.Module):
    def __init__(self, n_question, question_dim=123, know_emb=None, p=0.5):
        super().__init__()
        self.prednet_input_len = question_dim
        self.prednet_len1, self.prednet_len2 = 128,64  # changeable
        self.kn_emb = know_emb
        self.e_diff = nn.Parameter(torch.zeros(n_question,self.prednet_input_len))
        self.e_disc = nn.Parameter(torch.full((n_question,1), 0.5))
        # Linear can be replaced by PosLinear
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)
        
    def forward(self, student_embed):
        e_diff = self.e_diff
        e_disc = self.e_disc
        kn_emb = self.kn_emb
        student_embed = student_embed.unsqueeze(1)
    
        input_x = e_disc * (student_embed - e_diff) *kn_emb.to(device)
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = self.prednet_full3(input_x)
        output = output.squeeze()
        return output
    
    def cal_s_e(self, s_emb):
        s_e = s_emb - self.e_diff
        abs_s_e = abs(s_e)
        return abs_s_e
    
    def get_item_params(self):
        return {
            k: v for k, v in self.named_parameters()
        }
    
    def get_mix_output(self, student_embed, exer_pairs, lam):
        exer_pairs = exer_pairs.t() 
        e_diff_1 = self.e_diff[exer_pairs[0]]  
        e_diff_2 = self.e_diff[exer_pairs[1]] 
        e_disc_1 = self.e_disc[exer_pairs[0]]  
        e_disc_2 = self.e_disc[exer_pairs[1]] 
        mixed_e_diff = lam * e_diff_1 + (1 - lam) * e_diff_2
        mixed_e_disc = lam * e_disc_1 + (1 - lam) * e_disc_2

        kn_emb = lam * self.kn_emb[exer_pairs[0]] + (1 - lam) * self.kn_emb[exer_pairs[1]]
        input_x = mixed_e_disc * (student_embed - mixed_e_diff) * kn_emb
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = self.prednet_full3(input_x)
        output = output.squeeze()
        return output
        

class MAMLModel(nn.Module):
    def __init__(self, n_question,question_dim =1,dropout=0.5, sampling='active', n_query=10, low_dim = 20, know_emb=None,cd_type='biirt'):
        super().__init__()
        self.n_query = n_query
        self.sampling = sampling
        self.sigmoid = nn.Sigmoid()
        self.n_question = n_question
        self.n_know = question_dim
        self.cd_type = cd_type 
        self.know_emb = know_emb
        if self.cd_type == 'biirt':
            self.cd_model = IRT(n_question)        
        elif self.cd_type == 'bincdm':
            self.cd_model = NCDM(n_question, self.n_know, know_emb, dropout)
        
    def reset(self, batch):
        # reset state
        input_labels, _, input_mask = get_inputs(batch)
        obs_state = ((input_labels-0.5)*2.)  # B, 948
        train_mask = torch.zeros(
            input_mask.shape[0], self.n_question).long().to(device)
        env_states = {'obs_state': obs_state, 'train_mask': train_mask,
                      'action_mask': input_mask.clone()}
        return env_states
    
    def step(self, env_states):
        # get the state
        obs_state,  train_mask = env_states[
            'obs_state'], env_states['train_mask']
        state = obs_state*train_mask  # B, 948
        return state
        
    def get_item_params(self):
        return self.cd_model.get_item_params()
        
    def forward(self, batch, config):
        #get inputs
        input_labels = batch['input_labels'].to(device).float()
        student_embed = config['meta_param']
        output = self.compute_output(student_embed)
        train_mask = config['train_mask']
        group_types = batch['group_types']
        lam = config['lam']
        alpha = config['alpha']
        ratio = config['mix_ratio']
        hyp_1 = config['hyp_1']
        reweight = config['reweight']
        eta =  config['groupDRO']
        var_erm = torch.tensor(0.0)
        all_erm_loss = torch.tensor(0.0)
        #compute loss
        if config['mode'] == 'train':
            output_labels, output_mask = get_outputs(batch)
            #meta model parameters 
            group_loss, loss = compute_group_loss(output, output_labels, output_mask,  group_types, reduction=False)
            output_loss = loss/len(train_mask)
            if(config['env'] != 0):
                # cal env loss
                all_erm_loss, _, var_erm = cal_env_loss(output, output_labels, output_mask, group_types, config['env'])
                output_loss = output_loss + config['var_hyper']*var_erm
            if(lam != 0.0):
                # mixup
                output_loss = self.get_mixup_loss(student_embed, output_labels, output_mask, group_types, output, alpha, ratio, hyp_1) / len(train_mask)
            if(reweight != 0):
                # reight
                output_loss = self.get_reweight_loss(output, output_labels, output_mask, group_types)/len(train_mask) 
            if(eta != 0):
                # groupDRO
                output_loss = self.get_groupDRO_loss(output, output_labels, output_mask, group_types, eta)
            if self.n_query!=-1:
                input_loss = compute_loss(output, input_labels, train_mask, reduction=False)
            else:
                input_loss = normalize_loss(output, input_labels, train_mask)
            all_erm_loss = all_erm_loss.detach().cpu().float().tolist()
            return {'loss': output_loss, 
                    'train_loss': input_loss,
                    'env_loss': round(var_erm.detach().cpu().float().item(), 6),
                    'all_erm_loss': all_erm_loss,
                    'normal_group_loss': group_loss,
                    'output': self.sigmoid(output).detach().cpu().numpy()
                    }
        else:
            input_loss = compute_loss(output, input_labels, train_mask,reduction=False)
            return {'output': self.sigmoid(output).detach().cpu().numpy(), 'train_loss': input_loss}
    
    def _forward_simp(self, batch, config):
        #get inputs
        input_labels = batch['input_labels'].to(device).float()
        student_embed = config['meta_param']
        output = self.compute_output(student_embed)
        train_mask = config['train_mask']
        #compute loss
        if config['mode'] == 'train':
            output_labels, output_mask = get_outputs(batch)
            #meta model parameters 
            output_loss = compute_loss(output, output_labels, reduction=False)/len(train_mask)
            #for adapting meta model parameters
            if self.n_query!=-1:
                input_loss = compute_loss(output, input_labels, train_mask, reduction=False)
            else:
                input_loss = normalize_loss(output, input_labels, train_mask)
            return {'loss': output_loss, 'train_loss': input_loss, 'output': self.sigmoid(output).detach().cpu().numpy()}
        else:
            input_loss = compute_loss(output, input_labels, train_mask,reduction=False)
            return {'output': self.sigmoid(output).detach().cpu().numpy(), 'train_loss': input_loss}
    
    def _forward_get_params(self, batch, stu_params_batch):
        labels = batch['labels'].to(device).float()
        student_embed = stu_params_batch
        output = self.compute_output(student_embed)
        mask = batch['mask'].to(device)
        loss = compute_loss(output, labels, mask, reduction=False)  
        return {'loss': loss, 'output': self.sigmoid(output).detach().cpu().numpy()}

    def compute_output(self, student_embed):
        output = self.cd_model(student_embed) 
        return output
    
    def get_mixup_loss(self, stu_emb, labels, mask, group_types, output, alpha, ratio, hyp_1=1):
        """cal mixup loss
        """
        # origin_loss
        loss_function = nn.BCEWithLogitsLoss(reduction='none')
        origin_loss = loss_function(output, labels) * mask
        origin_loss = origin_loss.sum()

        indices_0 = torch.where(group_types == 0)[0].to(device)
        indices_1 = torch.where(group_types == 1)[0].to(device)
        indices_2 = torch.where(group_types == 2)[0].to(device)

        def process_group(group_indices, target_label, ratio, hyp_1):
            mix_loss = 0.
            for index in group_indices:
                # find the most similar student in group 1
                temp_stu_emb = stu_emb[index]
                similarities = torch.norm(stu_emb[indices_1] - temp_stu_emb, dim=1)
                closest_indices = indices_1[torch.argsort(similarities)[:1]]
                # get the valid indices for the current student
                valid_indices = torch.where(mask[index] == 1)[0]
                label_indices = valid_indices[labels[index, valid_indices] == target_label]
                valid_indices_1 = torch.where(mask[closest_indices] == 1)
                label_indices_1 = valid_indices_1[1][labels[closest_indices, valid_indices_1[1]] == target_label]
                if len(label_indices) == 0 or len(label_indices_1) == 0:
                    continue
                # calculate the number of pairs to mix
                num_pairs = int(len(label_indices) * ratio)
                sampled_exer = label_indices[torch.randint(0, len(label_indices), (num_pairs,))]
                sampled_indices_1 = torch.randint(0, len(label_indices_1), (num_pairs,))
                sampled_exer_1 = label_indices_1[sampled_indices_1]
                exer_pairs = torch.stack([sampled_exer, sampled_exer_1], dim=1)
                labels_pairs = labels[index, sampled_exer]
                # mixup
                lam = np.random.beta(alpha, alpha)
                mix_output = self.cd_model.get_mix_output(temp_stu_emb, exer_pairs, lam)
                loss = compute_mixup_loss(mix_output, labels_pairs, reduction=False)
                mix_loss += loss * hyp_1
            return mix_loss

        mix_loss_1 = process_group(indices_0, target_label=1, ratio=ratio, hyp_1=hyp_1)
        mix_loss_2 = process_group(indices_2, target_label=0, ratio=ratio, hyp_1=hyp_1)
        origin_loss += mix_loss_1 + mix_loss_2

        return origin_loss
    
    def get_reweight_loss(self, output, labels, mask, group_types):
        origin_mask = mask.clone()
        
        indices_0 = torch.where(group_types == 0)[0].to(device)
        indices_2 = torch.where(group_types == 2)[0].to(device)
        
        if len(indices_0) > 0:
            valid_mask_0 = mask[indices_0] == 1                  
            valid_labels_0 = labels[indices_0] == 1              
            target_positions_0 = valid_mask_0 & valid_labels_0   
            rows, cols = torch.where(target_positions_0)  
            origin_mask[indices_0[rows], cols] = 3

        if len(indices_2) > 0:
            valid_mask_2 = mask[indices_2] == 1                  
            valid_labels_2 = labels[indices_2] == 0              
            target_positions_2 = valid_mask_2 & valid_labels_2   
            rows, cols = torch.where(target_positions_2)  
            origin_mask[indices_2[rows], cols] = 3
        loss = compute_loss(output, labels, origin_mask, reduction=False)
        return loss
    
    def get_groupDRO_loss(self, output, labels, mask, group_types, eta):
        num_groups = 3
        q = torch.ones(num_groups * 2, device=device)
        group_losses = torch.zeros(num_groups * 2, device=device)  
        group_counts = torch.zeros(num_groups * 2, device=device)
        for group in range(num_groups):
            group_indices = torch.where(group_types == group)[0]  
            if len(group_indices) == 0:
                continue  
            group_output = output[group_indices]
            group_labels = labels[group_indices]
            group_mask = mask[group_indices] == 1
            temp_output = group_output[group_mask]
            temp_labels = group_labels[group_mask]
            for label_value in [0, 1]:
                label_indices = torch.where(temp_labels == label_value)[0]
                if len(label_indices) == 0:
                    continue
                label_output = temp_output[label_indices]
                label_labels = temp_labels[label_indices]
                loss_function = nn.BCEWithLogitsLoss(reduction='none')
                label_loss = loss_function(label_output, label_labels)
                group_id = group * 2 + label_value  
                group_losses[group_id] = label_loss.sum() / len(group_indices)
                group_counts[group_id] = len(label_indices)  
        q *= torch.exp(eta * group_losses)
        q /= q.sum() 
        total_loss = 0
        for group_id in range(num_groups * 2):
            if group_counts[group_id] > 0:
                total_loss += q[group_id] * group_losses[group_id]
        return total_loss / num_groups
    
    
