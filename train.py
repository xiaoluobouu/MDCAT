import torch
import numpy as np
import os
from dataset import Dataset, Dataset_ood, collate_group_fn
from utils.utils import compute_auc, compute_accuracy, data_split, get_cd_pretrain, get_group_data, cal_group_accs, filter_ood_data
from model_cd import MAMLModel
from policy import StraightThrough
from copy import deepcopy
from utils.configuration_2 import create_parser, initialize_seeds
import time
import os
import json
from datetime import datetime
from tqdm import *
from torch.utils.tensorboard import SummaryWriter
DEBUG = False if torch.cuda.is_available() else True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_val_score, best_test_score = 0, 0
best_val_auc, best_test_auc, best_val_wgc = 0, 0, 0
best_test_ood_score, best_test_ood_auc, best_test_wgc = 0, 0, 0
best_meta_params = []
best_epoch = -1
cd_model = 'irt'
best_train_dict = {}
best_model = {}
best_policy = {}
best_stu_param = {}
best_metrics = {}
best_train_ood_dict = {}


def clone_meta_params(batch):
    return [meta_params[0].expand(len(batch['input_labels']),  -1).clone(
    )]

def inner_algo(batch, config, new_params, create_graph=False):
    for _ in range(params.inner_loop):
        config['meta_param'] = new_params[0]
        res = model(batch, config)
        loss = res['train_loss']
        grads = torch.autograd.grad(
            loss, new_params, create_graph=create_graph)
        new_params = [(new_params[i] - params.inner_lr*grads[i])
                      for i in range(len(new_params))]
        del grads
        torch.cuda.empty_cache()
    config['meta_param'] = new_params[0]
    return

def pick_biased_samples(batch, config):
    new_params = clone_meta_params(batch)
    env_states = model.reset(batch)
    action_mask, train_mask = env_states['action_mask'], env_states['train_mask']
    for i in range(params.n_query):
        with torch.no_grad():
            state = model.step(env_states)
            train_mask = env_states['train_mask']
        if config['mode'] == 'train':
            train_mask_sample, actions = st_policy.policy(state, action_mask)
        else:
            with torch.no_grad():
                train_mask_sample, actions = st_policy.policy(state, action_mask)
        action_mask[range(len(action_mask)), actions] = 0
        env_states['train_mask'], env_states['action_mask'] = train_mask + \
            train_mask_sample.data, action_mask
        if config['mode'] == 'train':
            config['train_mask'] = train_mask_sample+train_mask
            inner_algo(batch, config, new_params, create_graph=True)
            res = model(batch, config)
            loss = res['loss']
            st_policy.update(loss)
        else:
            config['train_mask'] = train_mask_sample+train_mask
            inner_algo(batch, config, new_params, create_graph=True)
    config['train_mask'] = env_states['train_mask']
    return 


def run_biased(batch, config):
    new_params = clone_meta_params(batch)
    if config['mode'] == 'train':
        model.eval()
    pick_biased_samples(batch, config)
    # optimizer.zero_grad()
    meta_params_optimizer.zero_grad() 
    inner_algo(batch, config, new_params)
    if config['mode'] == 'train':
        model.train()
        # optimizer.zero_grad()
        res = model(batch, config)
        loss = res['loss']
        loss.backward()
        # optimizer.step()
        meta_params_optimizer.step()
        return res['output'], res['loss']
    else:
        with torch.no_grad():
            res = model(batch, config)
        return res['output'], res['train_loss']


def train_model():
    global best_val_auc, best_test_auc, best_val_score, best_test_score, \
        best_epoch,best_train_dict,best_model, best_metrics, best_policy,best_stu_param, cd_model,\
            best_meta_params,best_test_ood_auc,best_test_ood_score, best_train_ood_dict, best_val_wgc,best_test_wgc
    config['mode'] = 'train'
    config['epoch'] = epoch
    config['cd_model'] = cd_model
    model.train()
    N = [idx for idx in range(100, 100+params.repeat)]
    
    total_loss = 0.
    i = 0
    for batch in train_loader:
        if sampling == 'biased':
           _, loss = run_biased(batch, config)
           total_loss += loss.item()
           i += 1
           
    avg_loss = round(total_loss / i, 4)
    writer.add_scalar(f'{base}_{params.meta_lr}_{params.inner_lr}_{params.policy_lr}_{params.wait}_train_loss', avg_loss, epoch)
    val_scores, val_aucs = [],[]
    # Validation
    for idx in N:
        _, _, val_metrics = test_model(id_=idx, split='val')
        val_scores.append(val_metrics['acc'])
        val_aucs.append(val_metrics['auc'])
    val_score = round(sum(val_scores)/(len(N)+1e-20), 4)
    val_auc = sum(val_aucs)/(len(N)+1e-20)
    
    # test
    if best_val_score < val_score:
        best_val_score = val_score
        best_epoch = epoch
        best_meta_params = meta_params[0].detach().cpu().tolist()
        best_model = model.state_dict()
        best_policy = st_policy.policy.state_dict()
        test_scores, test_aucs, test_wgas = [],[],[]
        # Run on test set
        for idx in N:
            _, temp_test_data_dict, test_metrics = test_model(id_=idx, split='test', type = 'iid')
            test_scores.append(test_metrics['acc'])
            test_aucs.append(test_metrics['auc'])
            test_wgas.append(test_metrics['worst_acc'])
        test_score = round(sum(test_scores)/(len(N)+1e-20), 4)
        test_auc = round(sum(test_aucs)/(len(N)+1e-20), 4)
        test_wgc = round(sum(test_wgas)/(len(N)+1e-20), 4)
        test_metrics['acc'] = test_score
        test_metrics['worst_acc'] = test_wgc
        test_metrics['auc'] = test_auc
        
        test_ood_scores, test_ood_aucs, test_ood_wgas = [],[],[]
        for idx in N:
            _, temp_test_data_ood_dict, test_ood_metrics = test_model(id_=idx, split='test', type = 'ood')
            test_ood_scores.append(test_ood_metrics['acc'])
            test_ood_aucs.append(test_ood_metrics['auc'])
            test_ood_wgas.append(test_ood_metrics['worst_acc'])
        test_ood_score = round(sum(test_ood_scores)/(len(N)+1e-20), 4)
        test_ood_auc = round(sum(test_ood_aucs)/(len(N)+1e-20), 4)
        test_ood_wgc = round(sum(test_ood_wgas)/(len(N)+1e-20), 4)
        
        
        best_test_score = test_score
        best_test_auc = test_auc
        test_data_dict = temp_test_data_dict
        
        best_test_ood_score = test_ood_score
        best_test_ood_auc = test_ood_auc
        test_data_ood_dict = temp_test_data_ood_dict

        best_train_dict = {
            'test_data_dict':test_data_dict
        }
        best_train_ood_dict = {
            'test_data_ood_dict':test_data_ood_dict
        }

        best_metrics = {
            'best_epoch': best_epoch,
            'test_metrics': {
                'acc': test_score,
                'worst_acc': test_wgc,
                'auc': test_auc,
                
            },
            'test_ood_metrics': {
                'acc': test_ood_score,
                'worst_acc': test_ood_wgc,
                'auc': test_ood_auc
            },
            'test_group_accs': test_metrics['group_label_accs'],
            'test_ood_group_accs': test_ood_metrics['group_label_accs'],
        }
    print(f'Train_Epoch: {epoch}; training_loss: {avg_loss:.4f}; val_score: {val_score}; best_epoch: {best_epoch}; best_test_score: {best_test_score};')
    return best_metrics

def test_model(id_, split='val', type='iid'):
    model.eval()
    if split=='val':
        config['mode'] = 'val'
        valid_dataset.seed = id_
        loader = torch.utils.data.DataLoader(
            valid_dataset, collate_fn=collate_group_fn, batch_size=params.test_batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
    elif split == 'test':
        config['mode'] = 'test'
        if type == 'ood':
            final_test_dataset = test_ood_dataset
        else:
            final_test_dataset = test_dataset
        final_test_dataset.seed = id_
        loader = torch.utils.data.DataLoader(
            final_test_dataset, collate_fn=collate_group_fn, batch_size=params.test_batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
        
    all_group_targets = {0: {'label_0': [], 'label_1': []}, 
                         1: {'label_0': [], 'label_1': []}, 
                         2: {'label_0': [], 'label_1': []}}
    all_group_preds = {0: {'label_0': [], 'label_1': []}, 
                       1: {'label_0': [], 'label_1': []}, 
                       2: {'label_0': [], 'label_1': []}}
    total_loss, all_preds, all_targets = 0., [], []
    
    n_batch = 0
    for batch in loader:
        if sampling == 'biased':
            output, loss = run_biased(batch, config)
            output_labels = batch['output_labels']
            output_mask = batch['output_mask']
            group_types = batch['group_types']
            total_loss += loss.item()
        
        target = batch['output_labels'].float().numpy()
        mask = batch['output_mask'].numpy() == 1
        group_targets, group_preds = get_group_data(group_types, output_labels, output, output_mask)
        
        for group in [0, 1, 2]:
            if len(group_targets[group]['label_0']) > 0:
                all_group_targets[group]['label_0'].extend(group_targets[group]['label_0'])
                all_group_preds[group]['label_0'].extend(group_preds[group]['label_0'])
            
            if len(group_targets[group]['label_1']) > 0:
                all_group_targets[group]['label_1'].extend(group_targets[group]['label_1'])
                all_group_preds[group]['label_1'].extend(group_preds[group]['label_1'])
        
        all_preds.append(output[mask])
        all_targets.append(target[mask])
        n_batch += 1
        
    all_pred = np.concatenate(all_preds, axis=0)
    all_target = np.concatenate(all_targets, axis=0)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    
    metrics = {
        'acc' : round(accuracy, 4),
        'auc' : round(auc, 4),
    }
    stu_data = []

    if split == 'test':
        group_label_accs, min_acc = cal_group_accs(all_group_targets, all_group_preds)
        metrics['worst_acc'] = min_acc
        metrics['group_label_accs'] = group_label_accs
    return total_loss/n_batch, stu_data, metrics


if __name__ == "__main__":
    params = create_parser()
    print(params)
    config = {}
    initialize_seeds(params.seed)
    # build cconfig
    config['env'] = params.env
    config['var_hyper'] = params.var_hyper
    config['reweight'] = params.reweight
    config['lam'] = params.lam
    config['alpha'] = params.alpha
    config['mix_ratio'] = params.mix_ratio
    config['hyp_1'] = params.hyp_1
    # config['drop_out'] = params.drop_out
    config['groupDRO'] = params.groupDRO
    
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
                        n_question=params.n_question, question_dim=params.question_dim, 
                        low_dim = params.low_dim,know_emb=concepts_emb,cd_type=base).to(device)
    if(base == 'biirt'):
        meta_params = [torch.Tensor(
            1,1).normal_(-1., 1.).to(device).requires_grad_()]
    else:
        meta_params = [torch.Tensor(
            1, params.question_dim).normal_(-1., 1.).to(device).requires_grad_()]
    model_path = get_cd_pretrain(params.dataset, base, params.seed)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    if sampling == 'biased':
        policy_name = 'st'
        betas = (0.9, 0.999)
        st_policy = StraightThrough(params.n_question, params.n_question,
                                    params.policy_lr, betas)
    best_meta_params = meta_params[0].detach().cpu().tolist()
    
    meta_params_optimizer = torch.optim.SGD(
        meta_params, lr=params.meta_lr, weight_decay=2e-6, momentum=0.9)
    print(model)
    
    # load data
    data_path = os.path.normpath('data/train_task_'+params.dataset+'.json')
    train_data, valid_data, test_data = data_split(
        data_path, params.fold,  params.seed)
    test_ood_data = filter_ood_data(test_data)
    train_dataset, valid_dataset, test_dataset = Dataset(
            train_data), Dataset(valid_data), Dataset(test_data)
    if params.training_type == 'ood':
        test_ood_dataset = Dataset_ood(test_ood_data)
    num_workers = 1
    collate_group_fn = collate_group_fn(params.n_question)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=collate_group_fn, batch_size=params.train_batch_size, num_workers=num_workers, shuffle=True, drop_last=True)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'runs/{base}_{params.inner_lr}_{params.meta_lr}_{params.policy_lr}_{params.seed}_{current_time}'
    writer = SummaryWriter(log_dir)
    
    start_time = time.time()
    for epoch in tqdm(range(params.n_epoch)):
        results  = train_model()
        torch.cuda.empty_cache()
        if epoch >= (best_epoch+params.wait):
            break
        
    writer.close()
        
    timestamp = datetime.now().strftime(f'%Y%m%d_%H%M%S')
    r_folder_name = f"./history/{params.dataset}/{params.model}/{params.training_type}/{timestamp}"
    os.makedirs(r_folder_name, exist_ok=True)
    with open(f'{r_folder_name}/final_test_data.json', 'w') as f:
        json.dump(best_train_dict['test_data_dict'], f, indent=2)
    with open(f'{r_folder_name}/final_test_ood_data.json', 'w') as f:
        json.dump(best_train_ood_dict['test_data_ood_dict'], f, indent=2)
    with open(f'{r_folder_name}/final_metrics.json', 'w') as f:
        json.dump(best_metrics, f, indent=2)
    best_model_path = os.path.join(r_folder_name, f"best_model.pth")
    best_policy_path = os.path.join(r_folder_name, f"best_policy.pth")
    torch.save(best_model, best_model_path)
    torch.save(best_policy, best_policy_path)
    
    print(f'data_save_path: {r_folder_name}')
    print(f'log_dir: {log_dir}')
    print(f'best_epoch: {best_epoch}')
    end_time = time.time()
    cost_time = (end_time-start_time) / 3600
    print(f'cost_time: {cost_time:.2f}h')
    end_time = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    print(f'end_time: {end_time}')
    print(f'best_metrics: {best_metrics}')
    print('Training completed!')