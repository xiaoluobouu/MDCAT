# seed=11
training_type='ood'
# reweight 1 or not 0
reweight=0
# mixup 1 or not 0
lam=1
alpha=0.4
hyp_1=0.6
# inv 1 or not 0
env=0
var_hyper=0.1
# groupDRO 1 or not 0
groupDRO=0
# met can be pretrain, raw, reweight, mixup
met='mixup'
seed=221
n_query=10
ratio=2.0
dataset='eedi-3'
model_name='biirt-biased'
log_path="history/log/${seed}/${dataset}/${met}"
mkdir -p ${log_path}

# pretrain
CUDA_VISIBLE_DEVICES=0 python pretrain.py --dataset ${dataset} --model ${model_name} \
    --lam ${lam} --alpha ${alpha} --mix_ratio ${ratio} --hyp_1 ${hyp_1} \
    --n_query ${n_query} --training_type ${training_type} --inner_lr 0.05 --lr 1e-2 --meta_lr 1e-4 --policy_lr 0.002 \
    --seed ${seed} --n_epoch 150 --wait 30  --env ${env} --var_hyper ${var_hyper} --reweight ${reweight} \
    --groupDRO ${groupDRO} | tee -a ${log_path}/${model_name}_${hyp_1}.log


# train
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset ${dataset} --model ${model_name} \
#     --lam ${lam} --alpha ${alpha} --mix_ratio ${ratio} --hyp_1 ${hyp_1} \
#     --n_query ${n_query} --training_type ${training_type} --inner_lr 0.05 --lr 1e-2 --meta_lr 1e-4 --policy_lr 0.002 \
#     --seed ${seed} --n_epoch 150 --wait 30  --env ${env} --var_hyper ${var_hyper} --reweight ${reweight} \
#     --groupDRO ${groupDRO} | tee -a ${log_path}/${model_name}_${hyp_1}.log


