#!/bin/bash

gpuid=$1
task=$2
target_col=$3
split_dir=$4
split_names=$5
dataroots=("$@")

feat='extracted-vit_large_patch16_224.dinov2.uni_mass100k'
input_dim=2048  
mag_combine='204x'
patch_size=256

bag_size='-1'
batch_size=64
out_size=16     
out_type='allcat'
model_tuple='PANTHER,default'    
max_epoch=130     
lr=0.0001   
wd=0.00001
lr_scheduler='cosine'
opt='adamW'   
grad_accum=1
loss_fn='cox'   
n_label_bin=4   
alpha=0.5
em_step=70   
load_proto=1  
es_flag=0
tau=1.0
eps=1
n_fc_layer=0
proto_num_samples='1.0e+05'
save_dir_root=results

# Multimodal args
model_mm_type='coattn'    
append_embed='random'
histo_agg='mean'
num_coattn_layers=1

IFS=',' read -r model config_suffix <<< "${model_tuple}"
model_config=${model}_${config_suffix}
feat_name=$(echo $feat | sed 's/^extracted-//')
exp_code=${task}::${model_config}::${feat_name}
save_dir=${save_dir_root}/${exp_code}

th=0.00005
if awk "BEGIN {exit !($lr <= $th)}"; then
  warmup=0
  curr_lr_scheduler='constant'
else
  curr_lr_scheduler=$lr_scheduler
  warmup=1
fi

all_feat_dirs=""
for dataroot_path in "${dataroots[@]}"; do
  feat_dir=${dataroot_path}/extracted_mag${mag_combine}_patch${patch_size}_fp/${feat}/feats_h5
  if ! test -d $feat_dir
  then
    continue
  fi

  if [[ -z ${all_feat_dirs} ]]; then
    all_feat_dirs=${feat_dir}
  else
    all_feat_dirs=${all_feat_dirs},${feat_dir}
  fi
done



echo $feat_dir

# Actual command
cmd="CUDA_VISIBLE_DEVICES=$gpuid python -m training.main_survival \\
--data_source ${all_feat_dirs} \\
--results_dir ${save_dir} \\
--split_dir ${split_dir} \\
--split_names ${split_names} \\
--task ${task} \\
--target_col ${target_col} \\
--model_histo_type ${model} \\
--model_histo_config ${model}_default \\
--n_fc_layers ${n_fc_layer} \\
--in_dim ${input_dim} \\
--opt ${opt} \\
--lr ${lr} \\
--lr_scheduler ${curr_lr_scheduler} \\
--accum_steps ${grad_accum} \\
--wd ${wd} \\
--warmup_epochs ${warmup} \\
--max_epochs ${max_epoch} \\
--train_bag_size ${bag_size} \\
--batch_size ${batch_size} \\
--seed 1 \\
--num_workers 28 \\
--em_iter ${em_step} \\
--tau ${tau} \\
--n_proto ${out_size} \\
--out_type ${out_type} \\
--loss_fn ${loss_fn} \\
--nll_alpha ${alpha} \\
--n_label_bins ${n_label_bin} \\
--early_stopping ${es_flag} \\
--ot_eps ${eps} \\
--fix_proto \\
--num_coattn_layers ${num_coattn_layers} \\
--model_mm_type ${model_mm_type} \\
--append_embed ${append_embed} \\
--histo_agg ${histo_agg} \\
--net_indiv \\
"

# Specifiy representation path if load_proto is True
if [[ $load_proto -eq 1 ]]; then
  cmd="$cmd --load_proto \\
  --proto_path "splits/${split_dir}/representation/combine_c${out_size}_extracted-${feat_name}_hdbscan_num_${proto_num_samples}.pkl" \\
  "
fi

eval "$cmd"