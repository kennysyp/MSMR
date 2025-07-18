#!/bin/bash
gpuid=$1
split_dir=$2
split_names=$3
dataroots_combine=$4
dataroots_x20=$5
dataroots_x40=$6

feat='extracted-vit_large_patch16_224.dinov2.uni_mass100k'  # UNI
input_dim=2048  
mag_combine='204x'
mag_x20='20x'
mag_x40='40x'
patch_size=256
n_sampling_patches=100000 
mode='hdbscan'  # Optional 'kmeans', 'faiss', 'clique', 'vlique', 'dbscan','hdbscan'
n_proto=16      
n_init=3  

echo "Dataroots combine: ${dataroots_combine[@]}"
echo "Dataroots x20: ${dataroots_x20[@]}"
echo "Dataroots x40: ${dataroots_x40[@]}"

### combine
# Validity check for feat paths
all_feat_dirs_combine=""
for dataroot_path_combine in "${dataroots_combine[@]}"; do
  echo "Checking path: $dataroot_path_combine"
  feat_dir_combine=${dataroot_path_combine}/extracted_mag${mag_combine}_patch${patch_size}_fp/${feat}/feats_h5
  if ! test -d $feat_dir_combine
  then
    echo "Directory not found: $feat_dir_combine"
    continue
  fi

  if [[ -z ${all_feat_dirs_combine} ]]; then
    all_feat_dirs_combine=${feat_dir_combine}
  else
    all_feat_dirs_combine=${all_feat_dirs_combine},${feat_dir_combine}
  fi
done

### x20
# Validity check for feat paths
all_feat_dirs_x20=""
for dataroot_path_x20 in "${dataroots_x20[@]}"; do
  echo "Checking path: $dataroot_path_x20"
  feat_dir_x20=${dataroot_path_x20}/extracted_mag${mag_x20}_patch${patch_size}_fp/${feat}/feats_h5
  if ! test -d $feat_dir_x20
  then
    echo "Directory not found: $feat_dir_x20"
    continue
  fi

  if [[ -z ${all_feat_dirs_x20} ]]; then
    all_feat_dirs_x20=${feat_dir_x20}
  else
    all_feat_dirs_x20=${all_feat_dirs_x20},${feat_dir_x20}
  fi
done

### x40
# Validity check for feat paths
all_feat_dirs_x40=""
for dataroot_path_x40 in "${dataroots_x40[@]}"; do
  feat_dir_x40=${dataroot_path_x40}/extracted_mag${mag_x40}_patch${patch_size}_fp/${feat}/feats_h5
  if ! test -d $feat_dir_x40
  then
    continue
  fi

  if [[ -z ${all_feat_dirs_x40} ]]; then
    all_feat_dirs_x40=${feat_dir_x40}
  else
    all_feat_dirs_x40=${all_feat_dirs_x40},${feat_dir_x40}
  fi
done

echo "Data sources combine: $all_feat_dirs_combine"
echo "Data sources x20: $all_feat_dirs_x20"
echo "Data sources x40: $all_feat_dirs_x40"


### --data_source_x40 ${feat_dir_x40} \\
cmd="CUDA_VISIBLE_DEVICES=$gpuid python -m training.main_cluster \\
--mode ${mode} \\
--data_source_combine ${all_feat_dirs_combine} \\
--data_source_x20 ${all_feat_dirs_x20} \\
--data_source_x40 ${all_feat_dirs_x40} \\
--split_dir ${split_dir} \\
--split_names ${split_names} \\
--in_dim ${input_dim} \\
--n_proto_patches ${n_sampling_patches} \\
--n_proto ${n_proto} \\
--n_init ${n_init} \\
--seed 1 \\
--num_workers 28 \\
"

eval "$cmd"