from __future__ import print_function

import argparse
import torch
from torch.utils.data import DataLoader
from wsi_datasets import WSIProtoDataset
from utils.utils import seed_torch, read_splits
from utils.file_utils import save_pkl
from utils.proto_utils import cluster
import os
from os.path import join as j_
import pickle 
import h5py
import time
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F 

def build_datasets(csv_splits, batch_size=1, num_workers=2, train_kwargs={}):
    dataset_splits = {}
    for k in csv_splits.keys(): # ['train']
        df = csv_splits[k]
        dataset_kwargs = train_kwargs.copy()
        dataset = WSIProtoDataset(df, **dataset_kwargs)

        batch_size = 1
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        dataset_splits[k] = dataloader
        print(f'split: {k}, n: {len(dataset)}')

    return dataset_splits

def swish(x):
    return x * torch.sigmoid(x)

# integrating the two parts of 'features'
def merge_h5_files(data_source_x40, data_source_x20, output_dir, l1_lambda=0.01):
    os.makedirs(output_dir, exist_ok=True)

    for path_x40, path_x20 in zip(data_source_x40, data_source_x20):
        print(f'Processing files in:\n - x40: {path_x40}\n - x20: {path_x20}')
        
        file_list_x40 = [f for f in os.listdir(path_x40) if f.endswith('.h5')]
        file_list_x20 = [f for f in os.listdir(path_x20) if f.endswith('.h5')]

        # check if the same file exists in two paths
        common_files = set(file_list_x40).intersection(file_list_x20)
        total_files = len(common_files)

        start_time = time.time()

        for idx, file_name in enumerate(tqdm(common_files, desc='Merging H5 files')):
            output_file_path = os.path.join(output_dir, file_name)
            
            # check if the file already exists
            if os.path.exists(output_file_path):
                print(f'Skipping {file_name} as it already exists.')
                continue
            
            file_path_x40 = os.path.join(path_x40, file_name)
            file_path_x20 = os.path.join(path_x20, file_name)

            with h5py.File(file_path_x40, 'r') as f_x40, h5py.File(file_path_x20, 'r') as f_x20:
                # Read features and coords
                features_x40 = f_x40['features'][:]
                coords_x40 = f_x40['coords'][:]
                features_x20 = f_x20['features'][:]
                coords_x20 = f_x20['coords'][:]

                # Check if coords are consistent
                if not (coords_x40 == coords_x20).all():
                    raise ValueError(f'Coords do not match in file {file_name}')

                merged_features = torch.cat([torch.from_numpy(features_x40), torch.from_numpy(features_x20)], dim=-1)
                merged_features = F.gelu(merged_features)  

            with h5py.File(output_file_path, 'w') as f_out:
                f_out.create_dataset('features', data=merged_features.numpy())
                f_out.create_dataset('coords', data=coords_x40)

            print(f'File {idx+1}/{total_files} ({file_name}) merged successfully.')

        total_time = time.time() - start_time
        print(f'Merging completed for {total_files} files. Total time: {total_time:.2f} seconds.')

def main(args):
    seed_torch(args.seed)
    csv_splits = read_splits(args)
    print('\nsuccessfully read splits for: ', list(csv_splits.keys()))

    os.makedirs(j_(args.split_dir, 'prototypes'), exist_ok=True)
    # combining multi-scale data
    merge_h5_files(args.data_source_x40, args.data_source_x20, 'path/to/tcga_brca_combined_feature')   

    print('\nProcessing combine images...', end=' ')
    train_kwargs_combine = dict(data_source=args.data_source_combine) 
    dataset_splits_combine = build_datasets(csv_splits, batch_size=1, num_workers=args.num_workers, train_kwargs=train_kwargs_combine)
    combine_loader_train = dataset_splits_combine['train']  

    _, combine_weights = cluster(combine_loader_train,
                             n_proto=args.n_proto,
                             n_iter=args.n_iter,
                             n_init=args.n_init,
                             feature_dim=args.in_dim,
                             mode=args.mode,
                             n_proto_patches=args.n_proto_patches,
                             use_cuda=True if torch.cuda.is_available() else False)
    
    combined_save_fpath = j_(args.split_dir, 'prototypes', f"combine_prototypes_c{args.n_proto}_{args.data_source_combine[0].split('/')[-2]}_{args.mode}_num_{args.n_proto_patches:.1e}.pkl")
    save_pkl(combined_save_fpath, {'prototypes': combine_weights})

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
# model / loss fn args ###
parser.add_argument('--n_proto', type=int, help='Number of prototypes')
parser.add_argument('--n_proto_patches', type=int, default=10000,
                    help='Number of patches per prototype to use. Total patches = n_proto * n_proto_patches')
parser.add_argument('--n_init', type=int, default=5,
                    help='Number of different KMeans initialization (for FAISS)')
parser.add_argument('--n_iter', type=int, default=50,
                    help='Number of iterations for Kmeans clustering')
parser.add_argument('--in_dim', type=int)
parser.add_argument('--mode', type=str, choices=['kmeans', 'faiss', 'clique', 'vlique', 'dbscan', 'hdbscan'], default='kmeans')
parser.add_argument('--data_source_combine', type=str, default=None,
                    help='manually specify the data source')
parser.add_argument('--data_source_x20', type=str, default=None,
                    help='manually specify the data source')
parser.add_argument('--data_source_x40', type=str, default=None,
                    help='manually specify the data source')
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use')
parser.add_argument('--split_names', type=str, default='train,val,test',
                    help='delimited list for specifying names within each split')
parser.add_argument('--num_workers', type=int, default=8)


args = parser.parse_args()

if __name__ == "__main__":
    args.split_dir = j_('splits', args.split_dir)
    args.split_name = os.path.basename(args.split_dir)
    print('split_dir: ', args.split_dir)
    
    args.data_source_combine = [src for src in args.data_source_combine.split(',')]
    args.data_source_x20 = [src for src in args.data_source_x20.split(',')]
    args.data_source_x40 = [src for src in args.data_source_x40.split(',')]

    
    results = main(args)
