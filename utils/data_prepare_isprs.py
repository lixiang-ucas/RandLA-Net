from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import pandas as pd
import os, sys, glob, pickle
import pdb

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply
from helper_tool import DataProcessing as DP

dataset_path = './data/ISPRS/block30m/'
label_to_names = {0: 'Powerline',
                   1: 'low_veg',
                   2: 'imp_surf',
                   3: 'car',
                   4: 'fence',
                   5: 'roof',
                   6: 'facade',
                   7: 'shrub',
                   8: 'tree'}
num_classes = len(label_to_names)
label_values = np.sort([k for k, v in label_to_names.items()])
label_to_idx = {l: i for i, l in enumerate(label_values)}

sub_grid_size = 0.5
original_pc_folder = join(dirname(dataset_path), 'original_ply')
sub_pc_folder = join(dirname(dataset_path), 'input_{:.3f}'.format(sub_grid_size))
os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None
os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None
out_format = '.ply'

train_f = open('./data/ISPRS/block30m/train_merge_min_norm_fea.pickle', 'rb')
train_xyz, train_label, train_feats = pickle.load(train_f, encoding='bytes')
train_f.close()

test_f = open('./data/ISPRS/block30m/test_merge_min_norm_fea_paper_height.pickle', 'rb')
test_xyz, test_label, test_feats = pickle.load(test_f, encoding='bytes')
test_feats = [tt[:,1:2] for tt in test_feats]
test_f.close()

print('train length, test length', len(train_xyz), len(test_xyz))

all_xyz = train_xyz + test_xyz
all_label = train_label + test_label
all_feats = train_feats +  test_feats

all_label_expand = []
for i in range(len(train_xyz) + len(test_xyz)):
    if i<len(train_xyz):
        prefix = 'trset'
        feat_idx = 3
    else:
        prefix = 'valset'
        feat_idx = 0

    xyz = all_xyz[i].astype(np.float32)
    xyz -= xyz.min(0)
    colors = all_feats[i][:,[feat_idx]].astype(np.float32)
    colors = colors/colors.max(0)
    colors = np.concatenate([colors, colors, colors], axis=1).astype('float32')
    labels = all_label[i].astype(np.uint8)
    all_label_expand.extend(labels.flatten())
    save_path = join(original_pc_folder, prefix + str(i) + '.ply')
    write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    # save sub_cloud and KDTree file
    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
#     sub_colors = sub_colors / 255.0
    sub_ply_file = join(sub_pc_folder, prefix + str(i) + '.ply')
    write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    search_tree = KDTree(sub_xyz)
    kd_tree_file = join(sub_pc_folder, prefix + str(i) + '_KDTree.pkl')
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
    proj_idx = proj_idx.astype(np.int32)
    proj_save = join(sub_pc_folder, prefix + str(i) + '_proj.pkl')
    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx, labels], f)
pdb.set_trace()
all_label_expand = np.array(all_label_expand)
cls_num = np.zeros(num_classes,)
for c in range(num_classes):
    cls_num[c]= np.sum(all_label_expand==c)
print(cls_num)
