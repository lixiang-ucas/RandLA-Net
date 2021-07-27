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

dataset_path = './data/DFC/'
label_to_names = {1: 'Ground',
                   2: 'high_veg',
                   3: 'Building',
                   4: 'water',
                   5: 'Bridge_deck',
                   0: 'unkonw'}

num_classes = len(label_to_names)
# label_values = np.sort([k for k, v in self.label_to_names.items()])
# label_to_idx = {l: i for i, l in enumerate(self.label_values)}

sub_grid_size = 0.06
original_pc_folder = join(dirname(dataset_path), 'original_ply_mask_v2')
sub_pc_folder = join(dirname(dataset_path), 'input_{:.3f}_mask_v2'.format(sub_grid_size))
os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None
os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None
out_format = '.ply'

train_f = open('./data/DFC/train_1599.pickle', 'rb')
# train_xyz, train_label, train_feats = pickle.load(train_f, encoding='bytes')
train_xyz, train_label, train_feats = pickle.load(train_f,encoding='iso-8859-1')
train_f.close()

test_f = open('./data/DFC/test_160.pickle', 'rb')
# test_xyz, test_label, test_feats = pickle.load(test_f, encoding='bytes')
test_xyz, test_label, test_feats = pickle.load(test_f,encoding='iso-8859-1')
test_f.close()

print('train length, test length', len(train_xyz), len(test_xyz))

all_xyz = train_xyz + test_xyz
all_label = train_label + test_label
all_feats = train_feats +  test_feats
####create mask index]
all_masks = []
for i in range(len(all_xyz)):
    masks = np.zeros((len(all_xyz[i]),1), dtype=bool)
    mask_idx = np.random.choice(len(all_xyz[i]), int(0.1*len(all_xyz[i])), replace=False)
    masks[mask_idx] = True
    all_masks.append(masks)

pdb.set_trace()
    
all_label_expand = []
for i in range(len(train_xyz) + len(test_xyz)):
    if i<len(train_xyz):
        prefix = 'trset'
        feat_idx = 0
    else:
        prefix = 'valset'
        feat_idx = 0

    xyz = all_xyz[i].astype(np.float32)
    xyz -= xyz.min(0)
    colors = all_feats[i][:,[feat_idx]].astype(np.float32)
    colors = colors/colors.max(0)
    colors = np.concatenate([colors, colors, colors], axis=1).astype('float32')
    labels = all_label[i].astype(np.uint8)
    masks = all_masks[i].astype(np.uint8)
    
    print(xyz.shape, np.unique(labels))
    labels[labels==2] = 1
    labels[labels==5] = 2
    labels[labels==6] = 3
    labels[labels==9] = 4
    labels[labels==17] = 5
    
    if i<len(train_xyz):
        all_label_expand.extend(labels.flatten())
    save_path = join(original_pc_folder, prefix + str(i) + '.ply')
    write_ply(save_path, (xyz, colors, labels, masks), ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'mask'])

    # save sub_cloud and KDTree file
    sub_xyz, sub_colors_masks, sub_labels = DP.grid_sub_sampling(xyz, np.concatenate([colors, masks],-1), labels, sub_grid_size)
    sub_colors = sub_colors_masks[:,:-1]
    sub_masks = sub_colors_masks[:,-1]
    sub_ply_file = join(sub_pc_folder, prefix + str(i) + '.ply')
    write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels, sub_masks], ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'mask'])

    search_tree = KDTree(sub_xyz)
    kd_tree_file = join(sub_pc_folder, prefix + str(i) + '_KDTree.pkl')
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
    proj_idx = proj_idx.astype(np.int32)
    proj_save = join(sub_pc_folder, prefix + str(i) + '_proj.pkl')
    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx, labels, masks], f)

all_label_expand = np.array(all_label_expand)
cls_num = np.zeros(num_classes,)
for c in range(num_classes):
    cls_num[c]= np.sum(all_label_expand==c)
print(cls_num)
