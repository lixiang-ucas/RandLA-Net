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

dataset_path = './data/ISPRS/'
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

sub_grid_size = 0.1
original_pc_folder = join(dirname(dataset_path), 'original_ply_wh')
sub_pc_folder = join(dirname(dataset_path), 'input_{:.3f}_wh_wr'.format(sub_grid_size))
os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None
os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None
out_format = '.ply'

trainSet = np.loadtxt('./data/ISPRS/train_height.pts',skiprows=1)
# //X Y Z User_Data Classification Intensity return_number number_of_returns
prefix = 'trset'
label_w = trainSet[:,4].astype('uint8')
trainSet[:,3] = trainSet[:,3]/trainSet[:,3].max() #height above ground
trainSet[:,5] = trainSet[:,5]/trainSet[:,5].max() #reflectance
trainSet[:,6] = trainSet[:,6]/trainSet[:,6].max() #return_number
trainFeats = trainSet[:,[3,5,6]] #use reflectance and height above ground
# trainFeats = trainSet[:,5:6] #only use reflectance

xyz = trainSet[:,:3].astype(np.float32)
xyz -= xyz.min(0)
labels = trainSet[:,4].astype(np.uint8)
colors = trainFeats.astype(np.float32)

i = 0
save_path = join(original_pc_folder, prefix + str(i) + '.ply')
write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

# save sub_cloud and KDTree file
sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
# sub_colors = sub_colors / 255.0
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

all_label_expand = np.array(labels)
cls_num = np.zeros(num_classes,)
for c in range(num_classes):
    cls_num[c]= np.sum(all_label_expand==c)
print(cls_num)

all_label_expand = np.array(sub_labels)
cls_num = np.zeros(num_classes,)
for c in range(num_classes):
    cls_num[c]= np.sum(all_label_expand==c)
print(cls_num)


########################
####### test #########
########################
trainSet = np.loadtxt('./data/ISPRS/test_height3.pts',skiprows=2)
# //X Y Z Intensity return_number number_of_returns User_Data Classification
prefix = 'valset'

xyz = trainSet[:,:3].astype(np.float32)
xyz -= xyz.min(0)
labels = trainSet[:,-1].astype(np.uint8)
trainSet[:,6] = trainSet[:,6]/trainSet[:,6].max() #height above ground
trainSet[:,3] = trainSet[:,3]/trainSet[:,3].max() #reflectance
trainSet[:,4] = trainSet[:,4]/trainSet[:,4].max() #return_number
trainFeats = trainSet[:,[6,3,4]] 
colors = trainFeats.astype(np.float32)

save_path = join(original_pc_folder, prefix + str(i) + '.ply')
write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

# save sub_cloud and KDTree file
sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
# sub_colors = sub_colors / 255.0
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

all_label_expand = np.array(labels)
cls_num = np.zeros(num_classes,)
for c in range(num_classes):
    cls_num[c]= np.sum(all_label_expand==c)
print(cls_num)

all_label_expand = np.array(sub_labels)
cls_num = np.zeros(num_classes,)
for c in range(num_classes):
    cls_num[c]= np.sum(all_label_expand==c)
print(cls_num)