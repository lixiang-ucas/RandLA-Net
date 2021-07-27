from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import helper_tf_util
# from helper_tool import ConfigISPRS as cfg
import time, pickle, argparse, glob, os
from helper_ply import read_ply
import pdb

import sys,os
sys.path.append(os.path.join('./', 'Util'))
import Tool
import SmoothConstraint
import ProbLabelPropagation as PLP

def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


class Network:
    def __init__(self, dataset, config):
        flat_inputs = dataset.flat_inputs
        self.config = config
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                self.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            else:
#                 self.saving_path = self.config.saving_path
                self.saving_path = join(self.config.saving_path, time.strftime('Log_%Y-%m-%d_%H-%M-%S', time.gmtime()))
            makedirs(self.saving_path) if not exists(self.saving_path) else None
            self.Log_file = open(join(self.saving_path, 'log_train_' + dataset.name + str(dataset.val_split) + '.txt'), 'a') #log_train_dfc_
        
        with tf.variable_scope('inputs'):
            self.inputs = dict()
            num_layers = config.num_layers
            
            cfg = config
            self.batch_sz = cfg.batch_size
            self.point_sz = cfg.num_points
            K = cfg.k_n
            NP = self.point_sz
            
#             #####version 1, for testing
#             self.inputs['xyz'] = flat_inputs[:num_layers]
#             self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]
#             self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]
#             self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]
#             self.inputs['features'] = flat_inputs[4 * num_layers]
#             self.inputs['labels'] = flat_inputs[4 * num_layers + 1]
#             self.inputs['mask_boolen'] = flat_inputs[4 * num_layers + 2]
#             self.inputs['input_inds'] = flat_inputs[4 * num_layers + 3]
#             self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 4]
#             #####end of version 1
            
            #####version 2, for training
            self.inputs['xyz'] = cfg.num_layers*[None]
            self.inputs['neigh_idx'] = cfg.num_layers*[None]
            self.inputs['sub_idx'] = cfg.num_layers*[None]
            self.inputs['interp_idx'] = cfg.num_layers*[None]
    
            for i in range(cfg.num_layers):
                self.inputs['xyz'][i] = tf.placeholder(dtype=tf.float32, shape=[None, NP, 3], name='xyz')
                self.inputs['neigh_idx'][i] = tf.placeholder(dtype=tf.int32, shape=[None, NP, K], name='neigh_idx')
                self.inputs['sub_idx'][i] = tf.placeholder(dtype=tf.int32, shape=[None, NP//cfg.sub_sampling_ratio[i], K], name='pool_i')
                self.inputs['interp_idx'][i] = tf.placeholder(dtype=tf.int32, shape=[None, NP, 1], name='up_i')
                NP = NP// cfg.sub_sampling_ratio[i]
            
            self.inputs['features'] = tf.placeholder(dtype=tf.float32, shape=[None, self.point_sz, 6], name='features')  # B*N
            self.inputs['labels'] = tf.placeholder(dtype=tf.int32, shape=[None, self.point_sz], name='labels')  # B*N
            self.inputs['mask_boolen'] = tf.placeholder(dtype=tf.int32, shape=[None, self.point_sz], name='Mask')  # B*N
            self.inputs['input_inds'] = tf.placeholder(dtype=tf.int32, shape=[None, self.point_sz], name='point_idx')  # B*N
            self.inputs['cloud_inds'] = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='cloud_idx')  # B*N
            #####end of version 2

            self.labels = self.inputs['labels']
            self.mask_boolen = self.inputs['mask_boolen']
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.eval_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            self.class_weights = DP.get_class_weights(dataset.name)
            
            self.rampup = 20
            

        with tf.variable_scope('layers'):
            self.logits = self.inference(self.inputs, self.is_training)

        #####################################################################
        # Ignore the invalid point (unlabeled) when calculating the loss #
        #####################################################################
        self.batch_logits = self.logits
        self.batch_labels = self.labels
        self.batch_masks = self.mask_boolen
        
        with tf.variable_scope('loss'):
            self.logits = tf.reshape(self.logits, [-1, config.num_classes])
            self.labels = tf.reshape(self.labels, [-1])
            self.mask_boolen = tf.reshape(self.mask_boolen, [-1])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)
            valid_mask_init = tf.gather(self.mask_boolen, valid_idx, axis=0)

            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)
            valid_masks = tf.gather(reducing_list, valid_mask_init)
            
            self.WeakSupLoss()
            if True:
                self.loss = self.get_loss(valid_logits, valid_labels, self.class_weights, valid_masks)
            else:
                self.loss = self.get_loss(valid_logits, valid_labels, self.class_weights)

        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results'):
            self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
        
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
        self.test_writer = tf.summary.FileWriter(config.test_sum_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        
        
        self.ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.logits,
                       self.labels,
                       self.mask_boolen,
                       self.accuracy]
        
        ###########################
        
        if cfg.name == 'isprs':
            self.name = 'ISPRS'
            self.path = './data/ISPRS/'
            self.label_to_names = {0: 'Powerline',
                                   1: 'low_veg',
                                   2: 'imp_surf',
                                   3: 'car',
                                   4: 'fence',
                                   5: 'roof',
                                   6: 'facade',
                                   7: 'shrub',
                                   8: 'tree'}
            self.all_files = glob.glob(join(self.path, 'original_ply_wh_mask', '*.ply'))
        elif cfg.name == 'dfc':
            self.name = 'DFC'
            self.path = './data/DFC/'
            self.label_to_names = {1: 'Ground',
                           2: 'high_veg',
                           3: 'Building',
                           4: 'water',
                           5: 'Bridge_deck',
                           0: 'unkonw'}
            self.all_files = glob.glob(join(self.path, 'original_ply_mask_v2', '*.ply'))

        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])
        self.val_split = 'valset'
        
        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.val_masks = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_masks = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size)
        ###########################
        
    def load_sub_sampled_clouds(self, sub_grid_size):
        if self.config.name == 'isprs':
            tree_path = join(self.path, 'input_{:.3f}_wh_mask'.format(sub_grid_size))
        elif self.config.name == 'dfc':
            tree_path = join(self.path, 'input_{:.3f}_mask_v2'.format(sub_grid_size))
        
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.val_split in cloud_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))
            
            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']
            sub_masks = data['mask']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                print('kd_tree_file', kd_tree_file)
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_masks[cloud_split] += [sub_masks]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels, masks = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                self.val_masks += [masks]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))
    
    def get_batch_data(self, split='training'):
        cfg = self.config
        if split == 'training':
            num_per_epoch = cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_batch_size
        
        self.possibility[split] = []
        self.min_possibility[split] = []
        
        # Random initialize
        for i, tree in enumerate(self.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]
        
        
        queried_pc_xyz_list = []
        queried_pc_colors_list = []
        queried_pc_labels_list = []
        queried_pc_masks_list = []
        queried_idx_list = []
        cloud_idx_list = []

        for i in range(num_per_epoch):

            # Choose the cloud with the lowest probability
            cloud_idx = int(np.argmin(self.min_possibility[split]))

            # choose the point with the minimum of possibility in the cloud as query point
            point_ind = np.argmin(self.possibility[split][cloud_idx])

            # Get all points within the cloud from tree structure
            points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

            # Center point of input region
            center_point = points[point_ind, :].reshape(1, -1)

            # Add noise to the center point
            noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
            pick_point = center_point + noise.astype(center_point.dtype)

            # Check if the number of points in the selected cloud is less than the predefined num_points
            if len(points) < cfg.num_points:
                # Query all points within the cloud
                queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
            else:
                # Query the predefined number of points
                queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

            # Shuffle index
            queried_idx = DP.shuffle_idx(queried_idx)
            # Get corresponding points and colors based on the index
            queried_pc_xyz = points[queried_idx]
            queried_pc_xyz = queried_pc_xyz - pick_point
            queried_pc_colors = self.input_colors[split][cloud_idx][queried_idx]
            queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx]
            queried_pc_masks = self.input_masks[split][cloud_idx][queried_idx]

            # Update the possibility of the selected points
            dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
            delta = np.square(1 - dists / np.max(dists))
            self.possibility[split][cloud_idx][queried_idx] += delta
            self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

            # up_sampled with replacement
            if len(points) < cfg.num_points:
                queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels, queried_pc_masks = \
                    DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points, queried_pc_masks)

            queried_pc_xyz_list.append(queried_pc_xyz.astype(np.float32))
            queried_pc_colors_list.append(queried_pc_colors.astype(np.float32))
            queried_pc_labels_list.append(queried_pc_labels)
            queried_pc_masks_list.append(queried_pc_masks)
            queried_idx_list.append(queried_idx.astype(np.int32))
            cloud_idx_list.append(np.array([cloud_idx]).astype(np.int32))
        
        queried_pc_xyz_list = np.stack(queried_pc_xyz_list, 0)
        queried_pc_colors_list = np.stack(queried_pc_colors_list, 0)
        queried_pc_labels_list = np.stack(queried_pc_labels_list, 0)
        queried_pc_masks_list = np.stack(queried_pc_masks_list, 0)
        queried_idx_list = np.stack(queried_idx_list, 0)
        cloud_idx_list = np.stack(cloud_idx_list, 0)
        
        return queried_pc_xyz_list, queried_pc_colors_list, queried_pc_labels_list, queried_pc_masks_list,\
            queried_idx_list, cloud_idx_list
    
        
    def inference(self, inputs, is_training):

        d_out = self.config.d_out
        feature = inputs['features']
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)

        # ###########################Encoder############################
#         pdb.set_trace()
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i],
                                                 'Encoder_layer_' + str(i), is_training)
            f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i])
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                        'decoder_0',
                                        [1, 1], 'VALID', True, is_training)

        # ###########################Decoder############################
#         pdb.set_trace()
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(feature, inputs['interp_idx'][-j - 1])
            f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([f_encoder_list[-j - 2], f_interp_i], axis=3),
                                                          f_encoder_list[-j - 2].get_shape()[-1].value, [1, 1],
                                                          'Decoder_layer_' + str(j), [1, 1], 'VALID', bn=True,
                                                          is_training=is_training)
            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        f_layer_fc1 = helper_tf_util.conv2d(f_decoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        f_out = tf.squeeze(f_layer_fc3, [2])
        return f_out

    def train(self, dataset):
        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
#         self.sess.run(dataset.train_init_op)
        while self.training_epoch < self.config.max_epoch:
            self.train_one_epoch_full()
            m_iou = self.evaluate_one_epoch(dataset)
            if m_iou > np.max(self.mIou_list):
                # Save the best model
                snapshot_directory = join(self.saving_path, 'snapshots')
                makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
            self.mIou_list.append(m_iou)
            log_out('Best m_IoU is: {:5.3f}'.format(max(self.mIou_list)), self.Log_file)

            # Update learning rate
            op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                       self.config.lr_decays[self.training_epoch]))
            self.sess.run(op)
            log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

        print('finished')
        self.sess.close()
    
    def train_one_epoch_full(self, split='training'):
        
        cfg = self.config
        if split == 'training':
            num_batches = cfg.train_steps
        elif split == 'validation':
            num_batches = cfg.val_steps
        
        for batch_idx in range(num_batches):

            batch_xyz, batch_features, batch_labels, batch_masks, batch_pc_idx, batch_cloud_idx = self.get_batch_data(split)
            
            #siamese inputs
            data_feed = []
            feature_feed = []
            batch_size = batch_xyz.shape[0]
            
            if self.training_epoch >= self.rampup:
                
                for i in range(batch_size):
                    data_0, feat_0 = batch_xyz[i,:,:], batch_features[i,:,:]
                    data_feed.append(data_0)
                    feature_feed.append(feat_0)
                    data_i = data_0.copy()

                    # random data augmentation by flip x , y or x+y
                    flip_aug = True
                    if flip_aug:
                        flip_type = np.random.choice(4, 1)
                        if flip_type == 1:
                            data_i[:, 0] = -data_i[:, 0]
                        elif flip_type == 2:
                            data_i[:, 1] = -data_i[:, 1]
                        elif flip_type == 3:
                            data_i[:, :2] = -data_i[:, :2]

                    ## Randomly Rotate the Shape in XoY plane
                    theta = np.random.uniform(0, 2*np.pi)   # from 0 to 2pi
                    # R = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])#绕y旋转
                    R = np.array(
                        [[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])  # 绕z旋转
                    data_i[:,0:3] = np.einsum('jk,lk->lj', R, data_i[:,0:3])
                    ## Append Augmented Sample
                    data_feed.append(data_i)
                    feature_feed.append(feat_0)
            else:
                for i in range(batch_size):
                    data_0, feat_0 = batch_xyz[i,:,:], batch_features[i,:,:]
                    data_feed.append(data_0)
                    data_feed.append(data_0)
                    feature_feed.append(feat_0)
                    feature_feed.append(feat_0)
            
            batch_xyz = np.stack(data_feed, axis=0)
            batch_features = np.stack(feature_feed, axis=0)
            batch_features = np.concatenate([batch_xyz, batch_features], axis=-1)
            
            ## Prepare Labels
            seg_feed=[]
            for i in range(batch_size):
                seg_feed.append(batch_labels[i,:])
                seg_feed.append(batch_labels[i,:])
            batch_labels = np.stack(seg_feed)
            
            #### Prepare Incomplete Labelled Training Data
            Mask_bin_feed = []
            for i in range(batch_size):
                Mask_bin_feed.append(batch_masks[i,:])
                Mask_bin_feed.append(batch_masks[i,:])
            batch_masks = np.stack(Mask_bin_feed)
            
            
            pcidx_feed = []
            for i in range(batch_size):
                pcidx_feed.append(batch_pc_idx[i,:])
                pcidx_feed.append(batch_pc_idx[i,:])
            batch_pc_idx = np.stack(pcidx_feed)
            
            cidx_feed = []
            for i in range(batch_size):
                cidx_feed.append(batch_cloud_idx[i])
                cidx_feed.append(batch_cloud_idx[i])
            batch_cloud_idx = np.stack(cidx_feed)
            
            ####finish preparation
#             pdb.set_trace()
            feed_dict = {self.inputs['features']: batch_features,
                         self.inputs['labels']: batch_labels,
                         self.inputs['mask_boolen']: batch_masks,
                         self.inputs['input_inds']: batch_pc_idx,
                         self.inputs['cloud_inds']: batch_cloud_idx,
                         self.is_training: True}
            
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []
            
            for i in range(cfg.num_layers):
                neighbour_idx = DP.knn_search(batch_xyz, batch_xyz, cfg.k_n) #get neighbour index for all points
                sub_points = batch_xyz[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :] #fetch the first N/4 points, the index has been shuffed, no need to do random sampling
                pool_i = neighbour_idx[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]
                up_i = DP.knn_search(sub_points, batch_xyz, 1) #get upsample index
                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
#                 print('neighbour_idx', neighbour_idx.shape)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points
                
#             pdb.set_trace()
            for i in range(cfg.num_layers):
                feed_dict.update({
                    self.inputs['xyz'][i] : input_points[i],
                    self.inputs['neigh_idx'][i] : input_neighbors[i],
                    self.inputs['sub_idx'][i] : input_pools[i],
                    self.inputs['interp_idx'][i] : input_up_samples[i],
                })
                
            t_start = time.time()    
            _, _, summary, l_out, probs, labels, _, acc = self.sess.run(self.ops, feed_dict=feed_dict)
            self.train_writer.add_summary(summary, self.training_step)
            t_end = time.time()
            if self.training_step % 50 == 0:
                message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
            self.training_step += 1

        self.training_epoch += 1

        
    def evaluate_one_epoch(self, dataset):

        # Initialise iterator with validation data
#         self.sess.run(dataset.val_init_op)

        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0
        
        cfg = self.config

        for step_id in range(self.config.val_steps):
            
            if step_id % 50 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                
                batch_xyz, batch_features, batch_labels, batch_masks, batch_pc_idx, batch_cloud_idx = self.get_batch_data('validation')
                batch_features = np.concatenate([batch_xyz, batch_features], axis=-1)
                
                feed_dict = {self.inputs['features']: batch_features,
                         self.inputs['labels']: batch_labels,
                         self.inputs['mask_boolen']: batch_masks,
                         self.inputs['input_inds']: batch_pc_idx,
                         self.inputs['cloud_inds']: batch_cloud_idx,
                         self.is_training: False}
            
                input_points = []
                input_neighbors = []
                input_pools = []
                input_up_samples = []

                for i in range(cfg.num_layers):
                    neighbour_idx = DP.knn_search(batch_xyz, batch_xyz, cfg.k_n) #get neighbour index for all points
                    sub_points = batch_xyz[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :] #fetch the first N/4 points, the index has been shuffed, no need to do random sampling
                    pool_i = neighbour_idx[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]
                    up_i = DP.knn_search(sub_points, batch_xyz, 1) #get upsample index
                    input_points.append(batch_xyz)
                    input_neighbors.append(neighbour_idx)
    #                 print('neighbour_idx', neighbour_idx.shape)
                    input_pools.append(pool_i)
                    input_up_samples.append(up_i)
                    batch_xyz = sub_points

                for i in range(cfg.num_layers):
                    feed_dict.update({
                        self.inputs['xyz'][i] : input_points[i],
                        self.inputs['neigh_idx'][i] : input_neighbors[i],
                        self.inputs['sub_idx'][i] : input_pools[i],
                        self.inputs['interp_idx'][i] : input_up_samples[i],
                    })

                ops = (self.merged, self.prob_logits, self.labels, self.accuracy)
                summary, stacked_prob, labels, acc = self.sess.run(ops, feed_dict=feed_dict)
                self.test_writer.add_summary(summary, self.eval_step)
                self.eval_step += 1
                pred = np.argmax(stacked_prob, 1)
                if not self.config.ignored_label_inds:
                    pred_valid = pred
                    labels_valid = labels
                else:
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    labels_valid = labels_valid - 1
                    pred_valid = np.delete(pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)

            except tf.errors.OutOfRangeError:
                break

        iou_list = []
        f1_list = []
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)
            f1 = 2*true_positive_classes[n] / float(gt_classes[n] + positive_classes[n])
            f1_list.append(f1)
        mean_iou = sum(iou_list) / float(self.config.num_classes)

        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
        log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)
        
        mean_f1 = sum(f1_list) / float(self.config.num_classes)
        mean_f1 = 100 * mean_f1
        log_out('Avg F1 = {:.1f}%'.format(mean_f1), self.Log_file)
        s = '{:5.2f} | '.format(mean_f1)
        for f1 in f1_list:
            s += '{:5.2f} '.format(100 * f1)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)
        
        return mean_f1
    
    def WeakSupLoss(self):
        '''
        Define additional losses for weakly supervised segmentation
        Inputs:
            self.logits = tf.reshape(self.logits, [-1, config.num_classes])
            self.labels = tf.reshape(self.labels, [-1])
            self.mask_boolen = tf.reshape(self.mask_boolen, [-1])
        Returns:

        '''
        
        ## MIL Branch
        one_hot_labels = tf.one_hot(self.batch_labels, depth=self.config.num_classes, axis=-1)
#         one_hot_labels = tf.reshape(one_hot_labels, [self.inputs.shape[0],self.inputs.shape[1],self.config.num_classes])
        L_gt = tf.cast(tf.reduce_max(one_hot_labels[0::2], axis=1), tf.float32)  # inexact labels B*13
        L_pred = tf.reduce_max(self.batch_logits[0::2], axis=1)
        loss_ineaxct = tf.nn.sigmoid_cross_entropy_with_logits(labels=L_gt, logits=L_pred)  # B*K, independent loss for each class
        self.loss_inexact = 1* tf.reduce_mean(loss_ineaxct)
        
        ## Siamese Branch
        self.prob_logits = tf.nn.softmax(self.batch_logits, axis=-1)
        self.loss_siamese = 1* tf.reduce_mean(tf.reduce_sum((self.prob_logits[0::2] - self.prob_logits[1::2]) ** 2, axis=-1))
        
#         pdb.set_trace()
        ## Smooth Branch
#         self.loss_smooth = 10* SmoothConstraint.Loss_SpatialColorSmooth_add_SelfContain(self.prob_logits[0::2], self.inputs['features'][0::2, :, 0:3], M=self.batch_masks[0::2], Y=self.batch_labels[0::2], gamma=1e-1)
        self.loss_smooth = 10 * SmoothConstraint.Loss_SpatialColorSmooth_add_SelfContain_mustLink(self.prob_logits[0::2], self.inputs['features'][0::2, :, 0:6], M=self.batch_masks[0::2], Y=self.batch_labels[0::2], gamma=1e-1)#0:5

#         self.loss_smooth = 0 
    

    def get_loss(self, logits, labels, pre_cal_weights, mask_boolen=None):
#         pdb.set_trace()
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        
        if mask_boolen is not None:
            weighted_losses = tf.reduce_sum(tf.cast(mask_boolen, tf.float32) * weighted_losses) / tf.reduce_sum(tf.cast(mask_boolen, tf.float32))
            output_loss = weighted_losses + \
                                tf.cast(tf.greater_equal(self.training_epoch, self.rampup), dtype=tf.float32) * (
                                            self.loss_siamese + self.loss_inexact + self.loss_smooth)
        else:
            output_loss = tf.reduce_mean(weighted_losses)
        return output_loss

    def dilated_res_block(self, feature, xyz, neigh_idx, d_out, name, is_training):
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)
        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None)
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)

    def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)

        f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features

    @staticmethod
    def att_pooling(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
        att_scores = tf.nn.softmax(att_activation, axis=1)
        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return f_agg
