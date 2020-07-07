import numpy as np
import os
from easydict import EasyDict as edict

config = edict()

config.bn_mom = 0.9
config.workspace = 256
config.emb_size = 512
config.ckpt_embedding = True
config.net_se = 0
config.net_act = 'prelu'
config.net_unit = 3
config.net_input = 1
config.net_blocks = [1,4,6,2]
config.net_output = 'E'
config.net_multiplier = 1.0
config.val_targets = ['lfw']
config.ce_loss = True
config.fc7_lr_mult = 1.0
config.fc7_wd_mult = 1.0
config.fc7_no_bias = False
config.max_steps = 0
config.data_rand_mirror = True
config.data_cutoff = False
config.data_color = 0
config.data_images_filter = 0
config.count_flops = True
config.memonger = False #not work now


# network settings
network = edict()

network.r100 = edict()
network.r100.net_name = 'fresnet'
network.r100.num_layers = 100


# dataset settings
dataset = edict()



dataset.ms1m = edict()
dataset.ms1m.dataset = 'ms1m'
dataset.ms1m.dataset_path = '../datasets/ms1mv3_clean'
dataset.ms1m.num_classes = 87639
dataset.ms1m.image_shape = (112,96,1)
dataset.ms1m.val_targets = ['lfw']

dataset.IJBC_ORI = edict()
dataset.IJBC_ORI.dataset = 'IJBC_ORI'
dataset.IJBC_ORI.dataset_path = '../datasets/IJBC-CROP-RESIZED'
dataset.IJBC_ORI.num_classes = 3531
dataset.IJBC_ORI.image_shape = (112,96,1)
dataset.IJBC_ORI.val_targets = ['lfw']

dataset.IJBC_RETINA = edict()
dataset.IJBC_RETINA.dataset = 'IJBC_ALIGNED'
dataset.IJBC_RETINA.dataset_path = '../datasets/IJBC-CROP-RETINA_rec'
dataset.IJBC_RETINA.num_classes = 3531
dataset.IJBC_RETINA.image_shape = (112,96,1)
dataset.IJBC_RETINA.val_targets = ['lfw']



loss = edict()

loss.arcface = edict()
loss.arcface.loss_name = 'margin_softmax'
loss.arcface.loss_s = 64.0
loss.arcface.loss_m1 = 1.0
loss.arcface.loss_m2 = 0.7
loss.arcface.loss_m3 = 0.0


# default settings
default = edict()

# default network
default.network = 'r100'
default.pretrained = ''
default.pretrained_epoch = 1
# default dataset
default.dataset = 'emore'
default.loss = 'arcface'
default.frequent = 20
default.verbose = 1000
default.kvstore = 'device'

default.end_epoch = 10000
default.lr = 0.1
default.wd = 0.00005
default.mom = 0.9
default.per_batch_size = 128
default.ckpt = 2
default.lr_steps = '10000,20000,30000'
default.models_root = './models'


def generate_config(_network, _dataset, _loss):
    for k, v in loss[_loss].items():
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in network[_network].items():
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in dataset[_dataset].items():
      config[k] = v
      if k in default:
        default[k] = v
    config.loss = _loss
    config.network = _network
    config.dataset = _dataset
    config.num_workers = 1
    if 'DMLC_NUM_WORKER' in os.environ:
      config.num_workers = int(os.environ['DMLC_NUM_WORKER'])

