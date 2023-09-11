from __future__ import print_function
# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
# sys.path.append('/home/lgj/research/month_project/HRDFuse')
import argparse
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys
import random
import torch
import torch.nn as nn

import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import math
from metrics import *
from tqdm import tqdm
# from data_loader_matterport3d import Dataset
from data_loader_stanford import Dataset
import cv2
import supervision as L
import spherical as S360
from sync_batchnorm import convert_model
import matplotlib.pyplot as plot
# from model.HRDFuse import hrdfuse
from model.HRDfuse import hrdfuse
# import model.HRDFuse.hrdfuse
from ply import write_ply
import csv
from util import *
import shutil
import torchvision.utils as vutils
# from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser(description='360Transformer')
parser.add_argument('--input_dir', default='/home/lgj/lgj/360/dataset360/standford.new/',
                    # parser.add_argument('--input_dir', default='/home/rtx2/NeurIPS/spherical_mvs/data/omnidepth',
                    # parser.add_argument('--input_dir', default='/media/rtx2/DATA/Structured3D/',
                    help='input data directory')
parser.add_argument('--trainfile', default='./standford2d3d_train.txt',
                    help='train file name')
parser.add_argument('--testfile', default='./standford2d3d_test.txt',
                    help='validation file name')
parser.add_argument('--valfile', default='./standford2d3d_val.txt',
                    help='validation file name')
parser.add_argument('--epochs', type=int, default=80,
                    help='number of epochs to train')
parser.add_argument('--batch', type=int, default=6,
                    help='number of batch to train')
parser.add_argument('--visualize_interval', type=int, default=20,
                    help='number of batch to train')
parser.add_argument('--patchsize', type=list, default=(256, 256),
                    help='patch size')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--fov', type=float, default=80,
                    help='field of view')
parser.add_argument('--nrows', type=int, default=4,
                    help='number of rows, options are 3, 4, 5, 6')
parser.add_argument('--confidence', action='store_true', default=True,
                    help='use confidence map or not')
parser.add_argument('--checkpoint', default=None,
                    help='load checkpoint path')
parser.add_argument('--save_checkpoint', default='checkpoints',
                    help='save checkpoint path')
parser.add_argument('--save_path', default='results/matterport/hrdfuse_256_80',
                    help='save checkpoint path')
parser.add_argument('--tensorboard_path', default='logs',
                    help='tensorboard path')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--min_val', type=float, default=0.1,
                    help='number of batch to train')
parser.add_argument('--max_val', type=float, default=10.0,
                    help='number of batch to train')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Save Checkpoint -------------------------------------------------------------
if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
else:
    shutil.rmtree(args.save_path)
if not os.path.isdir(os.path.join(args.save_path, args.save_checkpoint)):
    os.makedirs(os.path.join(args.save_path, args.save_checkpoint))

# result visualize Path -----------------------
writer_path = os.path.join(args.save_path, args.tensorboard_path)
image_path = os.path.join(args.save_path, "image")
if not os.path.isdir(writer_path):
    os.makedirs(writer_path)
if not os.path.isdir(image_path):
    os.makedirs(image_path)
# writer = SummaryWriter(log_dir=writer_path)

result_view_dir = args.save_path
# shutil.copy('train_fusion.py', result_view_dir)
# shutil.copy('model/spherical_fusion.py', result_view_dir)
# shutil.copy('model/ViT/miniViT.py', result_view_dir)
# shutil.copy('model/ViT/layers.py', result_view_dir)

# Random Seed -----------------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# ------------------------------------------tensorboard_pathf training files
input_dir = args.input_dir
train_file_list = args.trainfile
val_file_list = args.valfile  # File with list of validation files
# ------------------------------------
# -------------------------------------------------------------------
batch_size = 2
visualize_interval = args.visualize_interval
init_lr = args.lr
fov = (args.fov, args.fov)  # (48, 48)
patch_size = args.patchsize
nrows = args.nrows
npatches_dict = {3: 10, 4: 18, 5: 26, 6: 46}
min_val=args.min_val
max_val=args.max_val
# -------------------------------------------------------------------
# data loaders
train_dataloader = torch.utils.data.DataLoader(
    dataset=Dataset(
        rotate=True,
        flip=True,
        root_path=input_dir,
        path_to_img_list=train_file_list),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True)

val_dataloader = torch.utils.data.DataLoader(
    dataset=Dataset(
        root_path=input_dir,
        path_to_img_list=val_file_list),
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    drop_last=False)

# ----------------------------------------------------------
# first network, coarse depth estimation
# option 1, resnet 360
num_gpu = torch.cuda.device_count()
import ipdb
ipdb.set_trace()
model_dict = torch.load('./checkpoint_best.tar')
network = hrdfuse(nrows=nrows, npatches=npatches_dict[nrows], patch_size=patch_size, fov=fov, min_val=min_val, max_val=max_val)



path = './checkpoint_best.tar'
model_dict = network.state_dict()
pretrained_dict = torch.load(path)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
network.load_state_dict(model_dict)




# network.load_state_dict(model_dict)
print('test')
























