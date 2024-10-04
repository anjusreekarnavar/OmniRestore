
/*
 * Copyright (c) 2024 ---Anjusree Karnavar,Griffith University
 */
import argparse
import datetime
import json
import numpy as np
import os
from torchvision.utils import save_image
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.init as init
import sys
from metrics_eval import AverageMeter,Conversion
from metrics_eval import compute_psnr_ssim
from augmentations import converto_low_resolution,blur_input_image
from pathlib import Path
import PIL
from create_experts2 import create_experts_restoration
import torch
import torch.backends.cudnn as cudnn
from aggregator import MOE,DepthCNN
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import  DataLoader
from PIL import Image, ImageFilter,ImageOps
import timm
from callback import EarlyStopping
from torch.utils.data import  DataLoader, random_split
from decoder import Decoder1,Decoder2,Decoder3,Decoder4,Decoder5
assert timm.__version__ == "0.5.4"  # version check
from custom_dataset import ImageRestorationDataset,DataLoaderVal
from util import misc
import model_restoration
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from stage3_engineddp import train_one_epoch
from multidecoders import ImageRestoration
import timm.optim.optim_factory as optim_factory
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from earlystopping import EarlyStopping
def get_args_parser():
    parser = argparse.ArgumentParser('restoreMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
  
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

   

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--sigma', default=0.25, type=float,
                        help='Std of Gaussian noise')
    parser.add_argument('--radius', default=1, type=int,
                        help='blurring radius')
    parser.add_argument('--downsampling_factor', default=4, type=int,
                        help='downsampling')
    parser.add_argument('--mask_ratio', default=0.75, type=int,
                        help='masking')
    
    #labels for checking threshold
    parser.add_argument('--denoise_flag',type=int,default=0,help='value will be equal to 1 if loss than 0')
    parser.add_argument('--blur_flag',type=int,default=0,help='value will be equal to 1 if loss than 0')
    parser.add_argument('--super_flag',type=int,default=0,help='value will be equal to 1 if loss than 0')
    parser.add_argument('--mask_flag',type=int,default=0,help='value will be equal to 1 if loss than 0')
    parser.add_argument('--inpaint_flag',type=int,default=0,help='value will be equal to 1 if loss than 0')
    parser.add_argument('--mask_shape',type=str,default='ellipse',help='for inpaint mask type')
    parser.add_argument('--percentage',type=int,default=0.25,help='for percentage to mask')
    parser.add_argument('--max_vertices',type=int,default=10,help='maximum vertices for irregular mask')
    parser.add_argument('--mask_radius',type=int,default=5,help='radius for mask')
    parser.add_argument('--num_lines',type=int,default=10,help='lines for mask')
    parser.add_argument('--num_patches',type=int,default=10,help='number of patches in the mask')
    parser.add_argument('--patch_size',type=int,default=16,help='size of each patch')
    parser.add_argument('--aggregator',type=int,default=1,help='aggregator selection 0 for gating and 1 for DCNN')


     #validation flags
    parser.add_argument('--denoise_valflag',type=int,default=0,help='value will be equal to 1 if loss than 0')
    parser.add_argument('--blur_valflag',type=int,default=0,help='value will be equal to 1 if loss than 0')
    parser.add_argument('--super_valflag',type=int,default=0,help='value will be equal to 1 if loss than 0')
    parser.add_argument('--mask_valflag',type=int,default=0,help='value will be equal to 1 if loss than 0')
    parser.add_argument('--inpaint_valflag',type=int,default=0,help='value will be equal to 1 if loss than 0')
    # Optimizer parameters
    parser.add_argument('--test_flag',type=int,default=0,help='value will be equal to 1 if loss than 0')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')

    parser.add_argument('--output_dir', default='/scratch3/ven073/dcnn_outputddp',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/scratch3/ven073/dcnn_outputddp/log_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--num_experts', default=5, type=int, metavar='N',
                        help='number of experts')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def kaiming_init_weights(m):
    if isinstance (m, (nn.Linear)) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight) 


def main(args):
    misc.init_distributed_mode(args)  #distributed training

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
  
    train_dir='/scratch3/ven073/datanew/train2'
    val_dir='/scratch3/ven073/datanew/val2'
    cudnn.benchmark = True

    # simple augmentation
    # in order to add noise, the normalization is done in the dmae model

    # built in  function for transformations

 
    # load data 
    #dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_train=DataLoaderVal(train_dir)
    dataset_val=DataLoaderVal(val_dir)
    #dataset_train=transform(dataset_train)
    #dataset_val=transform(dataset_val)
    #train_size = int(0.8 * len(dataset_train))  # 80% of the dataset for training
    #test_size = len(dataset_train) - train_size   
    #train_data, test_data = random_split(dataset_train, [train_size, test_size])   
    #dataset_train=DataLoaderVal(input_dir,transform)
    #dataset_val=DataLoaderVal(val_dir,transform)
    print('training data size',len(dataset_train)) 
    print('validation data size',len(dataset_val)) 


    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
            
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # data loader for loading training data
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

     # data loader for loading validation data
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    log_writer = SummaryWriter(log_dir=args.log_dir)

    early_stopping = EarlyStopping(patience=10, verbose=True)      
    #load the pre encoder
    print('cuda availability', torch.cuda.is_available())
  
    best_psnr=0
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    experts,shared_encoder=create_experts_restoration(args)
 

    depth_cnn=DepthCNN()
    depth_cnn= depth_cnn.to(args.device)
    #depth_cnn.apply(kaiming_init_weights)

    optimizer2 = torch.optim.Adam(depth_cnn.parameters(), lr=0.00001,betas=(0.9, 0.999))
    scheduler2=optim.lr_scheduler.ReduceLROnPlateau(optimizer2,factor=0.1,patience=5,verbose=True)
    if args.distributed:
        depth_cnn = torch.nn.parallel.DistributedDataParallel(depth_cnn, device_ids=[args.gpu], find_unused_parameters=True)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        
        trained_model,psnr_val = train_one_epoch(
            shared_encoder,depth_cnn,data_loader_train,data_loader_val,optimizer2,scheduler2,experts,device,
            epoch,log_writer,best_psnr,early_stopping,args)
         best_psnr=psnr_val

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
