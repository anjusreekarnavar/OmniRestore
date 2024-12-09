import argparse
import datetime
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.init as init
import sys
from pathlib import Path
import PIL
import torch
import torch.backends.cudnn as cudnn
from MoEwithtransformer import MOE
import torchvision.transforms as transforms
import timm
from custom_train_validset import DataLoaderTrain,DataLoaderVal
from create_experts_freezed import create_experts_restoration
from callback import EarlyStopping
assert timm.__version__ == "0.5.4"  # version check
from util import misc
from tensorboardX import SummaryWriter
from engine_moe2 import aggregator_train
import timm.optim.optim_factory as optim_factory
from util.misc import NativeScalerMoE as NativeScaler
from earlystopping import EarlyStopping
def get_args_parser():
    parser = argparse.ArgumentParser('restoreMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
  
    parser.add_argument('--epochs', default=500, type=int)
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
    parser.add_argument('--mask_ratio', default=0.20, type=int,
                        help='masking')
    
    #labels for checking threshold
    parser.add_argument('--sigma', default=0.1, type=float,
                        help='Std of Gaussian noise')
    parser.add_argument('--radius', default=1, type=int,
                        help='blurring radius')
    parser.add_argument('--downsampling_factor', default=4, type=int,
                        help='downsampling')
    parser.add_argument('--mask_root', default='/scratch3/ven073/datanew/data', type=str,
                        help='mask path')
    parser.add_argument('--mask_type', default='thin', type=str,
                        help='thin,thick,nn2,genhalf,ex64,ev2li')
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.08,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    # Dataset parameters
    parser.add_argument('--train_dir', default='', type=str,
                        help='dataset path')
    parser.add_argument('--val_dir', default='', type=str,
                        help='dataset path')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--freeze', default=False, type=str,
                        help='freezing expert and encoder')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='',
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


def main(args):

    misc.init_distributed_mode(args)  #distributed training

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cudnn.benchmark = True
    transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
            ])
    transform_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            ])

    dataset_train=DataLoaderTrain(args.train_dir,transform_train)
    dataset_val=DataLoaderVal(args.val_dir,transform_val)
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
        dataset_val,batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    log_writer = SummaryWriter(log_dir=args.log_dir)
    eval_now = len(data_loader_train)//4
    #load the pre encoder
    print('cuda availability', torch.cuda.is_available())

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    #load experts
    
    moe=MOE(args)
    moe.to(args.device)  
    
    start_epoch=0
    loss_scaler = NativeScaler()
    best_psnr=0

    early_stop = EarlyStopping(patience=30, delta=0.01) 
    
    if args.distributed:
        moe = torch.nn.parallel.DistributedDataParallel(moe, device_ids=[args.gpu], find_unused_parameters=True)
    #param_groups = optim_factory.add_weight_decay(moe.module.gating_network, args.weight_decay)
    #optimizer = torch.optim.AdamW(param_groups, lr=1e-5, betas=(0.9, 0.999))#1.5e-6
    if args.freeze:
        optimizer = optim.Adam(moe.module.parameters(), lr=1e-5, betas=(0.9, 0.999),eps=1e-5, weight_decay=args.weight_decay)#1e-5
    else:
        encoder_params = list(moe.module.shared_encoder.parameters())
        gating_params = list(moe.module.gating_network.parameters())
        expert_params = [param for expert in moe.module.trained_experts for param in expert.parameters()]
        optimizer = torch.optim.AdamW([
                {'params': encoder_params, 'lr': 1e-4,'weight_decay': 0.02},  
                {'params': gating_params, 'lr': 1e-5,'weight_decay': 0.05} ,
                {'params': expert_params, 'lr': 1e-5,'weight_decay': 0.02}  
            ], betas=(0.9, 0.999))
    #scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1,patience=5,verbose=True)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-start_epoch+1, eta_min=1e-9) 
 
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        
        psnr_val,val_loss=aggregator_train(data_loader_train,data_loader_val,optimizer,moe,device,
                     loss_scaler,epoch,log_writer,best_psnr,eval_now,scheduler,args)
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
