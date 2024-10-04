import argparse
import datetime
import json
import numpy as np
import os
from torchvision.utils import save_image
import torch.nn as nn

import time
import sys
from metrics_eval import AverageMeter,Conversion
from metrics_eval import compute_psnr_ssim
from augmentations import converto_low_resolution,blur_input_image
from pathlib import Path
import PIL
import torch
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import  DataLoader
from PIL import Image, ImageFilter,ImageOps
import timm
from callback import EarlyStopping
from torch.utils.data import  DataLoader, random_split
from decoder import Decoder1,Decoder2,Decoder3,Decoder4,Decoder5
from new_decoders import Denoise_Expert,Super_Expert,Deblur_Expert,Inpaint_Expert,Demask_Expert
assert timm.__version__ == "0.5.4"  # version check
from decoder import Decoder1,Decoder2,Decoder3,Decoder4,Decoder5
from util import misc
import model_restoration
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from main_engine import train_one_epoch
import timm.optim.optim_factory as optim_factory
from util.misc import NativeScalerWithGradNormCount as NativeScaler

def get_args_parser():
    parser = argparse.ArgumentParser('restoreMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
  
    parser.add_argument('--epochs', default=1600, type=int)
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
    parser.add_argument('--slr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blur', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--mlr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--ilr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')#previous value 1e-3
    parser.add_argument('--blr2', type=float, default=1.5e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--bslr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')#previous value 1.5e-4
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/scratch3/ven073/datanew', type=str,
                        help='dataset path')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')

    parser.add_argument('--output_dir', default='/scratch3/ven073/base_output3',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/scratch3/ven073/base_output3/log_dir',
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



def loading_checkpoint(model):

    path='/home/ven073/anju/dmae2/dmae_base_sigma_0.25_mask_0.75_1100e.pth'
    
    pretrained_path = torch.load(path)


    new_model_dict = model.state_dict()

    pretrained_weights = { k:v for k , v in pretrained_path.items() if k in new_model_dict}

    new_model_dict.update(pretrained_weights)

    model.load_state_dict(pretrained_weights, strict = False)
    return model
def load_decoders(device):
    denoise_expert=Decoder1()
    denoise_expert=loading_checkpoint(denoise_expert)
    denoise_expert = denoise_expert.to(device, non_blocking=True)
    deblur_expert=Decoder3()
    deblur_expert=loading_checkpoint(deblur_expert)
    deblur_expert = deblur_expert.to(device, non_blocking=True)
    superr_expert=Decoder2()
    superr_expert=loading_checkpoint(superr_expert)
    superr_expert = superr_expert.to(device, non_blocking=True)
    masking_expert=Decoder5()
    masking_expert=loading_checkpoint(masking_expert)
    masking_expert = masking_expert.to(device, non_blocking=True)
    inpaint_expert=Decoder4()
    inpaint_expert=loading_checkpoint(inpaint_expert)
    inpaint_expert = inpaint_expert.to(device, non_blocking=True)
    return denoise_expert,superr_expert,deblur_expert,masking_expert,inpaint_expert

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
  

    cudnn.benchmark = True

    # simple augmentation
    # in order to add noise, the normalization is done in the dmae model
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
   
    dataset_train= datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_val=datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_train)
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

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    early_stopping_denoise = EarlyStopping()
    early_stopping_deblur = EarlyStopping()
    early_stopping_demask = EarlyStopping()
    early_stopping_super = EarlyStopping()
    early_stopping_inpaint = EarlyStopping()

              
    #load the pre trained model
    print('cuda availability', torch.cuda.is_available())
    # define the model
    path='/home/ven073/anju/dmae2/dmae_base_sigma_0.25_mask_0.75_1100e.pth'
    shared_encoder = model_restoration.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    pretrained_path = torch.load(path)


    new_model_dict = shared_encoder.state_dict()

    pretrained_weights = { k:v for k , v in pretrained_path.items() if k in new_model_dict}

    new_model_dict.update(pretrained_weights)

    shared_encoder.load_state_dict(pretrained_weights, strict = False)
    shared_encoder=shared_encoder.to(device)
 
    print('cuda availability', torch.cuda.is_available())



    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    #misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    denoise_expert,superr_expert,deblur_expert,inpaint_expert,masking_expert=load_decoders(device)
  
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr2 * eff_batch_size / 256
    param_groups = optim_factory.add_weight_decay(denoise_expert, args.weight_decay)
    optimizer1 = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    decoder1=denoise_expert.to(device)
 
    if args.slr is None:  # only base_lr is specified
        args.slr = args.bslr * eff_batch_size / 256
    param_groups = optim_factory.add_weight_decay(superr_expert, args.weight_decay)
    optimizer2 = torch.optim.AdamW(param_groups, lr=args.slr, betas=(0.9, 0.95))
    decoder2=superr_expert.to(device)
   
    if args.blur is None:  # only base_lr is specified
        args.blur = args.bslr * eff_batch_size / 256
    param_groups = optim_factory.add_weight_decay(deblur_expert, args.weight_decay)
    optimizer3 = torch.optim.AdamW(param_groups, lr=args.blur, betas=(0.9, 0.95))
    decoder3=deblur_expert.to(device)
    if args.mlr is None:  # only base_lr is specified
        args.mlr = args.blr * eff_batch_size / 256
    param_groups = optim_factory.add_weight_decay(masking_expert, args.weight_decay)
    optimizer5 = torch.optim.AdamW(param_groups, lr=args.mlr, betas=(0.9, 0.95))
    decoder5=masking_expert.to(device)
    
    if args.ilr is None:  # only base_lr is specified
        args.ilr = args.blr * eff_batch_size / 256
    param_groups = optim_factory.add_weight_decay(inpaint_expert, args.weight_decay)
    optimizer4 = torch.optim.AdamW(param_groups, lr=args.ilr, betas=(0.9, 0.95))
    decoder4=inpaint_expert.to(device)
    decoders=[decoder1,decoder2,decoder3,decoder4,decoder5]
   
    if args.distributed:
        shared_encoder=torch.nn.parallel.DistributedDataParallel(shared_encoder, device_ids=[args.gpu], find_unused_parameters=True)
        deco1= torch.nn.parallel.DistributedDataParallel(decoder1, device_ids=[args.gpu], find_unused_parameters=True)
        
        deco2= torch.nn.parallel.DistributedDataParallel(decoder2, device_ids=[args.gpu], find_unused_parameters=True)
      
        deco3= torch.nn.parallel.DistributedDataParallel(decoder3, device_ids=[args.gpu], find_unused_parameters=True)
  
        deco4=torch.nn.parallel.DistributedDataParallel(decoder4, device_ids=[args.gpu], find_unused_parameters=True)

        deco5=torch.nn.parallel.DistributedDataParallel(decoder5, device_ids=[args.gpu], find_unused_parameters=True)
        decoders=[deco1,deco2,deco3,deco4,deco5]

    optimizers=[optimizer1,optimizer2,optimizer3,optimizer4,optimizer5]
    
    loss_scaler1 = NativeScaler()
    loss_scaler2 = NativeScaler()
    loss_scaler3 = NativeScaler()
    loss_scaler4 = NativeScaler()
    loss_scaler5 = NativeScaler()
    noise_flag=0
    blur_flag=0
    super_flag=0
    inpaint_flag=0
    mask_flag=0
    flags=[noise_flag,super_flag,blur_flag,inpaint_flag,mask_flag]
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        if noise_flag==0 or blur_flag==0 or super_flag==0 or inpaint_flag==0 or mask_flag==0:
            encoder,flags = train_one_epoch(shared_encoder,decoders,data_loader_train,data_loader_val,device,optimizers,loss_scaler1,loss_scaler2,loss_scaler3,loss_scaler4,loss_scaler5,epoch,flags,log_writer,args)
            
            noise_flag=flags[0]
               
            super_flag=flags[1]
                
            blur_flag=flags[2]
          
            inpaint_flag=flags[3]
                
            mask_flag=flags[4]
            flags=[noise_flag,super_flag,blur_flag,inpaint_flag,mask_flag]
        else:
            break
        torch.cuda.empty_cache()
        if args.output_dir and (epoch % 100 == 0 or epoch + 1 == args.epochs):
           
                torch.save({
                'model_state_dict': encoder.module.state_dict(),
           
                }, f'/scratch3/ven073/base_output3/shared_encoder_{epoch+1}.pth')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
