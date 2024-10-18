import argparse
import datetime
import json
import numpy as np
import os
from torchvision.utils import save_image
import torch.nn as nn
from multi_dataloading import create_dataset_train,create_dataset_val
import time
import sys
from metrics_eval import Conversion
from pathlib import Path
import PIL
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import  DataLoader
from PIL import Image, ImageFilter,ImageOps
import timm
from torch.utils.data import  DataLoader, random_split
assert timm.__version__ == "0.5.4"  # version check
from decoder import Decoder
from util import misc
import yaml
import model_restoration
import options as option
from tensorboardX import SummaryWriter
from shared_encoder_engine import train_one_epoch
import timm.optim.optim_factory as optim_factory
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from multidecoders import MultiImageRestoration
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

    parser.add_argument('--mask_ratio', default=0.75, type=int,
                        help='masking')
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

    parser.add_argument('--decoder_depth',  default=10, type=int,
                        help='dataset path')
    parser.add_argument('--decoder_depth2',  default=16, type=int,
                        help='dataset path')

    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
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
def load_decoders(args):
    expert1=Decoder(decoder_depth=args.decoder_depth)
    expert1=loading_checkpoint(expert1)
    expert2=Decoder(decoder_depth=args.decoder_depth2)
    expert2=loading_checkpoint(expert2)
    return expert1,expert2

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

  
    with open('/home/ven073/anju/dmae2/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    data_loader_train=create_dataset_train(config,args)
      
    data_loader_val=create_dataset_val(config,args)

    
    
    log_writer = SummaryWriter(log_dir=args.log_dir)
    
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
    
    
    print('cuda availability', torch.cuda.is_available())
    decoder1,decoder2=load_decoders(args)
    
    model=MultiImageRestoration(shared_encoder,decoder1,decoder2)
    model.to(device)
    
    tasks=['denoising','deblurring','super_resolution','inpainting','demasking']
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            for _, data_loader_train_ in data_loader_train.items():
                data_loader_train_.sampler.set_epoch(epoch)
        model_trained= train_one_epoch(model,data_loader_train,data_loader_val,tasks,device,optimizer,loss_scaler,epoch,log_writer,args)
            
        if args.output_dir and (epoch % 100 == 0 or epoch + 1 == args.epochs):
           
                torch.save({
                'model_state_dict': model_trained.module.encoder.state_dict(),
           
                }, f'{args.output_dir}/shared_encoder_{epoch+1}.pth')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
