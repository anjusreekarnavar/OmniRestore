# Copyright (c) Meta Platforms, Inc. and affiliates.
#Anjusree Karnavar Griffith University 2024
#anjusree.karnavar@griffithuni.edu.au
# --------------------------------------------------------


import argparse
import datetime
import json
import numpy as np
import os
from torchvision.utils import save_image
import torch.nn as nn
from inpaint_mask_generator import generate_mask,patch_generator
import time
import sys
from metrics_eval import AverageMeter,Conversion
from metrics_eval import compute_psnr_ssim
from augmentations import converto_low_resolution,blur_input_image
from pathlib import Path
import PIL
import torch
import torch.backends.cudnn as cudnn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import  DataLoader
from PIL import Image, ImageFilter,ImageOps
import timm
from custom_testset import TestDataset
from callback import EarlyStopping
from aggregator3 import MOE
from torch.utils.data import  DataLoader, random_split
from decoder import Decoder1,Decoder2,Decoder3,Decoder4,Decoder5

from temporary import Conversion
assert timm.__version__ == "0.5.4"  # version check
from util import misc
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import model_restoration
from custom_testset import DataNoisy,DataBlurry,DataSuper,DataInpaint,DataMask
from engine2 import train_one_epoch
from multidecoders import ImageRestoration
import timm.optim.optim_factory as optim_factory
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from create_experts2 import create_experts_restoration

def get_args_parser():
    parser = argparse.ArgumentParser('restoreMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
  
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='dmae_vit_base_patch16', type=str, metavar='MODEL',
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
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/scratch3/ven073/newdataset', type=str,
                        help='dataset path')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')

    parser.add_argument('--output_dir', default='/scratch3/ven073/base_output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/scratch3/ven073/base_output/log_dir',
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

def compute_psnr_ssim(recoverd, clean):

    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)

    recoverd = recoverd.transpose(0, 2, 3, 1)
    clean = clean.transpose(0, 2, 3, 1)
    psnr=0
    ssim=0

    for i in range(recoverd.shape[0]):
        # psnr_val += compare_psnr(clean[i], recoverd[i])
        # ssim += compare_ssim(clean[i], recoverd[i], multichannel=True)
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
        ssim += structural_similarity(clean[i], recoverd[i], data_range=1, multichannel=True,channel_axis=-1)

    return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]


 
def add_distortions(imgs,args):
    batch_size,_,_,_=imgs.shape
    noise = torch.randn_like(imgs) * args.sigma
    imgs_noised = imgs + noise
    lrimage=converto_low_resolution(imgs,args.downsampling_factor)
    blur_image=blur_input_image(imgs, args.radius)

    inpaint_mask=generate_mask(batch_size, args.input_size,args.percentage,args.max_vertices,args.mask_radius,args.num_lines)
    inpaint_mask=inpaint_mask.to(args.device)
    blended_image = imgs *inpaint_mask


    patch_mask=patch_generator(imgs,args.num_patches,args.patch_size)
    patch_mask=patch_mask.to(args.device)
    final_mask=imgs*(1-patch_mask)

    return imgs_noised,lrimage,blur_image,blended_image,final_mask
class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def main(args):
    print('cuda availability', torch.cuda.is_available())
    device = torch.device(args.device)
    convert=Conversion()
    #load encoder and load the pretrained weight
  
    transform_test = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    experts,shared_encoder=create_experts_restoration(args)
    #transform data

    test_dir='/scratch3/ven073/test'
    dataset_noisy=DataNoisy(test_dir)
    dataset_super=DataSuper(test_dir)
    dataset_blurry=DataBlurry(test_dir)
    dataset_inpaint=DataInpaint(test_dir)
    dataset_mask=DataMask(test_dir)
    
    print('length of dataset',len(dataset_noisy))
    data_loader_noisy = DataLoader(dataset_noisy, batch_size=16, pin_memory=True, shuffle=False, num_workers=0)
    data_loader_super = DataLoader(dataset_super, batch_size=16, pin_memory=True, shuffle=False, num_workers=0)
    data_loader_blurry = DataLoader(dataset_blurry, batch_size=16, pin_memory=True, shuffle=False, num_workers=0)
    data_loader_inpaint = DataLoader(dataset_inpaint, batch_size=16, pin_memory=True, shuffle=False, num_workers=0)
    data_loader_mask = DataLoader(dataset_mask, batch_size=16, pin_memory=True, shuffle=False, num_workers=0)#load MOE model

    #load MOE model
    moe=MOE(experts,shared_encoder,args)
    
    #load weight of MOE moe_output2 for freezed and moe_output for not feezed
    checkpoint2= torch.load('/scratch3/ven073/moe_output2ddp/aggregator1_moe_model1000.pth',map_location=torch.device('cpu') )
    moe.load_state_dict(checkpoint2['model_state_dict'],strict=False)
    moe=moe.to(device)
    moe.eval()
    
#creating objects for evaluating psnr and ssim
    psnr_exp1 = AverageMeter()
    ssim_exp1 = AverageMeter()
    psnr_exp2 = AverageMeter()
    ssim_exp2 = AverageMeter()
    psnr_exp3 = AverageMeter()
    ssim_exp3 = AverageMeter()
    psnr_exp4 = AverageMeter()
    ssim_exp4 = AverageMeter()
    psnr_exp5 = AverageMeter()
    ssim_exp5 = AverageMeter()

    with torch.no_grad():   
        for ii, data_val in enumerate((data_loader_noisy), 0): 
            samples = data_val[0]
            imgs_noised = data_val[1]
            samples = samples.to(device, non_blocking=True)
            imgs_noised = imgs_noised.to(device, non_blocking=True)
            #imgs_noised,_,_,_,_=add_distortions(samples,args)
            mask_ratio=args.mask_ratio
            org_input=convert.normalization(samples)
            #task1
            output_moe,_=moe(org_input,imgs_noised,args)
            output_moe=convert.unpatchify(output_moe)
            a=convert.denormalization(output_moe)
            temp_psnr1, temp_ssim1, N = compute_psnr_ssim(a, samples)
            psnr_exp1.update(temp_psnr1, N)
            ssim_exp1.update(temp_ssim1, N)
    print("PSNR moe denoise : %.2f, SSIM moe denoise: %.4f" % (psnr_exp1.avg, ssim_exp1.avg))
    with torch.no_grad():   
        for ii, data_val in enumerate((data_loader_super), 0): 
            samples = data_val[0]
            lrimage = data_val[1]
            samples = samples.to(device, non_blocking=True)
            lrimage = lrimage.to(device, non_blocking=True)
            
            #_,lrimage,_,_,_=add_distortions(samples,args)
            mask_ratio=args.mask_ratio
            org_input=convert.normalization(samples)        
            #task2

            output_moe,_=moe(org_input,lrimage,args)
            output_moe=convert.unpatchify(output_moe)
            b=convert.denormalization(output_moe)
            temp_psnr2, temp_ssim2, N = compute_psnr_ssim(b, samples)
            psnr_exp2.update(temp_psnr2, N)
            ssim_exp2.update(temp_ssim2, N)
    print("PSNR moe superresolution : %.2f, SSIM moe superresolution: %.4f" % (psnr_exp2.avg, ssim_exp2.avg))
    with torch.no_grad():   
        for ii, data_val in enumerate((data_loader_blurry), 0): 

            samples = data_val[0]
            blur_image= data_val[1]
            samples = samples.to(device, non_blocking=True)
            blur_image = blur_image.to(device, non_blocking=True)
            
            #_,_,blur_image,_,_=add_distortions(samples,args)
            mask_ratio=args.mask_ratio
            org_input=convert.normalization(samples)     
            #task3
            output_moe,_=moe(org_input,blur_image,args)
            output_moe=convert.unpatchify(output_moe)
            c=convert.denormalization(output_moe)
            temp_psnr3, temp_ssim3, N = compute_psnr_ssim(c, samples)
            psnr_exp3.update(temp_psnr3, N)
            ssim_exp3.update(temp_ssim3, N)
        
    print("PSNR moe deblurring : %.2f, SSIM moe deblurring : %.4f" % (psnr_exp3.avg, ssim_exp3.avg))
    with torch.no_grad():   
        for ii, data_val in enumerate((data_loader_inpaint), 0):
            samples = data_val[0]
            inpaint_mask = data_val[1]
            samples = samples.to(device, non_blocking=True)
            inpaint_mask = inpaint_mask.to(device, non_blocking=True)
            
            #_,_,_,inpaint_mask,_=add_distortions(samples,args)
            mask_ratio=args.mask_ratio
            org_input=convert.normalization(samples)  
            
            #task4
         
            output_moe,_=moe(org_input,inpaint_mask,args)
            output_moe=convert.unpatchify(output_moe)
            d=convert.denormalization(output_moe)
            temp_psnr4, temp_ssim4, N = compute_psnr_ssim(d, samples)
            psnr_exp4.update(temp_psnr4, N)
            ssim_exp4.update(temp_ssim4, N)
    print("PSNR moe inpainting: %.2f, SSIM moe inpainting: %.4f" % (psnr_exp4.avg, ssim_exp4.avg))
    with torch.no_grad():   
        for ii, data_val in enumerate((data_loader_mask), 0): 
            samples = data_val[0]
            patch_mask = data_val[1]
            samples = samples.to(device, non_blocking=True)
            patch_mask = patch_mask.to(device, non_blocking=True)
            
            #_,_,_,_,patch_mask=add_distortions(samples,args)
            mask_ratio=args.mask_ratio
            org_input=convert.normalization(samples)  
            #task5
            output_moe,_=moe(org_input,patch_mask,args)
            output_moe=convert.unpatchify(output_moe)
            e=convert.denormalization(output_moe)
            temp_psnr5, temp_ssim5, N = compute_psnr_ssim(e, samples)
            psnr_exp5.update(temp_psnr5, N)
            ssim_exp5.update(temp_ssim5, N)
    print("PSNR moe demasking: %.2f, SSIM moe demasking: %.4f" % (psnr_exp5.avg, ssim_exp5.avg))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


 
