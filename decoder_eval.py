
# Copyright (c) Meta Platforms, Inc. and affiliates.
#Anjusree Karnavar Griffith University 2024
#anjusree.karnavar@griffithuni.edu.au
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
#from torchvision.utils import save_image
import torch.nn as nn
from inpaint_mask_generator import generate_mask,patch_generator2,mask_to
import time
import sys
from metrics_eval import AverageMeter,Conversion
from metrics_eval import compute_psnr_ssim
from augmentations import converto_low_resolution,blur_input_image,to_low_resolution
from pathlib import Path
import PIL
import torch
from torchvision.utils import save_image
from custom_dataset import TestDataset
import torch.backends.cudnn as cudnn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from metrics2 import calculate_metrics
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import  DataLoader
from PIL import Image, ImageFilter,ImageOps
import timm
from bkp_files.callback import EarlyStopping
from torch.utils.data import  DataLoader, random_split
from decoder import Decoder1,Decoder2,Decoder3,Decoder4,Decoder5

from temporary import Conversion
assert timm.__version__ == "0.5.4"  # version check
from util import misc
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import model_restoration

import timm.optim.optim_factory as optim_factory
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def get_args_parser():
    parser = argparse.ArgumentParser('restoreMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
  
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_root', default='/scratch3/ven073/datanew/data', type=str,
                        help='mask path')
    parser.add_argument('--mask_type', default='thin', type=str,
                        help='thin,thick,nn2,genhalf,ex64,ev2li')
   

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--sigma', default=0.25, type=float,
                        help='Std of Gaussian noise')
    parser.add_argument('--radius', default=2, type=int,
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
    parser.add_argument('--data_path', default='/scratch3/ven073/test/groundtruth', type=str,
                        help='dataset path')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')

    parser.add_argument('--output_dir', default='/scratch3/ven073/base_output4',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/scratch3/ven073/base_output4/log_dir',
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


def load_decoders(device):
    #loading noise expert
    denoise_expert=Decoder1()
    #path_noise='/export/home/s5284664/dmae2/output/decoder_denoise_epoch_81.pth'
    #denoise_expert=load_model(path_noise,denoise_expert)
    checkpoint1 =torch.load('/scratch3/ven073/base_output2/decoder_denoise_epoch_501.pth',map_location=torch.device('cpu') )
    denoise_expert.load_state_dict(checkpoint1['model_state_dict'],strict=False)
  
    denoise_expert=denoise_expert.to(device)
    denoise_expert.eval()
    
    #loading deblurring expert

    deblur_expert=Decoder3()
    #deblur_expert=load_model(path_deblur,deblur_expert)
    checkpoint2= torch.load('/scratch3/ven073/base_output2/decoder_deblur_epoch_501.pth',map_location=torch.device('cpu') )
    deblur_expert.load_state_dict(checkpoint2['model_state_dict'],strict=False)
    deblur_expert=deblur_expert.to(device)
    deblur_expert.eval()
    #loading superresolution expert
   
    superr_expert=Decoder2()
    #superr_expert=load_model(path_super,superr_expert)
    checkpoint3= torch.load('/scratch3/ven073/base_output2/decoder_super_epoch_501.pth',map_location=torch.device('cpu') )
    superr_expert.load_state_dict(checkpoint3['model_state_dict'],strict=False)
    superr_expert=superr_expert.to(device)
    superr_expert.eval()
    #loading masking expert
    masking_expert=Decoder5()
   
    checkpoint4= torch.load('/scratch3/ven073/base_output2/decoder_masking_epoch_501.pth',map_location=torch.device('cpu') )
    masking_expert.load_state_dict(checkpoint4['model_state_dict'],strict=False)
    masking_expert=masking_expert.to(device)
    masking_expert.eval()
    #loading inpaint expert
    inpaint_expert=Decoder4()
   
    checkpoint5= torch.load('/scratch3/ven073/base_output2/decoder_inpainting_epoch_501.pth',map_location=torch.device('cpu') )
    inpaint_expert.load_state_dict(checkpoint5['model_state_dict'],strict=False)
    inpaint_expert=inpaint_expert.to(device)
    inpaint_expert.eval()
    return denoise_expert,superr_expert,deblur_expert,inpaint_expert,masking_expert


def normalization(imgs,args):

        # normalization
        device=args.device
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

        if mean.device != imgs.device:
            mean = mean.to(device)
            std = std.to(device)
        imgs = (imgs - mean) / std
        
        return imgs

  
def add_distortions(imgs,args):
    batch_size,_,_,_=imgs.shape
    device=args.device
    noise = torch.randn_like(imgs) * args.sigma
    imgs_noised = imgs + noise
    imgs_noised = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(imgs_noised)
    save_image(imgs_noised[0],'/home/ven073/anju/dmae2/noiseimage.png')
    lrimage=to_low_resolution(imgs,args.downsampling_factor)
    save_image(lrimage[0],'/home/ven073/anju/dmae2/lrimage.png')
    #lrimage = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(lrimage)
    blur_image=blur_input_image(imgs, args.radius)
    blur_image = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(blur_image)
    save_image(blur_image[0],'/home/ven073/anju/dmae2/blurimage.png')
    #imgs_noised=normalization(imgs_noised,args)
    #lrimage=normalization(lrimage,args)
    #blur_image=normalization(lrimage,args)
    #inpaint_mask=generate_mask(batch_size, args.input_size,args.percentage,args.max_vertices,args.mask_radius,args.num_lines)
    #inpaint_mask=inpaint_mask.to(args.device)
    #blended_image = imgs *inpaint_mask
    inpaint_mask=mask_to(imgs,device,mask_root=args.mask_root,mask_type=args.mask_type)
    
    blended_image = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(inpaint_mask)
    blended_image = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(inpaint_mask)
    save_image(blended_image[0],'/home/ven073/anju/dmae2/inpaint.png')
    #inpaint_mask=normalization(blended_image,args)
    patch_mask=patch_generator2(imgs,device)
    patch_mask = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(patch_mask)
    #patch_mask=patch_mask.to(args.device)
    #final_mask=imgs*(patch_mask)
    save_image(patch_mask[0],'/home/ven073/anju/dmae2/masked.png')
    #final_mask=normalization(final_mask,args)
    return imgs_noised,lrimage,blur_image,blended_image,patch_mask
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
    
    
    shared_encoder = model_restoration.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    checkpoint= torch.load('/scratch3/ven073/base_output2/shared_encoder_501.pth',map_location=torch.device('cpu') )
    shared_encoder.load_state_dict(checkpoint['model_state_dict'],strict=False)
    shared_encoder=shared_encoder.to(device)
    shared_encoder.eval()
    
    #pretrained_path = torch.load(path)
    denoise_expert,superr_expert,deblur_expert,inpaint_expert,masking_expert=load_decoders(device)
    

    transform_test = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    data_path='/scratch3/ven073/test/groundtruth'
    dataset= TestDataset(image_folder=data_path,transform=transform_test)

    print('length of dataset',len(dataset))
    sampler_train = torch.utils.data.SequentialSampler(dataset)
   
    data_loader_test = DataLoader(dataset, batch_size=64, pin_memory=True, shuffle=False, num_workers=0)
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
    avg_psnr_all_batches = []
    avg_ssim_all_batches = []
    with torch.no_grad():   
         for samples in data_loader_test:
            samples = samples.to(device, non_blocking=True)
            samples = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(samples)
            imgs_noised,lrimage,blur_image,inpaint_mask,patch_mask=add_distortions(samples,args)
            mask_ratio=args.mask_ratio
            latent,mask,ids_restore=shared_encoder(imgs_noised,mask_ratio)
            prediction_denoise=denoise_expert(samples,latent,ids_restore,mask)
            denoised_image=convert.unpatchify(prediction_denoise[0])
            a=convert.denormalization(denoised_image)
            avg_psnr, avg_ssim = calculate_metrics(a, samples)

            avg_psnr_all_batches.append(avg_psnr)
            avg_ssim_all_batches.append(avg_ssim)
            temp_psnr1, temp_ssim1, N = compute_psnr_ssim(a, samples)
            psnr_exp1.update(temp_psnr1, N)
            ssim_exp1.update(temp_ssim1, N)
            latent,mask,ids_restore=shared_encoder(lrimage,mask_ratio)
            prediction_superresolve=superr_expert(samples,latent,ids_restore,mask)
            super_output=convert.unpatchify(prediction_superresolve[0])
            b=convert.denormalization(super_output)
            temp_psnr2, temp_ssim2, N = compute_psnr_ssim(b, samples)
            psnr_exp2.update(temp_psnr2, N)
            ssim_exp2.update(temp_ssim2, N)
            latent,mask,ids_restore=shared_encoder(blur_image,mask_ratio)
            prediction_deblur=deblur_expert(samples,latent,ids_restore,mask) 
            blur_output=convert.unpatchify(prediction_deblur[0])
            #blur_output=prediction_deblur[0]
            c=convert.denormalization(blur_output)
            temp_psnr3, temp_ssim3, N = compute_psnr_ssim(c, samples)
            psnr_exp3.update(temp_psnr3, N)
            ssim_exp3.update(temp_ssim3, N)
         
            latent,mask,ids_restore=shared_encoder(inpaint_mask,mask_ratio)
            prediction_inpaint=inpaint_expert(samples,latent,ids_restore,mask)
            inpaint_output=convert.unpatchify(prediction_inpaint[0])
            
            d=convert.denormalization(inpaint_output)
            temp_psnr4, temp_ssim4, N = compute_psnr_ssim(d, samples)
            psnr_exp4.update(temp_psnr4, N)
            ssim_exp4.update(temp_ssim4, N)
        
            latent,mask,ids_restore=shared_encoder(patch_mask,mask_ratio)
            prediction_demask=masking_expert(samples,latent,ids_restore,mask) 
            mask_output=convert.unpatchify(prediction_demask[0])
            
            e=convert.denormalization(mask_output)
            temp_psnr5, temp_ssim5, N = compute_psnr_ssim(e, samples)
            psnr_exp5.update(temp_psnr5, N)
            ssim_exp5.update(temp_ssim5, N)
    print("PSNR expert1 denoise : %.2f, SSIM expert1 denoise: %.4f" % (psnr_exp1.avg, ssim_exp1.avg))
    print("PSNR expert2 superresolution : %.2f, SSIM expert2 superresolution: %.4f" % (psnr_exp2.avg, ssim_exp2.avg))
    print("PSNR expert3 deblurring : %.2f, SSIM expert3 deblurring : %.4f" % (psnr_exp3.avg, ssim_exp3.avg))
    print("PSNR expert4 inpainting: %.2f, SSIM expert4 inpainting: %.4f" % (psnr_exp4.avg, ssim_exp4.avg))
    print("PSNR expert5 demasking: %.2f, SSIM expert5 demasking: %.4f" % (psnr_exp5.avg, ssim_exp5.avg))
    overall_avg_psnr = np.mean(avg_psnr_all_batches)
    overall_avg_ssim = np.mean(avg_ssim_all_batches)

    print(f"Average PSNR: {overall_avg_psnr}")
    print(f"Average SSIM: {overall_avg_ssim}")
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


 
