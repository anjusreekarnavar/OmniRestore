# Copyright (c) Meta Platforms, Inc. and affiliates.
#Anjusree Karnavar Griffith University 2024
#anjusree.karnavar@griffithuni.edu.au
# --------------------------------------------------------




import math
import sys
from typing import Iterable
from torch import nn
import torch
from torchvision.utils import save_image
import util.misc as misc
from inpaint_mask_generator import generate_mask,patch_generator
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched
from torch.utils.tensorboard import SummaryWriter
from bkp_files.callback import callback_For_Threshold
from bkp_files.callback import EarlyStopping

from augmentations import converto_low_resolution,blur_input_image
from temporary import Conversion
from torchvision import models

import random
import aggregator_copy 
from moe_training_validation_freeze import aggregator1_train
from temporary import Conversion
import torchvision.transforms as transforms
import PIL
import torch.nn.functional as F
layer_output={}
#random number generator for generating 
def random_generator():

    n=random.randint(1,5)
    return n

#from tensorboardX import SummaryWriter
def add_distortions(imgs,args):
    batch_size,_,_,_=imgs.shape
    noise = torch.randn_like(imgs) * args.sigma
    imgs_noised = imgs + noise
    lrimage=converto_low_resolution(imgs,args.downsampling_factor)
    blur_image=blur_input_image(imgs, args.radius)
    imgs_noised=normalization(imgs_noised,args)
    lrimage=normalization(lrimage,args)
    blur_image=normalization(lrimage,args)
    inpaint_mask=generate_mask(batch_size, args.input_size,args.percentage,args.max_vertices,args.mask_radius,args.num_lines)
    inpaint_mask=inpaint_mask.to(args.device)
    blended_image = imgs *inpaint_mask
    inpaint_mask=normalization(blended_image,args)
    patch_mask=patch_generator(imgs,args.num_patches,args.patch_size)
    patch_mask=patch_mask.to(args.device)
    final_mask=imgs*(1-patch_mask)
    final_mask=normalization(final_mask,args)
    return imgs_noised,lrimage,blur_image,inpaint_mask,final_mask

def early_stop(total_loss,length):
    early_stopping = EarlyStopping()
    avg_loss=total_loss/length
    early_stopping(avg_loss)
    return early_stopping.early_stop



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

#function for creating multi-distorted images 

def new_distorted_dataset(imgs,args):
    total_distortions=random_generator()
    available_distortions=['denoising','deblurring','super-resolution','inpainting','demasking']
    chosen_distortion=random.sample(available_distortions,total_distortions)
    to_distort=imgs
    batch_size,_,_,_=imgs.shape

    for i in chosen_distortion:
        if i=='denoising': 
            noise = torch.randn_like(to_distort) * args.sigma
            imgs_noised = to_distort + noise
            new_image=imgs_noised
        if i=='deblurring':
            blur_image=blur_input_image(to_distort, args.radius)
            new_image=blur_image
        if i=='super-resolution':
            lrimage=converto_low_resolution(to_distort,args.downsampling_factor)
            new_image=lrimage
        if i=='inpainting': 
             inpaint_mask=generate_mask(batch_size, args.input_size,args.percentage,args.max_vertices,args.mask_radius,args.num_lines)
             inpaint_mask=inpaint_mask.to(args.device)
             new_image = to_distort *inpaint_mask
        if i=='demasking':
            patch_mask=patch_generator(imgs,args.num_patches,args.patch_size)
            patch_mask=patch_mask.to(args.device)
            new_image=to_distort*(1-patch_mask)
        to_distort=new_image
    final_distorted_image=normalization(to_distort,args)
    final_distorted_image = final_distorted_image.to(args.device, non_blocking=True)
    return final_distorted_image

    
def train_one_epoch(model_moe,
                    data_loader, data_loader_val,optimizer,experts,
                    device,epoch:int,log_writer,loss_scaler,best_psnr,early_stopping,args):
   
    optimizer.zero_grad()
    model_moe,psnr_val=aggregator1_train(data_loader,data_loader_val,optimizer,model_moe,device,loss_scaler,epoch,log_writer,best_psnr,early_stopping,args)
        
    
    if args.output_dir and (epoch % 100 == 0 or epoch + 1 == args.epochs):
                torch.save({
                'model_state_dict': model_moe.module.state_dict(),
                            
                }, f'/scratch3/ven073/moe_outputddp/aggregator1_moe_model{epoch+1}.pth')

    return model_moe,psnr_val
    


    

        
    
