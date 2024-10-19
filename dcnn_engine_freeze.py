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
from dcnn_training_expert_freeze import aggregator2_train
from temporary import Conversion
import torchvision.transforms as transforms
import PIL
import torch.nn.functional as F


#function for creating multi-distorted images 


    
def train_one_epoch(encoder,depth_model,
                    data_loader_train, data_loader_val,optimizer,scheduler,experts,
                    device,epoch:int,log_writer,best_psnr,early_stopping,args):
    
    train_avg2,val_avg2,depth_model,psnr_val=aggregator2_train(experts,depth_model,encoder,data_loader_train,data_loader_val,optimizer,scheduler,epoch,device,best_psnr,args)
    log_writer.add_scalar(' Loss training DepthCNN',train_avg2,epoch+ 1)
    log_writer.add_scalar(' Loss validation DepthCNN',val_avg2,epoch+ 1)
    log_writer.add_scalars(' Loss DepthCNN',
                    { 'Training' :train_avg2, 'Validation' :val_avg2},epoch+ 1)
    if args.output_dir and (epoch % 100 == 0 or epoch + 1 == args.epochs):
            torch.save({
                'model_state_dict': depth_model.module.state_dict(),
                            
                }, f'/scratch3/ven073/dcnn_output2ddp/aggregator2_dcnn_epoch_{epoch+1}.pth')
    early_stopping(val_avg2)
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                torch.save({
                    'epoch': epoch, 
                    'model_state_dict': depth_model.module.state_dict()
                            
                    }, f'/scratch3/ven073/dcnn_output2ddp/aggregator_dcnn_latest.pth')
                break
    return depth_model,psnr_val
    


    

        
    
