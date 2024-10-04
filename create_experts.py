
/*
 * Copyright (c) 2024 ---Anjusree Karnavar,Griffith University
 */
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
from pathlib import Path
import PIL
import torch
import torch.backends.cudnn as cudnn
from aggregator import DepthCNN
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import  DataLoader
from PIL import Image, ImageFilter,ImageOps
import timm
from decoder import Decoder1,Decoder2,Decoder3,Decoder4,Decoder5
assert timm.__version__ == "0.5.4"  # version check
from util import misc
import model_restoration


   

def load_decoders_encoder(args,device):
    #loading noise expert
    denoise_expert=Decoder1()
    #path_noise='/export/home/s5284664/dmae2/output/decoder_denoise_epoch_81.pth'
    #denoise_expert=load_model(path_noise,denoise_expert)
    checkpoint1 =torch.load('/scratch3/ven073/base_output3/decoder_denoise_epoch_1600.pth',map_location=torch.device('cpu') )
    denoise_expert.load_state_dict(checkpoint1['model_state_dict'],strict=False)
  
    denoise_expert=denoise_expert.to(device)
 
    #loading deblurring expert

    deblur_expert=Decoder3()
    #deblur_expert=load_model(path_deblur,deblur_expert)
    checkpoint2= torch.load('/scratch3/ven073/base_output3/decoder_deblur_epoch_1600.pth',map_location=torch.device('cpu') )
    deblur_expert.load_state_dict(checkpoint2['model_state_dict'],strict=False)
    deblur_expert=deblur_expert.to(device)
    
    #loading superresolution expert
   
    superr_expert=Decoder2()
    #superr_expert=load_model(path_super,superr_expert)
    checkpoint3= torch.load('/scratch3/ven073/base_output3/decoder_super_epoch_1600.pth',map_location=torch.device('cpu') )
    superr_expert.load_state_dict(checkpoint3['model_state_dict'],strict=False)
    superr_expert=superr_expert.to(device)
    
    #loading masking expert
    masking_expert=Decoder5()
   
    checkpoint4= torch.load('/scratch3/ven073/base_output3/decoder_masking_epoch_1600.pth',map_location=torch.device('cpu') )
    masking_expert.load_state_dict(checkpoint4['model_state_dict'],strict=False)
    masking_expert=masking_expert.to(device)

   
    inpaint_expert=Decoder4()
   
    checkpoint5= torch.load('/scratch3/ven073/base_output3/decoder_inpainting_epoch_1600.pth',map_location=torch.device('cpu') )
    inpaint_expert.load_state_dict(checkpoint5['model_state_dict'],strict=False)
    inpaint_expert=inpaint_expert.to(device)
  

    shared_encoder = model_restoration.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    checkpoint= torch.load('/scratch3/ven073/base_output3/shared_encoder_1600.pth',map_location=torch.device('cpu') )
    shared_encoder.load_state_dict(checkpoint['model_state_dict'],strict=False)
    shared_encoder=shared_encoder.to(device)
  
   
    return denoise_expert,superr_expert,deblur_expert,inpaint_expert,masking_expert,shared_encoder

      # define the model


def create_experts_restoration(args):
    denoise_expert,superr_expert,deblur_expert,inpaint_expert,masking_expert,shared_encoder=load_decoders_encoder(args,args.device)
   
    
    experts=[denoise_expert,superr_expert,deblur_expert,inpaint_expert,masking_expert]
    return experts,shared_encoder
    

