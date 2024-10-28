import argparse
import datetime
import json
import numpy as np
import os
#from torchvision.utils import save_image
import torch.nn as nn
import time
import sys
from metrics_eval import AverageMeter
from pathlib import Path
import PIL
import torch
from torchvision.utils import save_image
from data.custom_dataset import TestDataset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import  DataLoader
from PIL import Image, ImageFilter,ImageOps
import timm
from torch.utils.data import  DataLoader, random_split
from architecture.decoder_with_cnn import Decoder
from conversion import Conversion
assert timm.__version__ == "0.5.4"  # version check
from util import misc
from tensorboardX import SummaryWriter
import architecture.model_restoration
from data.custom_testset import DataNoisy,DataBlurry,DataSuper,DataInpaint,DataMask
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

def get_args_parser():
    parser = argparse.ArgumentParser('restoreMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--mask_ratio', default=0, type=int,
                        help='masking')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)


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
    denoise_expert=Decoder(decoder_depth=10)
    checkpoint1 =torch.load('/scratch3/ven073/decoder_shared/decoder_denoise_epoch_21.pth',map_location=torch.device('cpu') )
    denoise_expert.load_state_dict(checkpoint1['model_state_dict'],strict=False)
    denoise_expert.to(device)
    denoise_expert.eval()
    
    deblur_expert=Decoder(decoder_depth=10)
    checkpoint2= torch.load('/scratch3/ven073/decoder_shared/decoder_deblur_epoch_21.pth',map_location=torch.device('cpu') )
    deblur_expert.load_state_dict(checkpoint2['model_state_dict'],strict=False)
    deblur_expert.to(device)
    deblur_expert.eval()
    #loading superresolution expert
   
    superr_expert=Decoder(decoder_depth=10)
    #superr_expert=load_model(path_super,superr_expert)
    checkpoint3= torch.load('/scratch3/ven073/decoder_shared/decoder_super_epoch_21.pth',map_location=torch.device('cpu') )
    superr_expert.load_state_dict(checkpoint3['model_state_dict'],strict=False)
    superr_expert=superr_expert.to(device)
    superr_expert.eval()
    #loading masking expert
    masking_expert=Decoder(decoder_depth=10)
   
    checkpoint4= torch.load('/scratch3/ven073/decoder_shared/decoder_masking_epoch_21.pth',map_location=torch.device('cpu') )
    masking_expert.load_state_dict(checkpoint4['model_state_dict'],strict=False)
    masking_expert.to(device)
    masking_expert.eval()
    #loading inpaint expert
    inpaint_expert=Decoder(decoder_depth=10)
   
    checkpoint5= torch.load('/scratch3/ven073/decoder_shared/decoder_inpainting_epoch_21.pth',map_location=torch.device('cpu') )
    inpaint_expert.load_state_dict(checkpoint5['model_state_dict'],strict=False)
    inpaint_expert.to(device)
    inpaint_expert.eval()
    return denoise_expert,superr_expert,deblur_expert,inpaint_expert,masking_expert


def denormalization(imgs,args):

        # normalization
        device=args.device
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # Shape (1, 3, 1, 1) for broadcasting
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        mean = mean.to(device)
        std = std.to(device) 
        denormalized_image = imgs * std + mean
        return denormalized_image
        
  
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
    shared_encoder = model_restoration.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    checkpoint= torch.load('/scratch3/ven073/decoder_shared/shared_encoder_21.pth',map_location=torch.device('cpu') )
    shared_encoder.load_state_dict(checkpoint['model_state_dict'],strict=False)
    shared_encoder.to(device)
    shared_encoder.eval()
    denoise_expert,superr_expert,deblur_expert,inpaint_expert,masking_expert=load_decoders(device)

    test_dir='/scratch3/ven073/test'
    dataset_noisy=DataNoisy(test_dir)
    dataset_super=DataSuper(test_dir)
    dataset_blurry=DataBlurry(test_dir)
    dataset_inpaint=DataInpaint(test_dir)
    dataset_mask=DataMask(test_dir)
    alex = lpips.LPIPS(net='alex').to(device)

    data_loader_noisy = DataLoader(dataset_noisy, batch_size=1, pin_memory=True, shuffle=False, num_workers=3)
    data_loader_super = DataLoader(dataset_super, batch_size=1, pin_memory=True, shuffle=False, num_workers=3)
    data_loader_blurry = DataLoader(dataset_blurry, batch_size=1, pin_memory=True, shuffle=False, num_workers=3)
    data_loader_inpaint = DataLoader(dataset_inpaint, batch_size=1, pin_memory=True, shuffle=False, num_workers=3)
    data_loader_mask = DataLoader(dataset_mask, batch_size=1, pin_memory=True, shuffle=False, num_workers=3)#load MOE model
#creating objects for evaluating psnr and ssim
    psnr_exp1 = AverageMeter()
    ssim_exp1 = AverageMeter()
    
    psnr,ssim, pips = [], [], []
    with torch.no_grad():   
        for ii, data_val in enumerate((data_loader_noisy), 0): 
            samples = data_val[0]
            save_image(samples[0],'/home/ven073/anju/original.jpg')
            imgs_noised = data_val[1]
            samples = samples.to(device, non_blocking=True)
            imgs_noised = imgs_noised.to(device, non_blocking=True)
            samples = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(samples)
            imgs_noised= transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(imgs_noised)
            latent,mask,ids_restore=shared_encoder(imgs_noised,args.mask_ratio)
            prediction_denoise=denoise_expert(samples,latent,ids_restore,mask,args)
            #denoised_image=convert.unpatchify(prediction_denoise[0])
            restored=denormalization(prediction_denoise[0],args)
            save_image(restored[0],'/home/ven073/anju/restored_denorm.jpg')
            restored=torch.clamp(restored,0,1)
            save_image(restored[0],'/home/ven073/anju/restored_clamp.jpg')
            temp_psnr1, temp_ssim1, N = compute_psnr_ssim(restored, samples)
            psnr_exp1.update(temp_psnr1, N)
            ssim_exp1.update(temp_ssim1, N)
        print("PSNR expert1 denoise : %.2f, SSIM expert1 denoise: %.4f" % (psnr_exp1.avg, ssim_exp1.avg))
        for ii, data_val in enumerate((data_loader_super), 0): 
            samples = data_val[0]
            lrimage = data_val[1]
            samples = samples.to(device, non_blocking=True)
            lrimage = lrimage.to(device, non_blocking=True)
            latent,mask,ids_restore=shared_encoder(lrimage,args.mask_ratio)
            prediction_superresolve=superr_expert(samples,latent,ids_restore,mask,args)
            #super_output=convert.unpatchify(prediction_superresolve[0])
            restored=convert.denormalization(prediction_superresolve[0])
            restored=torch.clamp(restored,0,1)
            temp_psnr2, temp_ssim2, N = compute_psnr_ssim(restored, samples)
            psnr_exp1.update(temp_psnr2, N)
            ssim_exp1.update(temp_ssim2, N)
        print("PSNR expert2 superresolution : %.2f, SSIM expert2 superresolution : %.4f" % (psnr_exp1.avg, ssim_exp1.avg))
        for ii, data_val in enumerate((data_loader_blurry), 0): 
            samples = data_val[0]
            blur_image = data_val[1]
            samples = samples.to(device, non_blocking=True)
            blur_image = blur_image.to(device, non_blocking=True)
            latent,mask,ids_restore=shared_encoder(blur_image,args.mask_ratio)
            prediction_deblur=deblur_expert(samples,latent,ids_restore,mask,args) 
            #blur_output=convert.unpatchify(prediction_deblur[0])
            #blur_output=prediction_deblur[0]
            restored=denormalization(prediction_deblur[0],args)
            restored=torch.clamp(restored,0,1)
            temp_psnr3, temp_ssim3, N = compute_psnr_ssim(restored, samples)
            psnr_exp1.update(temp_psnr3, N)
            ssim_exp1.update(temp_ssim3, N)
        print("PSNR expert3 deblurring : %.2f, SSIM expert3 deblurring : %.4f" % (psnr_exp1.avg, ssim_exp1.avg))
        for ii, data_val in enumerate((data_loader_inpaint), 0):
            samples = data_val[0]
            inpaint_mask = data_val[1]
            samples = samples.to(device, non_blocking=True)
            inpaint_mask = inpaint_mask.to(device, non_blocking=True)   
            latent,mask,ids_restore=shared_encoder(inpaint_mask,args.mask_ratio)
            prediction_inpaint=inpaint_expert(samples,latent,ids_restore,mask,args)
            #inpaint_output=convert.unpatchify(prediction_inpaint[0])
            restored=denormalization(prediction_inpaint[0],args)
            restored=torch.clamp(restored,0,1)
            temp_psnr4, temp_ssim4, N = compute_psnr_ssim(restored, samples)
            psnr_exp1.update(temp_psnr4, N)
            ssim_exp1.update(temp_ssim4, N)
        print("PSNR expert4 inpainting: %.2f, SSIM expert4 inpainting: %.4f" % (psnr_exp1.avg, ssim_exp1.avg))
        for ii, data_val in enumerate((data_loader_mask), 0): 
            samples = data_val[0]
            patch_mask = data_val[1]
            samples = samples.to(device, non_blocking=True)
            patch_mask = patch_mask.to(device, non_blocking=True) 
            latent,mask,ids_restore=shared_encoder(patch_mask,args.mask_ratio)
            prediction_demask=masking_expert(samples,latent,ids_restore,mask,args) 
            #mask_output=convert.unpatchify(prediction_demask[0])
            restored=denormalization(prediction_demask[0],args)
            restored=torch.clamp(restored,0,1)
            temp_psnr5, temp_ssim5, N = compute_psnr_ssim(restored, samples)
            psnr_exp1.update(temp_psnr5, N)
            ssim_exp1.update(temp_ssim5, N)
    print("PSNR expert5 demasking: %.2f, SSIM expert5 demasking: %.4f" % (psnr_exp1.avg, ssim_exp1.avg))
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)


 