import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch.nn as nn
from torch import optim
import torch
import math
from timm.models.vision_transformer import PatchEmbed
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from typing import Iterable, Optional
from torch.optim.lr_scheduler import StepLR
import timm
from torchvision.models import vgg16
import torch.nn.init as init
from perceptualloss import LossNetwork
import PIL
assert timm.__version__ == "0.5.4"  # version check
import model_multirestoration
from math import log10
import util.misc as misc
from torch.utils.data import  DataLoader, random_split

from torch import autograd
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
from mixture_of_experts import DeblurringExpert,SuperResolutionExpert,MaskedExpert,InpaintingExpert,MOE
from newmodel import DenoiseExpert
import torchvision.transforms as transforms
from PIL import Image, ImageFilter,ImageOps
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import model_multirestoration
from callback import EarlyStopping
from augmentations import converto_low_resolution,blur_input_image


def get_args_parser():
    parser = argparse.ArgumentParser('DMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--accum_iter', default=400, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='dmae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--expert_epochs', default=500, type=int)
    

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--topk', default=1, type=int,
                        help='topk expert for gating')
    parser.add_argument('--expert_lr', default=0.001, type=int,
                        help='images input size')
    parser.add_argument('--hidden_dim', default=128, type=int)

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--inpaint_ratio', default=0.20, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--sigma', default=0.25, type=float,
                        help='Std of Gaussian noise')
    parser.add_argument('--downsampling_factor', default=4, type=int,
                        help='for superresolution')
    parser.add_argument('--radius', default=1, type=int,
                        help='for radius')

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
    parser.add_argument('--data_path', default='/lscratch/s5284664/datanew', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='/export/home/s5284664/dmae',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/export/home/s5284664/dmae/output/log_dir',
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
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser
class Conversion(nn.Module):
    def __init__(self,img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024):
        super().__init__()
        self. mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
       
        self.gdevice = torch.device('cuda')
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed=self.patch_embed.to(self.gdevice)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim,device=self.gdevice))
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim,device=self.gdevice), requires_grad=False)  # fixed sin-cos embedding
        
    def convert_noisy(self,imgs,sigma):
        
        noise = torch.randn_like(imgs) * sigma
        imgs_noised = imgs + noise
        
        imgs = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(imgs)
        imgs_noised = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(imgs_noised)
        
        return imgs_noised
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    def masking(self,imgs,mask_ratio):
        x = self.patch_embed(imgs)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1) # the order of elements

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) 
        mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask = self.unpatchify(mask)  # 1 is removing, 0 is keeping
        
        
        im_masked = imgs * (1 - mask)

        return im_masked
    
    def normalization(self,imgs):

        # normalization

        if self.mean.device != imgs.device:
            self.mean = self.mean.to(imgs.device)
            self.std = self.std.to(imgs.device)
        imgs = (imgs - self.mean) / self.std
        
        return imgs

    def denormalization(self,imgs):
            self.mean = self.mean.to(imgs.device)
            self.std = self.std.to(imgs.device)
            denormalized_image = imgs * self.std + self.mean
            return denormalized_image

    def prepare_model(self,chkpt_dir, arch):
    # build model
    #model = getattr(models_dmae, arch)()
        model = getattr(model_multirestoration, arch)()
    # load model
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)

 
        return model
@torch.no_grad()
def init_weights(m):
  
  if isinstance(m, nn.Conv2d):
    nn.init.xavier_normal_(m.weight)
    m.bias.fill_(0.0)

class ClassExperts(nn.Module):
     def __init__(self,args):
            super().__init__()
            self.output_dim=3
            self.input_dim=3
            self.hidden_dim=args.hidden_dim
            self.device=args.device
            self.sigma=args.sigma
            self.expert_epochs=args.expert_epochs
            self.log_dir=args.log_dir
            self.lambda_loss=0.04
            self.expert_lr=args.expert_lr
             # normalization parameters
            self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
          
     def forward(self,data_loader_train,data_loader_val,model_restoration):
            
            convert_imgs=Conversion()
            log_writer = SummaryWriter(log_dir=self.log_dir)
            
            expert1 = DenoiseExpert(num_stages=3)
            expert1.apply(init_weights)
           
            expert2 = MaskedExpert(self.input_dim,self.output_dim)
    
            expert3 = SuperResolutionExpert(self.input_dim,self.output_dim)
            
            expert4 =DeblurringExpert(self.input_dim,self.output_dim)
           
            expert5 = InpaintingExpert(self.input_dim,self.output_dim)
   

           
            
           
            expert1_lr=0.0001
            expert2_lr=0.0000001
            expert3_lr=0.0000001
            expert4_lr=0.0000001
            expert5_lr=0.0000001
            expert1=expert1.cuda()
            expert2=expert2.cuda()
            expert3=expert3.cuda()
            expert4=expert4.cuda()
            expert5=expert5.cuda()
              # Set up loss
            
            
            # Optimizers for experts
            optimizer_expert1 = optim.Adam(expert1.parameters(), lr=expert1_lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
            optimizer_expert2 = optim.Adam(expert2.parameters(), lr=expert2_lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
            optimizer_expert3 = optim.Adam(expert3.parameters(), lr=expert3_lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
            optimizer_expert4 = optim.Adam(expert4.parameters(), lr=expert4_lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
            optimizer_expert5 = optim.Adam(expert5.parameters(), lr=expert5_lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        
        
            
            start_time2 = time.time()
            dur=[]
           
            # --- Define the perceptual loss network --- #
            vgg_model = vgg16(pretrained=True).features[:16]
            vgg_model = vgg_model.to(self.device)
            vgg_model.eval()
            for param in vgg_model.parameters():
                param.requires_grad = False
            loss_network = LossNetwork(vgg_model)
            
            loss_network=loss_network.cuda()
            loss_network.eval()
            for param in loss_network.parameters():
                param.requires_grad = False
            #expert2
            start_time2 = time.time()
            dur=[]
            
            early_stopping_expert2 = EarlyStopping()
            early_stopping_expert1 = EarlyStopping()
            early_stopping_expert3 = EarlyStopping()
            early_stopping_expert4 = EarlyStopping()
            early_stopping_expert5 = EarlyStopping()
            model_restoration.eval()
            start_time2 = time.time()
            dur=[]
            log_writer.flush()
            start_time2 = time.time()
            dur=[]
            model_restoration.eval()
            for epoch_expert1 in range(self.expert_epochs):
                expert1_loss=0
                for param in model_restoration.parameters():
                    param.requires_grad = False
                for param in expert1.parameters():
                    param.requires_grad = True
                for step1, (samples3, _) in enumerate(data_loader_train):
                   
                    expert1.train(True)
                    t0 = time.time()
                    samples = samples3.to(self.device, non_blocking=True)
                    samples=samples.cuda()
                    samples=transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(samples)
                    noise = torch.randn_like(samples) * args.sigma
                    imgs_noisy= samples + noise
                    imgs_noisy = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(imgs_noisy)
                    samples=convert_imgs.normalization(samples)
                    imgs_noisy= convert_imgs.normalization(imgs_noisy)
                    
                    x=model_restoration.forward_encoder(imgs_noisy, mask_ratio=0,flag=1)
                    pred_1= model_restoration.forward_decoder(x, ids_restore=0,flag=1)
                    pred_1 = model_restoration.unpatchify(pred_1)
                  
                    optimizer_expert1.zero_grad()
                    with torch.cuda.amp.autocast():
                        
                        outputs_expert1 = expert1(pred_1)
                        #outputs_expert1=transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(outputs_expert1)
                        smooth_loss = F.smooth_l1_loss(outputs_expert1, samples)
                        perceptual_loss = loss_network(outputs_expert1, samples)
                        loss_expert1 = smooth_loss + self.lambda_loss*perceptual_loss 
                    loss_expert1.backward()
                    optimizer_expert1.step()
                  
                    expert1_loss+=loss_expert1.item()*samples.size(0)
                    dur.append(time.time() - t0)
                    print('Epoch: {},runningLoss: expert1{:.4f},Time(s) {:.4f}'.format(epoch_expert1,expert1_loss,np.mean(dur)))
              
                expert1_val=0
                expert1.eval()    
                with torch.no_grad():
                    for step11, (samples11, _) in enumerate(data_loader_val):   
                                samples = samples11.to(self.device, non_blocking=True)
                                samples=samples.cuda()
                                samples = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(samples)
                                noise = torch.randn_like(samples) * args.sigma
                                imgs_noisy= samples + noise
                                imgs_noisy = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(imgs_noisy)
                                samples=convert_imgs.normalization(samples)
                                imgs_noisy= convert_imgs.normalization(imgs_noisy)
                                x=model_restoration.forward_encoder(imgs_noisy, mask_ratio=0,flag=1)
                                pred_1= model_restoration.forward_decoder(x, ids_restore=0,flag=1)
                                pred_1 = model_restoration.unpatchify(pred_1)
                                
                                outputs_expert1 = expert1(pred_1)
                                #outputs_expert1=transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(outputs_expert1)
                                smooth_loss = F.smooth_l1_loss(outputs_expert1, samples)
                                perceptual_loss = loss_network(outputs_expert1, samples)
                                loss_expert1_val = smooth_loss + self.lambda_loss*perceptual_loss 
                                expert1_val+=loss_expert1_val.item()*samples.size(0)
                avg_expert1=expert1_val/len(data_loader_val)
                torch.cuda.empty_cache()
                log_writer.add_scalars(' Loss Expert1',
                    { 'Training' :expert1_loss/len(data_loader_train), 'Validation' :  avg_expert1 },
                    epoch_expert1)
                early_stopping_expert1(avg_expert1)
                if early_stopping_expert1.early_stop:
                     checkpoint1={
                    'epoch': epoch_expert1,
                    'model_state_dict': expert1.state_dict(),
                    'optimizer_state_dict': optimizer_expert1.state_dict(),
                    'loss': loss_expert1,
                     } 
                     torch.save(checkpoint1,f'/export/home/s5284664/dmae/output/expert1_best.pth')
                     break
           
                if  epoch_expert1 % 20 == 0 or epoch_expert1 + 1 == args.expert_epochs: 
                    checkpoint1={
                    'epoch': epoch_expert1,
                    'model_state_dict': expert1.state_dict(),
                    'optimizer_state_dict': optimizer_expert1.state_dict(),
                    'loss': loss_expert1,
                     } 
             
                    torch.save(checkpoint1,f'/export/home/s5284664/dmae/output/expert1-epoch{epoch_expert1}.pth')
            print('Training finished expert1, took {:.2f}s'.format(time.time() - start_time2))

            for epoch_expert5 in range(self.expert_epochs):
                expert5_loss=0
                for param in model_restoration.parameters():
                    param.requires_grad = False
                for param in expert5.parameters():
                    param.requires_grad = True
                for step5, (samples51,_) in enumerate(data_loader_train):
                    
                    expert5.train(True)
                    t0 = time.time()
                    samples = samples51.to(self.device, non_blocking=True)
                    
                    samples=convert_imgs.normalization(samples)
                    latent, mask, ids_restore = model_restoration.forward_encoder(samples, mask_ratio=0.60,flag=0)
                    pred_5 = model_restoration.forward_decoder(latent, ids_restore,flag=0)  # [N, L, p*p*3]
                    print(pred_5.shape)
                    pred_5 = model_restoration.unpatchify(pred_5)
                    
                    optimizer_expert5.zero_grad()
                    with torch.cuda.amp.autocast():
                        
                        outputs_expert5 = expert5(pred_5)
                        smooth_loss = F.smooth_l1_loss(outputs_expert5, samples)
                        perceptual_loss = loss_network(outputs_expert5, samples)
                        loss_expert5 =smooth_loss + self.lambda_loss*perceptual_loss 
                        
                    loss_expert5.backward()
                    optimizer_expert5.step()
                    
                    expert5_loss+=loss_expert5.item()*samples.size(0)
                    dur.append(time.time() - t0)
                    
                    print('Epoch: {},runningLossexpert5:{:.4f},Time(s) {:.4f}'.format(epoch_expert5,expert5_loss,np.mean(dur)))
                
                expert5_val=0    
                expert5.eval()
                with torch.no_grad():
                        for step21, (samples52, _) in enumerate(data_loader_val):
                                    
                                    samples = samples52.to(self.device, non_blocking=True)
                                   
                                    samples=convert_imgs.normalization(samples)
                                    latent, mask, ids_restore = model_restoration.forward_encoder(samples, mask_ratio=0.60,flag=0)
                                    pred_5 = model_restoration.forward_decoder(latent, ids_restore,flag=0)  # [N, L, p*p*3]
                                    pred_5 = model_restoration.unpatchify(pred_5)
                                    
                                    outputs_expert5 = expert5(pred_5)
                                    smooth_loss = F.smooth_l1_loss(outputs_expert5, samples)
                                    perceptual_loss = loss_network(outputs_expert5, samples)
                                    loss_expert5_val = smooth_loss + self.lambda_loss*perceptual_loss 
                                    expert5_val+=loss_expert5_val.item()*samples.size(0)
                                    print('Epoch: {},runningvalidationLossexpert5:{:.4f},Time(s) {:.4f}'.format(epoch_expert5,expert5_val,np.mean(dur)))
                avg_expert5=expert5_val/len(data_loader_val)
               
                log_writer.add_scalars(' Loss Expert5',
                    { 'Training' :expert5_loss/len(data_loader_train), 'Validation' :  avg_expert5 },
                    epoch_expert5)
                early_stopping_expert5(avg_expert5)
                if early_stopping_expert5.early_stop:
                    checkpoint5 = {
                         'epoch': epoch_expert5,
                         'model_state_dict': expert5.state_dict(),
                         'optimizer_state_dict': optimizer_expert5.state_dict(),
                        'loss': loss_expert5}
                    torch.save(checkpoint5, f'/export/home/s5284664/dmae/output/expert5_best.pth')
                    break
                if  epoch_expert5 % 20 == 0 or epoch_expert5 + 1 == args.expert_epochs: 
                         checkpoint5 = {
                         'epoch': epoch_expert5,
                         'model_state_dict': expert5.state_dict(),
                         'optimizer_state_dict': optimizer_expert5.state_dict(),
                        'loss': loss_expert5}
                         torch.save(checkpoint5, f'/export/home/s5284664/dmae/output/expert5{epoch_expert5}.pth')
                
            print('Training finished expert5, took {:.2f}s'.format(time.time() - start_time2))
      
            
            log_writer.flush()
            for epoch_expert2 in range(self.expert_epochs):
                expert2_loss=0
                for param in model_restoration.parameters():
                    param.requires_grad = True
                for step2, (samples2,_) in enumerate(data_loader_train):
                    
                    expert2.train(True)
                    t0 = time.time()
                    samples = samples2.to(self.device, non_blocking=True)
                    
                    samples=convert_imgs.normalization(samples)
                    latent, mask, ids_restore = model_restoration.forward_encoder(samples, mask_ratio=0.60,flag=0)
                    pred_2 = model_restoration.forward_decoder(latent, ids_restore,flag=0)  # [N, L, p*p*3]
                    pred_2 = model_restoration.unpatchify(pred_2)
                    
                    optimizer_expert2.zero_grad()
                    with torch.cuda.amp.autocast():
                        
                        outputs_expert2 = expert2(pred_2)
                        smooth_loss = F.smooth_l1_loss(outputs_expert2, samples)
                        perceptual_loss = loss_network(outputs_expert2, samples)
                        loss_expert2 = smooth_loss + self.lambda_loss*perceptual_loss 
                        
                    loss_expert2.backward()
                    optimizer_expert2.step()
                  
                    expert2_loss+=loss_expert2.item()*samples.size(0)
                    dur.append(time.time() - t0)
                    
                    print('Epoch: {},runningLossexpert2:{:.4f},Time(s) {:.4f}'.format(epoch_expert2,expert2_loss,np.mean(dur)))
                
                expert2_val=0    
                expert2.eval()
                with torch.no_grad():
                        for step21, (samples21, _) in enumerate(data_loader_val):
                                    
                                    samples = samples21.to(self.device, non_blocking=True)
                                   
                                    samples=convert_imgs.normalization(samples)
                                    latent, mask, ids_restore = model_restoration.forward_encoder(samples, mask_ratio=0.60,flag=0)
                                    pred_2 = model_restoration.forward_decoder(latent, ids_restore,flag=0)  # [N, L, p*p*3]
                                    pred_2 = model_restoration.unpatchify(pred_2)
                                    
                                    outputs_expert2 = expert2(pred_2)
                                    
                                    smooth_loss = F.smooth_l1_loss(outputs_expert2, samples)
                                    perceptual_loss = loss_network(outputs_expert2, samples)
                                    loss_expert2_val = smooth_loss + self.lambda_loss*perceptual_loss 
                                    expert2_val+=loss_expert2_val.item()*samples.size(0)
                                    print('Epoch: {},runningvalidationLossexpert2:{:.4f},Time(s) {:.4f}'.format(epoch_expert2,expert2_val,np.mean(dur)))
                avg_expert2=expert2_val/len(data_loader_val)
               
                log_writer.add_scalars(' Loss Expert2',
                    { 'Training' :expert2_loss/len(data_loader_train), 'Validation' :  avg_expert2 },
                    epoch_expert2)
                early_stopping_expert2(avg_expert2)
                if early_stopping_expert2.early_stop:
                    checkpoint2={
                    'epoch': epoch_expert2,
                    'model_state_dict': expert2.state_dict(),
                    'optimizer_state_dict': optimizer_expert2.state_dict(),
                    'loss': loss_expert2,
                     }
                    torch.save(checkpoint2,f'/export/home/s5284664/dmae/output/expert2_best.pth')
                    break
            
                if  epoch_expert2 % 20 == 0 or epoch_expert2 + 1 == args.expert_epochs:                
                    checkpoint2={'epoch': epoch_expert2,
                'model_state_dict': expert2.state_dict(),
                'optimizer_state_dict': optimizer_expert2.state_dict(),
                'loss': loss_expert2,
                 }
                    torch.save(checkpoint2,f'/export/home/s5284664/dmae/output/expert2-epoch{epoch_expert2}.pth')
            print('Training finished expert2, took {:.2f}s'.format(time.time() - start_time2))
            #expert 3
            start_time2 = time.time()
            dur=[]
            log_writer.flush()
            for epoch_expert3 in range(self.expert_epochs):
                expert3_loss=0
                for param in model_restoration.parameters():
                    param.requires_grad = True
                for step3, (samples, _) in enumerate(data_loader_train):
                    
                    expert3.train(True)
                    t0 = time.time()
                    samples = samples.to(self.device, non_blocking=True)
                    samples=samples.cuda()
                    lrim=converto_low_resolution(samples,args.downsampling_factor)
                    lrim= convert_imgs.normalization(lrim)
                    samples=convert_imgs.normalization(samples)
                    
                    y=model_restoration.forward_encoder(lrim, mask_ratio=0,flag=1)
                    pred_3 = model_restoration.forward_decoder(y, ids_restore=0,flag=1)
                    pred_3 = model_restoration.unpatchify(pred_3)
                 
                    t0 = time.time()
                    optimizer_expert3.zero_grad()
                    with torch.cuda.amp.autocast():
                   
                        outputs_expert3 = expert3(pred_3)
                        smooth_loss = F.smooth_l1_loss(outputs_expert3, samples)
                        perceptual_loss = loss_network(outputs_expert3, samples)
                        loss_expert3 = smooth_loss + self.lambda_loss*perceptual_loss 
                    loss_expert3.backward()
                    optimizer_expert3.step()
                    
                    expert3_loss+=loss_expert3.item()*samples.size(0)
                    dur.append(time.time() - t0)
                    print('Epoch:{},runningLossexpert3:{:.4f},Time(s) {:.4f}'.format(epoch_expert3,expert3_loss,np.mean(dur)))
                 
                expert3_val=0  
                expert3.eval()
                with torch.no_grad():      
                    for step31, (samples31, _) in enumerate(data_loader_val):
                            
                                     samples = samples31.to(self.device, non_blocking=True)
                                     samples=samples.cuda()
                                     lrim=converto_low_resolution(samples,args.downsampling_factor)
                                     lrim= convert_imgs.normalization(lrim)
                                     samples=convert_imgs.normalization(samples)
                                     y=model_restoration.forward_encoder(lrim, mask_ratio=0,flag=1)
                                     pred_3 = model_restoration.forward_decoder(y, ids_restore=0,flag=1)
                                     pred_3 = model_restoration.unpatchify(pred_3)
                              
                                     outputs_expert3 = expert3(pred_3)
                                     smooth_loss = F.smooth_l1_loss(outputs_expert3, samples)
                                     perceptual_loss = loss_network(outputs_expert3, samples)
                                     loss_expert3_val = smooth_loss + self.lambda_loss*perceptual_loss 
                                     expert3_val+=loss_expert3_val.item()*samples.size(0)

                avg_expert3=expert3_val/len(data_loader_val)
                
                log_writer.add_scalars(' Loss Expert3',
                    { 'Training' :expert3_loss/len(data_loader_train), 'Validation' : avg_expert3 },
                    epoch_expert3)
                early_stopping_expert3(avg_expert3)
                if early_stopping_expert3.early_stop:
                     checkpoint3={
                    'epoch': epoch_expert3,
                    'model_state_dict': expert3.state_dict(),
                    'optimizer_state_dict': optimizer_expert3.state_dict(),
                    'loss': loss_expert3,
                     }
                     torch.save(checkpoint3,f'/export/home/s5284664/dmae/output/expert3_best.pth')
                     break
                
           
                if  epoch_expert3 % 20 == 0 or epoch_expert3 + 1 == args.expert_epochs: 
                    checkpoint3={
            'epoch': epoch_expert3,
            'model_state_dict': expert3.state_dict(),
            'optimizer_state_dict': optimizer_expert3.state_dict(),
            'loss': loss_expert3,
            }
                    torch.save(checkpoint3,f'/export/home/s5284664/dmae/output/expert3-epoch{epoch_expert3}.pth')
            print('Training finished expert3, took {:.2f}s'.format(time.time() - start_time2))
          

    # Training loop for expert 4
    
            log_writer.flush()
            start_time2 = time.time()
            dur=[]
            
            for epoch_expert4 in range(self.expert_epochs):
                expert4_loss=0
                for param in model_restoration.parameters():
                    param.requires_grad = True
                for step4, (samples4, _) in enumerate(data_loader_train):
                    
                    expert4.train(True)
                    t0 = time.time()
                    samples = samples4.to(self.device, non_blocking=True)
                    samples=samples.cuda()
                    blurim=blur_input_image(samples, args.radius)
                    blurim=convert_imgs.normalization(blurim)
                    samples=convert_imgs.normalization(samples)
                   
                    z=model_restoration.forward_encoder(blurim, mask_ratio=0,flag=1)
                    pred_4 = model_restoration.forward_decoder(z, ids_restore=0,flag=1)
                    pred_4 = model_restoration.unpatchify(pred_4)
                  
                    optimizer_expert4.zero_grad()
                    with torch.cuda.amp.autocast():
                        
                        outputs_expert4 = expert4(pred_4)
                        smooth_loss = F.smooth_l1_loss(outputs_expert4, samples)
                        perceptual_loss = loss_network(outputs_expert4, samples)
                        loss_expert4 = smooth_loss + self.lambda_loss*perceptual_loss 
                    loss_expert4.backward()
                    optimizer_expert4.step()
                   
                    expert4_loss+=loss_expert4.item()*samples.size(0)
                    dur.append(time.time() - t0)
                    print('Epoch: {},runningLoss expert4:{:.4f},Time(s) {:.4f}'.format(epoch_expert4,expert4_loss,np.mean(dur)))
                
                expert4_val=0 
                expert4.eval()  
                with torch.no_grad():   
                    for step41, (samples41, _) in enumerate(data_loader_val):
    
                                samples = samples41.to(self.device, non_blocking=True)
                                samples=samples.cuda()
                                blurim=blur_input_image(samples, args.radius)
                                blurim=convert_imgs.normalization(blurim)
                                samples=convert_imgs.normalization(samples)
                                z=model_restoration.forward_encoder(blurim, mask_ratio=0,flag=1)
                                pred_4 = model_restoration.forward_decoder(z, ids_restore=0,flag=1)
                                pred_4 = model_restoration.unpatchify(pred_4)
                               
                                outputs_expert4 = expert4(pred_4)
                                smooth_loss = F.smooth_l1_loss(outputs_expert4, samples)
                                perceptual_loss = loss_network(outputs_expert4, samples)
                                loss_expert4_val = smooth_loss + self.lambda_loss*perceptual_loss 
                                expert4_val+= loss_expert4_val.item()*samples.size(0)
                avg_expert4=expert4_val/len(data_loader_val)
               
                log_writer.add_scalars(' Loss Expert4',
                    { 'Training' :expert4_loss/len(data_loader_train), 'Validation' : avg_expert4 },
                    epoch_expert4)
                early_stopping_expert4(avg_expert4)
                if early_stopping_expert4.early_stop:
                     checkpoint4={
                    'epoch': epoch_expert4,
                    'model_state_dict': expert4.state_dict(),
                    'optimizer_state_dict': optimizer_expert4.state_dict(),
                    'loss': loss_expert4,
                     }
                     torch.save(checkpoint4,f'/export/home/s5284664/dmae/output/experts4/expert4_best.pth')
                     break
               
         
                if  epoch_expert4 % 20 == 0 or epoch_expert4 + 1 == args.expert_epochs: 
                     checkpoint4={
            'epoch': epoch_expert4,
            'model_state_dict': expert4.state_dict(),
            'optimizer_state_dict': optimizer_expert4.state_dict(),
            'loss': loss_expert4,
            }
                     torch.save(checkpoint4,f'/export/home/s5284664/dmae/output/expert4-epoch{epoch_expert4}.pth')
            
            print('Training finished expert4, took {:.2f}s'.format(time.time() - start_time2))
            start_time2 = time.time()
            dur=[]
            log_writer.flush()
            

    
        
            return expert1,expert2,expert3,expert4,expert5





def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
   

  
    convert_imgs=Conversion()
    convert_imgs=convert_imgs.cuda()
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
    #dataset_val=datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_train)
    
    #print('classes',len(dataset_train.classes ))
    #print('dataset val classes', len(dataset_val.classes))
    train_size = int(0.8 * len(dataset_train))  # 80% of the dataset for training
    test_size = len(dataset_train) - train_size   
    train_data, test_data = random_split(dataset_train, [train_size, test_size])    
    print('length of training data',len(train_data))
    print('length of validation data',len(test_data))
    
    sampler_train = torch.utils.data.RandomSampler(train_data)
    sampler_val = torch.utils.data.SequentialSampler(test_data)
    
    data_loader_train = torch.utils.data.DataLoader(
        train_data, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        test_data, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    
    
   

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    path='/export/home/s5284664/dmae/experts/model_output2/checkpoint-180.pth'
   
    model_restoration = convert_imgs.prepare_model(path, args.model)
    model_restoration=model_restoration.cuda()
   
    
  
    expertobj=ClassExperts(args)
    expertobj=expertobj.cuda()
    exp1,exp2,exp3,exp4,exp5=expertobj(data_loader_train,data_loader_val,model_restoration)
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
