# Copyright (c) Meta Platforms, Inc. and affiliates.
#Anjusree Karnavar Griffith University 2024
#anjusree.karnavar@griffithuni.edu.au
# --------------------------------------------------------




import sys
import os
#import requests
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
import torch.nn as nn

from PIL import Image, ImageFilter,ImageOps
import torch
from timm.models.vision_transformer import PatchEmbed, Block
import PIL
#from skimage.util import random_noise
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

class Conversion(nn.Module):
      def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024):
        super().__init__()
   
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.gdevice = torch.device('cuda')
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed=self.patch_embed.to(self.gdevice)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim,device=self.gdevice), requires_grad=False) 
      def preprocessing(self,x):
           x = self.patch_embed(x)

        # add pos embed w/o cls token
        # x: (N, H*W/patch_size**2, patch_size**2 *3)
           x = x + self.pos_embed[:, 1:, :]
           return x

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
      
      def masking(self, imgs,mask_ratio):
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


def all_conversion(imgs,mask_ratio1,mask_ratio2,sigma,mean,std,device):
        conversion_obj=Conversion()
        conversion_obj.to(device)
        #adding noise
        noise = torch.randn_like(imgs) * sigma
        imgs_noised = imgs + noise
        #resizing
        
        imgs = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(imgs)
        imgs_noised = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(imgs_noised)
        #normalization
        if mean.device != imgs.device:
                mean = mean.to(device)
                std = std.to(imgs.device)
        imgs = (imgs - mean) / std
        imgs_noised = (imgs_noised - mean) / std
        img_to_convert=imgs
        img_to_blur=imgs
       
        #creating and preprocessinglow resolution image
        lrimage=converto_low_resolution(img_to_convert,downsampling_factor=4)
        
        #creating and preprocessing blur image
        blur_image=blur_input_image(img_to_blur, radius=5)
        blur_image = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(blur_image)
        
        #big masked image
       
        lmask = conversion_obj.masking(imgs, mask_ratio1)
        #small 
        smask=conversion_obj.masking(imgs, mask_ratio2)
       

        return imgs,lrimage,blur_image,imgs_noised,lmask,smask

def add_gaussian_noise(image,sigma):
    noise = torch.randn_like(image) * sigma
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)  # Clamp values between 0 and 1
def salt_pepper_noise(img,sigma):
   
        s_and_p = torch.tensor(random_noise(img, mode='s&p', salt_vs_pepper=sigma, clip=True))
        return s_and_p
def speckle_noise(img,sigma):
    
        speckle_noise = torch.tensor(random_noise(img, mode='speckle', mean=0, var=sigma, clip=True))
        return speckle_noise
def add_impulse_noise(image, sigma):
   
    mask = torch.rand(image.shape) < sigma
    noise = torch.randint(0, 2, image.shape).float()
    noisy_image = image * (1 - mask) + noise * mask
    return noisy_image
def add_poisson_noise_batch(images, sigma):
   
    poisson_noise = dist.Poisson(images * sigma)
    noisy_images = poisson_noise.sample()
    return noisy_images.float()

          
def assignlabels(randnum):
     if randnum==0:
          label='gaussian'
     elif randnum==1:   
          label=='salt_pepper'
     elif randnum==2:
          label='speckle_noise'
     elif randnum==3:
          label='impulse'
     else:
          label='poisson'
    
     return label      
def augment_with_multiple_noises(image_tensor,sigma):
    # Add multiple types of noise to the same image
    noisy_image = add_gaussian_noise(image_tensor, sigma=sigma/3)
    noisy_image = salt_pepper_noise(noisy_image, sigma=sigma/3)
    noisy_image= speckle_noise(noisy_image,sigma=sigma/3)
    return noisy_image

       
   
def converto_low_resolution(img,downsampling_factor):
  
  image_tensor = img  

  avg_pool = torch.nn.AvgPool2d(kernel_size=downsampling_factor, stride=downsampling_factor)
  low_res_image_tensor = avg_pool(image_tensor)
  
  #low_res_image =low_res_image_tensor.squeeze(0)  # Remove batch dimension
 
  out = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(low_res_image_tensor)
  return out

def to_low_resolution(img,downsampling_factor):
  
  image_tensor = img  
  target_size=img.shape[2]
  target_size=(target_size,target_size)
  new_height=img.shape[2]//downsampling_factor
  new_width=img.shape[3]//downsampling_factor
  downsampled_images=F.interpolate(img,size=(new_height,new_width),mode='bicubic',align_corners=False)
  resized_images = F.interpolate(downsampled_images, size=target_size, mode='bicubic', align_corners=False)
  #avg_pool = torch.nn.AvgPool2d(kernel_size=downsampling_factor, stride=downsampling_factor)
  #low_res_image_tensor = avg_pool(image_tensor)
  
  #low_res_image =low_res_image_tensor.squeeze(0)  # Remove batch dimension
 
  #out = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(low_res_image_tensor)
  return resized_images

# function for blurring

def blur_transform(image):
   pil_image = transforms.ToPILImage()(image)
   blurred_pil_image = pil_image.filter(ImageFilter.GaussianBlur(1))
   blurred_tensor_image = transforms.ToTensor()(blurred_pil_image)
   return blurred_tensor_image

   
def blur_input_image(batch_of_images, radius):
    """
    Apply Gaussian blur to an image tensor.
    """
    for i in range(batch_of_images.size(0)):
        pil_image = TF.to_pil_image(batch_of_images[i])

    # Apply blur filter
        blurred_image = pil_image.filter(ImageFilter.GaussianBlur(radius))

    # Convert back to tensor
        batch_of_images[i] = TF.to_tensor(blurred_image)
    #blurred_image=torch.stack([blur_transform(image)for image in tensor_images])
    #blurred_image=transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.2, 2.0))

   
    return batch_of_images 


    
 
    

  
    
