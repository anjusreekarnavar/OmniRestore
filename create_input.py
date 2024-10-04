import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import datasets, transforms, models
from temporary import Conversion
from inpaint_mask_generator import generate_mask,patch_generator
import PIL
import random
from augmentations import converto_low_resolution,blur_input_image
from perceptualloss import LossNetwork
from torchvision import models
import torch.nn.functional as F

def normalization(imgs,args):

        mean = imgs.mean(dim=(0, 2, 3), keepdim=True)  # Shape (1, 15, 1, 1)
        std = imgs.std(dim=(0, 2, 3), keepdim=True)
        mean=mean.to(args.device)
        std=std.to(args.device)
        std_nonzero = std.clone()  # Clone the std tensor
        std_nonzero[std_nonzero == 0] = 1
        normalized_images = (imgs - mean) / std_nonzero

        
        return normalized_images
def create_input_dcnn(experts,shared_encoder,samples,input_distort,device,args):
        denoise_expert=experts[0]
        superr_expert=experts[1]
        deblur_expert=experts[2]
        inpaint_expert=experts[3]
        masking_expert=experts[4]
        convert=Conversion()
        #expert1
        latent, mask,ids_restore=shared_encoder(input_distort,args.mask_ratio)
        prediction_1=denoise_expert(samples,latent,ids_restore,mask)
        prediction_1=convert.unpatchify(prediction_1[0])
        prediction_1=convert.denormalization(prediction_1)
        #expert2
        #save_image(prediction_1[4],'/home/ven073/anju/dmae2/from_expert1.png')
        prediction_2=deblur_expert(samples,latent,ids_restore,mask)
        prediction_2=convert.unpatchify(prediction_2[0])
        prediction_2=convert.denormalization(prediction_2)
        #expert3
        #save_image(prediction_2[4],'/home/ven073/anju/dmae2/from_expert2.png')
        prediction_3=superr_expert(samples,latent,ids_restore,mask)
        prediction_3=convert.unpatchify(prediction_3[0])
        prediction_3=convert.denormalization(prediction_3)
        #expert4
        #save_image(prediction_3[4],'/home/ven073/anju/dmae2/from_expert3.png')
        prediction_4=inpaint_expert(samples,latent,ids_restore,mask)
        prediction_4=convert.unpatchify(prediction_4[0])
        prediction_4=convert.denormalization(prediction_4)
        #expert5
        #save_image(prediction_4[4],'/home/ven073/anju/dmae2/from_expert4.png')
        prediction_5=masking_expert(samples,latent,ids_restore,mask)
        prediction_5=convert.unpatchify(prediction_5[0])
        prediction_5=convert.denormalization(prediction_5)

        #prediction_1=convert.normalization(prediction_1)
        #prediction_2=convert.normalization(prediction_2)
        #prediction_3=convert.normalization(prediction_3)
        #prediction_4=convert.normalization(prediction_4)
        #prediction_5=convert.normalization(prediction_5)
        


        outputs=[prediction_1,prediction_2,prediction_3,prediction_4,prediction_5]
        
        combined_images = torch.cat(outputs, dim=1)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        std=torch.tensor(std)
        mean=torch.tensor(mean)
        mean=mean.to(args.device)
        std=std.to(args.device)
        for i in range(5):
                combined_images[:, i*3:(i+1)*3] = (combined_images[:, i*3:(i+1)*3] - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
      
      
        return combined_images