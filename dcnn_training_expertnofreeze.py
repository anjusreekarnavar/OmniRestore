# Copyright (c) Meta Platforms, Inc. and affiliates.
#Anjusree Karnavar Griffith University 2024
#anjusree.karnavar@griffithuni.edu.au
# --------------------------------------------------------


import torch
import torch.nn as nn
import numpy as np
import torchvision
from PIL import Image
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
from distorted_dataset import new_distorted_dataset
from create_input import create_input_dcnn
from bkp_files.calculate_psnr_ssim import batch_PSNR


def loss_function(pred,inputs,args):
    vgg_model = models.vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to(args.device)
    vgg_model.eval()
    for param in vgg_model.parameters():
        param.requires_grad = False
    loss_network = LossNetwork(vgg_model)
            
    loss_network=loss_network.to(args.device)
    loss_network.eval()
    for param in loss_network.parameters():
        param.requires_grad = False
    lambda_loss=0.5#previous value was 0.25 for dcnn
  
    smooth_loss = F.smooth_l1_loss(pred, inputs)
    perceptual_loss = loss_network(pred, inputs)
    loss=smooth_loss + lambda_loss*perceptual_loss 
    return loss




def aggregator2_train(experts,depth_cnn_model,encoder,data_loader_train,data_loader_val,optimizer,scheduler,epoch,device,best_psnr,args):

    convert=Conversion()
    total_loss=0
    criterion = nn.MSELoss() 
    #for data_iter_step, (samples, _) in enumerate(data_loader_train):
    #for distorted_image,clean_image in data_loader_train:
    for data_iter_step, data_val in enumerate((data_loader_train), 0):
       
        samples = data_val[0].to(device, non_blocking=True)
        #samples = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(samples)
        save_image(samples[0],'/home/ven073/anju/dmae2/samples1dcnntrain.png')
        #combined_distortion=new_distorted_dataset(samples,args)
        combined_distortion=data_val[1].to(device, non_blocking=True)
        #samples = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(samples)
        #combined_distortion = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(combined_distortion)
        
        #combined_distortion=new_distorted_dataset(samples,args)
        #combined_distortion=combined_distortion.to(device, non_blocking=True)
        save_image(combined_distortion[0],'/home/ven073/anju/dmae2/combined_distorteddcnn.png')
        input_image=create_input_dcnn(experts,encoder,samples,combined_distortion,args.device,args)
        
        optimizer.zero_grad()
        depth_cnn_output=depth_cnn_model(input_image)
        
        org_input=convert.normalization(samples)
        loss=loss_function(depth_cnn_output,org_input,args)
        loss.backward()
        optimizer.step()
        print(f"DepthCNN  training loss  Epoch {epoch + 1}, Loss: {loss.item()}")
        total_loss+=loss
    mean_loss=total_loss/len(data_loader_train)
    scheduler.step(mean_loss)
    dt=convert.denormalization(depth_cnn_output)
    #dt = torch.clamp(dt, 0, 1)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    std=torch.tensor(std)
    mean=torch.tensor(mean)
    mean=mean.to(args.device)
    std=std.to(args.device)
    #scales = torch.tensor([1.0, 0.9, 1.0])
    #scales=scales.to(args.device)  # Reduce green channel slightly
    #dt=dt * scales[:, None, None]
    #output_images = depth_cnn_output * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)

# Clamp the output to ensure values are within [0, 1]
    #output_images = torch.clamp(output_images, 0, 1)
    #for i in output_images:
                #i*255.0
                #image = transforms.ToPILImage()(i)
                #image.save('/home/ven073/anju/dmae2/dcnntrain.png')
              
                
                #break
  
    val_avg,dep_cnn_model,best_psnr=aggregator2_validation(experts,depth_cnn_model,encoder,data_loader_val,epoch,device,best_psnr,args)
    return mean_loss,val_avg,dep_cnn_model,best_psnr

def aggregator2_validation(experts,depth_cnn_model,encoder,data_loader_val,epoch,device,best_psnr,args):
 
    convert=Conversion()
    total_loss=0
    criterion = nn.MSELoss() 
    depth_cnn_model.eval()
    with torch.no_grad():
        psnr_val_rgb=[]
        #for data_iter_step, (samples, _) in enumerate(data_loader_val):
        #for distorted_image,clean_image in data_loader_val:
        for  data_iter_step, data_val in enumerate((data_loader_val), 0):
    
            samples = data_val[0].to(device, non_blocking=True)
            combined_distortion = data_val[1].to(device, non_blocking=True)
            #samples = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(samples)
            #combined_distortion = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(combined_distortion)
            save_image(samples[0],'/home/ven073/anju/dmae2/samplesdcnnval.jpg')
         
            #combined_distortion=new_distorted_dataset(samples,args)
            #combined_distortion=combined_distortion.to(device, non_blocking=True)
            save_image(combined_distortion[0],'/home/ven073/anju/dmae2/combined_distorteddcnnval.png')
            input_image=create_input_dcnn(experts,encoder,samples,combined_distortion,args.device,args)
          
            depth_cnn_output=depth_cnn_model(input_image)
            org_input=convert.normalization(samples)
        
            loss=loss_function(depth_cnn_output,org_input,args)
            print(f"DepthCNN  validation loss  Epoch {epoch + 1}, Loss: {loss.item()}")
            total_loss+=loss
       
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        std=torch.tensor(std)
        mean=torch.tensor(mean)
        mean=mean.to(args.device)
        std=std.to(args.device)
    #scales = torch.tensor([1.0, 0.9, 1.0])
    #scales=scales.to(args.device)  # Reduce green channel slightly
    #dt=dt * scales[:, None, None]
        output_images = depth_cnn_output * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
        output_images2=output_images
# Clamp the output to ensure values are within [0, 1]
        output_images = torch.clamp(output_images, 0, 1)
        for i in output_images:
                i*255.0
                image = transforms.ToPILImage()(i)
                image.save('/home/ven073/anju/dmae2/dcnnval.png')
                break
        #output_images=convert.denormalization(output_images)
        psnr_val_rgb.append(batch_PSNR(output_images2, samples, False).item())
        psnr_val_rgb = sum(psnr_val_rgb)/len(data_loader_val)
        if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    print("[Ep %d\t PSNR dcnn: %.4f\t] ----  [best_Ep_dcnn  %d Best_PSNR_dcnn %.4f] " % (epoch, psnr_val_rgb,best_epoch,best_psnr))
                    torch.save({
                    'epoch': epoch, 
                    'model_state_dict': depth_cnn_model.module.state_dict()
                            
                    }, f'/scratch3/ven073/dcnn_outputddp/aggregator1_dcnn_best.pth')
        print("[Ep %d\t PSNR dcnn: %.4f\t] ----  [best_Ep_dcnn  %d Best_PSNR_dcnn %.4f] " % (epoch, psnr_val_rgb,best_epoch,best_psnr))
        
        return total_loss/len(data_loader_val),depth_cnn_model,best_psnr
    
  


