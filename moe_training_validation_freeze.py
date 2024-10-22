# Copyright (c) Meta Platforms, Inc. and affiliates.
#Anjusree Karnavar Griffith University 2024
#anjusree.karnavar@griffithuni.edu.au
# --------------------------------------------------------
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
import util.misc as misc
from augmentations import converto_low_resolution,blur_input_image
from perceptualloss import LossNetwork
from torchvision import models
import torch.nn.functional as F
import util.lr_sched as lr_sched
from distorted_dataset import new_distorted_dataset
from bkp_files.calculate_psnr_ssim import batch_PSNR

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

def loss_function_aggregator1(pred,inputs,args):
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

def aggregator1_train(data_loader,data_loader_val,optimizer,moe_model,device,loss_scaler,epoch,log_writer,best_psnr,early_stopping,args):
    moe_model.train()
    total_train_loss=0
    accum_iter = args.accum_iter

    #for data_iter_step, (samples, _) in enumerate(data_loader):
    for data_iter_step, data_val in enumerate((data_loader), 0):
        
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples = data_val[0].to(device, non_blocking=True)
        #samples = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(samples)
        save_image(samples[0],'/home/ven073/anju/dmae2/samples1moetrain.png')
        #combined_distortion=new_distorted_dataset(samples,args)
        combined_distortion=data_val[1].to(device, non_blocking=True)
        #combined_distortion= transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(combined_distortion)
        
        save_image(combined_distortion[0],'/home/ven073/anju/dmae2/combined_distortedmoetrain.png')
     
        convert=Conversion()
        org_input=convert.normalization(samples)
      
        with torch.autocast(device_type="cuda"):
            output_moe,gate_output=moe_model(org_input,combined_distortion,args)
            output_moe=convert.unpatchify(output_moe)
            
            loss=loss_function_aggregator1(output_moe,org_input,args)
        l1_loss=moe_model.module.l1_regularization(gate_output,args,epoch)
        loss=loss+l1_loss
        loss_value = loss.item()
        print(f"MOE training loss  Epoch {epoch + 1}, Loss: {loss_value}")
        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=moe_model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
          
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss_moe', loss_value_reduce, epoch_1000x)
        #print(f"MOE Training loss  Epoch {epoch + 1}, Loss: {loss.item()}")    
        #total_train_loss+=loss
        #loss.backward()
        #optimizer.step()
    #scheduler.step()
        
    #mean_loss=total_train_loss/len(data_loader)
    #output_moe=convert.unpatchify(output_moe)
    output_moe=convert.denormalization(output_moe)
  
    save_image(output_moe[6],'/home/ven073/anju/dmae2/moe.png')
        
    moe_model2,psnr_val=aggregator1_validation(data_loader_val,moe_model,device,epoch,log_writer,accum_iter,best_psnr,early_stopping,args)
    return moe_model2,psnr_val
def aggregator1_validation(data_loader_val,moe_model,device,epoch,log_writer,accum_iter,best_psnr,early_stopping,args):
    moe_model.eval()
    total_val_loss=0
   
    convert=Conversion()
    with torch.no_grad():
        psnr_val_rgb=[]
        #for data_iter_step, (samples, _) in enumerate(data_loader_val):
        for data_iter_step, data_val in enumerate((data_loader_val), 0):
            samples = data_val[0].to(device, non_blocking=True)
            #samples = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(samples)
            save_image(samples[0],'/home/ven073/anju/dmae2/samplesmoe.jpg')
            #combined_distortion=new_distorted_dataset(samples,args)
            combined_distortion=data_val[1].to(device, non_blocking=True)
            #combined_distortion= transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(combined_distortion)
            
            save_image(combined_distortion[0],'/home/ven073/anju/dmae2/combined_distortedmoeval.png')
        
            org_input=convert.normalization(samples)
            output_moe,gate_output=moe_model(org_input,combined_distortion,args)
          
            output_moe=convert.unpatchify(output_moe)
            
            loss=loss_function_aggregator1(output_moe,org_input,args)
            l1_loss=moe_model.module.l1_regularization(gate_output,args,epoch)
            loss=loss+l1_loss
            loss_value = loss.item()
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
       
                epoch_1000x = int((data_iter_step / len(data_loader_val) + epoch) * 1000)
                log_writer.add_scalar('moe validation loss', loss_value_reduce, epoch_1000x)
         
            print(f"MOE validation loss  Epoch {epoch + 1}, Loss: {loss_value}")
            early_stopping(loss_value_reduce)
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                torch.save({
                    'epoch': epoch, 
                    'model_state_dict': moe_model.module.state_dict()
                            
                    }, f'/scratch3/ven073/moe_outputddp/aggregator_moe_latest.pth')
                break
   
            output_moe=convert.denormalization(output_moe)
            save_image(output_moe[6],'/home/ven073/anju/dmae2/moe_val.png')
            #restored = torch.clamp(output_moe,0,1)  
            psnr_val_rgb.append(batch_PSNR(output_moe, samples, False).item())
        psnr_val_rgb = sum(psnr_val_rgb)/len(data_loader_val)
        if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    print("[Ep %d\t PSNR moe: %.4f\t] ----  [best_Ep_MOE  %d Best_PSNR_MOE %.4f] " % (epoch, psnr_val_rgb,best_epoch,best_psnr))
                    torch.save({
                    'epoch': epoch, 
                    'model_state_dict': moe_model.module.state_dict()
                            
                    }, f'/scratch3/ven073/moe_outputddp/aggregator1_moe_best.pth')
        
    return moe_model,best_psnr


