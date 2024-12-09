import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import  models
from temporary import Conversion
from torchvision import datasets, transforms, models
import PIL
import sys
import math
import random
import util.misc as misc
from perceptualloss import LossNetwork
from torchvision import models
import torch.nn.functional as F
import util.lr_sched_moe as lr_sched
from calculate_psnr_ssim import batch_PSNR



def loss_function_aggregator(pred,inputs,args):
    
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
    loss = smooth_loss+lambda_loss*perceptual_loss 
    
    return loss

def aggregator_train(data_loader_train,data_loader_val,optimizer,moe_model,device,
                     loss_scaler,epoch,log_writer,best_psnr,eval_now,scheduler,args):
   
    if args.freeze:
            for i, expert in enumerate(moe_model.module.trained_experts):
                for param in expert.parameters():
                     param.requires_grad=False
                expert.eval()
           
            
    else:  
            for i, expert in enumerate(moe_model.module.trained_experts):
                for param in expert.parameters():
                     param.requires_grad=True
             
            for param in moe_model.module.shared_encoder.parameters():
                     param.requires_grad=True 
    
    for param in moe_model.module.gating_network.parameters():
                param.required_grad=True
    accum_iter = args.accum_iter
    criterion=torch.nn.MSELoss()
    convert=Conversion()
    psnr_val_rgb=[]
    total_loss=0
    #for data_iter_step, (samples, _) in enumerate(data_loader):
    for data_iter_step, data_train in enumerate((data_loader_train), 0):
        
        
        samples = data_train[0].to(device, non_blocking=True)
        #samples = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(samples)
        save_image(samples[0],'/home/ven073/anju/dmae2/samples1moetrain.png')
        combined_distortion=data_train[1].to(device, non_blocking=True)
        save_image(combined_distortion[0],'/home/ven073/anju/dmae2/combined_distortedmoetrain.png')
        org_input=convert.normalization(samples)
       # with torch.autocast(device_type="cuda"):
        output_moe,gate_output=moe_model(samples,combined_distortion,args)
        #restored = torch.clamp(output_moe,0.0,1.0)
        loss=loss_function_aggregator(output_moe,org_input,args)
        l1_loss=moe_model.module.l1_regularization(gate_output)
        loss=loss+l1_loss
        restore=convert.denormalization(output_moe)
        save_image(restore[1],'/home/ven073/anju/dmae2/moe2.png') 
        loss_value = loss.item()
        total_loss+=loss_value
        print(f"MOE training loss  Epoch {epoch + 1}, Loss: {loss_value}")
        loss /= accum_iter

        if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for name, param in moe_model.module.gating_network.named_parameters():
                 print(f'{name}.grad: {param.grad}')
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
          
            epoch_1000x = int((data_iter_step / len(data_loader_train) + epoch) * 1000)
            log_writer.add_scalar('train_loss_moe', loss_value_reduce, epoch_1000x)
    mean_loss=total_loss/len(data_loader_train)
    
    moe_model.eval()
    for i, expert in enumerate(moe_model.module.trained_experts):
            expert.eval()
    moe_model.module.gating_network.eval() 
    val_loss=0
 
    with torch.no_grad():
                
  
                for data_iter_step, data_val in enumerate((data_loader_val), 0):
                    samples = data_val[0].to(device, non_blocking=True)
                    #samples = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(samples)
           
                    save_image(samples[0],'/home/ven073/anju/dmae2/samplesmoe.jpg')
     
                    combined_distortion=data_val[1].to(device, non_blocking=True)
                    org_input=convert.normalization(samples)
                    save_image(combined_distortion[0],'/home/ven073/anju/dmae2/combined_distortedmoeval.png')
                    output_moe,gate_output=moe_model(samples,combined_distortion,args)
                    #restored = torch.clamp(output_moe,0.0,1.0)
                    loss=loss_function_aggregator(output_moe,org_input,args)
                    l1_loss=moe_model.module.l1_regularization(gate_output)
                    loss=loss+l1_loss
                    restore=convert.denormalization(output_moe)
                    save_image(restore[1],'/home/ven073/anju/dmae2/moe_val2.png')
                    loss_value = loss.item()
                    val_loss+=loss_value
                    if not math.isfinite(loss_value):
                        print("Loss is {}, stopping training".format(loss_value))
                        sys.exit(1)
                    loss_value_reduce = misc.all_reduce_mean(loss_value)
                    if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
       
                        epoch_1000x = int((data_iter_step / len(data_loader_val) + epoch) * 1000)
                        log_writer.add_scalar('moe validation loss', loss_value_reduce, epoch_1000x)
         
                    print(f"MOE validation loss  Epoch {epoch + 1}, Loss: {loss_value}")
                    psnr_val_rgb.append(batch_PSNR(restore, samples, False).item())
                average_psnr  = sum(psnr_val_rgb)/len(data_loader_val)
                if average_psnr > best_psnr:
                    best_psnr = average_psnr
                    best_epoch = epoch
                    print("[Ep %d\t PSNR moe: %.4f\t] ----  [best_Ep_MOE  %d Best_PSNR_MOE %.4f] " % (epoch, average_psnr,best_epoch,best_psnr))
                    torch.save({
                    'epoch': epoch, 
                    'model_state_dict': moe_model.module.state_dict()
                            
                    }, f'{args.output_dir}/moe_unfreeze_best{epoch+1}.pth')
                val_loss /= len(data_loader_val)
            #scheduler.step(val_loss)
                if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
                    torch.save({
                'model_state_dict': moe_model.module.state_dict(),
                            
                }, f'{args.output_dir}/moe_unfreeze_{epoch+1}.pth')            
       
    return best_psnr,val_loss


