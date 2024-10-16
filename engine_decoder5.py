import math
import sys
from typing import Iterable
from torch import nn
import torch
import random
from torchvision.utils import save_image
import util.misc as misc
from inpaint_mask_generator import generate_mask,patch_generator
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from inpaint_mask_generator import generate_mask,patch_generator2,mask_to
import util.lr_sched as lr_sched
from torch.utils.tensorboard import SummaryWriter
from callback import callback_For_Threshold
from callback import EarlyStopping
import torchvision.transforms as transforms
from augmentations import converto_low_resolution,blur_input_image,to_low_resolution
from temporary import Conversion
from torchvision import models
from perceptualloss import LossNetwork
import PIL

#from tensorboardX import SummaryWriter





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



    
def  train_one_epoch(model,data_loader_train,data_loader_val,device,optimizer,loss_scaler,epoch,log_writer,args):
    mask_ratio=args.mask_ratio
   
    accum_iter = args.accum_iter
    
    optimizer.zero_grad()

    convert=Conversion()


    for data_iter_step, data_train in enumerate((data_loader_train), 0):
            
            clean_img = data_train[0].to(device, non_blocking=True)
            distorted=data_train[1].to(device, non_blocking=True)
            clean_img = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(clean_img)
            distorted = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(distorted)
            #task1      
            save_image(clean_img[0],'/home/ven073/anju/dmae2/original.jpg')
            save_image(distorted[0],'/home/ven073/anju/dmae2/distorted.jpg')
            if data_iter_step % accum_iter == 0:
                    lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_train) + epoch, args)
                        
            with torch.cuda.amp.autocast():
                
                output,_=model(clean_img,distorted,mask_ratio)
            prediction=output[0]    
            loss=output[1]     
            p=convert.unpatchify(prediction)
            p=convert.denormalization(p)
                       
            denoise_loss=loss
            denoiseloss_value=denoise_loss.item()
            if not math.isfinite(denoiseloss_value):
                print("Loss is {}, stopping training".format(denoiseloss_value))
                sys.exit(1)
            denoise_loss /= accum_iter
            print('epoch',epoch,'denoising training loss',denoiseloss_value)
            loss_scaler(denoise_loss,optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                    optimizer.zero_grad()
            reduce_denoise = misc.all_reduce_mean(denoiseloss_value)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                epoch_1000x = int((data_iter_step / len(data_loader_train) + epoch) * 1000)
                log_writer.add_scalar('denoisetraining loss', reduce_denoise, epoch_1000x)
                        #total_loss_denoise+=denoiseloss_value
                
    model.eval()
    with torch.no_grad():
        for data_iter_step, data_val in enumerate((data_loader_val), 0):
            
            clean_img = data_val[0].to(device, non_blocking=True)
            distorted=data_val[1].to(device, non_blocking=True)
            clean_img = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(clean_img)
            distorted = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(distorted)
            
                        
            with torch.cuda.amp.autocast():        
                output,_=model(clean_img,distorted,mask_ratio)
            prediction=output[0]    
            loss=output[1]     
            p=convert.unpatchify(prediction)
            p=convert.denormalization(p)               
                    
            denoise_loss=loss
            denoiseloss_value = denoise_loss.item()
                        
            print('epoch',epoch,'denoising validation loss',denoiseloss_value)
            reduce_denoise = misc.all_reduce_mean(denoiseloss_value)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                    epoch_1000x = int((data_iter_step / len(data_loader_val) + epoch) * 1000)
                    log_writer.add_scalar('denoisevalidation loss', reduce_denoise, epoch_1000x)
            if args.output_dir and (epoch % 100 == 0 or epoch + 1 == args.epochs):
                           
                torch.save({
                'model_state_dict': model.module.decoder5.state_dict(),
                           
                    }, f'/scratch3/ven073/decoder5/decoder_denoise_epoch_{epoch+1}.pth')
                       
                      
           
      
    return model

