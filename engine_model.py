import math
import sys
from typing import Iterable
from torch import nn
import torch
import random
from torchvision.utils import save_image
import util.misc as misc
from inpaint_mask_generator import patch_generator2,mask_to
import util.lr_sched as lr_sched
from callback import EarlyStopping
import torchvision.transforms as transforms
from augmentations import blur_input_image,to_low_resolution
from temporary import Conversion
import PIL

#from tensorboardX import SummaryWriter



def early_stop(total_loss,length):
    early_stopping = EarlyStopping()
    avg_loss=total_loss/length
    early_stopping(avg_loss)
    return early_stopping.early_stop


def add_distortions(imgs,args):
    batch_size,_,_,_=imgs.shape
    device=args.device
    noise = torch.randn_like(imgs) * args.sigma
    imgs_noised = imgs + noise
    imgs_noised = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(imgs_noised)
    
    lrimage=to_low_resolution(imgs,args.downsampling_factor)

    blur_image=blur_input_image(imgs, args.radius)
    blur_image = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(blur_image)
    
    inpaint_mask=mask_to(imgs,device,mask_root=args.mask_root,mask_type=args.mask_type)
    
    blended_image = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(inpaint_mask)
    lrimage = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(lrimage)
    patch_mask=patch_generator2(imgs,device)
    patch_mask = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(patch_mask)
    return imgs_noised,lrimage,blur_image,blended_image,patch_mask


    
def  train_one_epoch(shared_encoder,decoders,data_loader,data_loader_val,device,optimizers,loss_scaler1,loss_scaler2,loss_scaler3,loss_scaler4,loss_scaler5,epoch,flags,log_writer,optimizer_encoder,args):
    mask_ratio=args.mask_ratio
    
    
    noise_flag=flags[0]

    super_flag=flags[1]
   
    blur_flag=flags[2]
        
    inpaint_flag=flags[3]
    mask_flag=flags[4]
    
    accum_iter = args.accum_iter
    for optimizer in optimizers:
        optimizer.zero_grad()
    optimizer_encoder.zero_grad()
    
   
    num_tasks=5
   
    convert=Conversion()
    
    tasks=['denoise','superresolution','deblurring','inpainting','masking']
    for decoder in decoders:
        decoder.train(True)
    shared_encoder.train(True)

    for data_iter_step, (samples, _) in enumerate(data_loader):
            
            samples = samples.to(device, non_blocking=True)
            imgs_noised,lrimage,blur_image,inpaint_mask,patch_mask=add_distortions(samples,args)
            samples = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(samples)
       
            for i in range(5):

               

                if tasks[i]=='denoise':
                    if noise_flag==0:
                
            #task1      
                        if data_iter_step % accum_iter == 0:
                            lr_sched.adjust_learning_rate(optimizers[i],optimizer_encoder, data_iter_step / len(data_loader) + epoch, tasks,args)
                        
                        with torch.cuda.amp.autocast():
                            latent,mask,ids_restore=shared_encoder(imgs_noised,mask_ratio)
                            prediction_denoise=decoders[i](samples,latent,ids_restore,mask,args)
                          
                        denoise_loss=prediction_denoise[1]
                        denoiseloss_value=denoise_loss.item()
                        if not math.isfinite(denoiseloss_value):
                            print("Loss is {}, stopping training".format(denoiseloss_value))
                            sys.exit(1)
                        denoise_loss /= accum_iter
                        print('epoch',epoch,'denoising training loss',denoiseloss_value)
                        loss_scaler1(denoise_loss,optimizers[i],optimizer_encoder, parameters=list(decoders[i].parameters())+list(shared_encoder.parameters()),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
                        if (data_iter_step + 1) % accum_iter == 0:
                            optimizers[i].zero_grad()
                            optimizer_encoder.zero_grad()
                        reduce_denoise = misc.all_reduce_mean(denoiseloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('denoisetraining loss', reduce_denoise, epoch_1000x)
                        #total_loss_denoise+=denoiseloss_value
                
            #task2
                elif tasks[i]=='superresolution':
          
                    if super_flag==0:
                        if data_iter_step % accum_iter == 0:
                            lr_sched.adjust_learning_rate(optimizers[i],optimizer_encoder, data_iter_step / len(data_loader) + epoch, tasks,args)
                        with torch.cuda.amp.autocast():
                            latent,mask,ids_restore=shared_encoder(lrimage,mask_ratio)
                            prediction_superresolve=decoders[i](samples,latent,ids_restore,mask,args)
                      
                        super_loss=prediction_superresolve[1]
                        superloss_value=super_loss.item()
                        if not math.isfinite(superloss_value):
                            print("Loss is {}, stopping training".format(superloss_value))
                            sys.exit(1)
                        super_loss /= accum_iter
                        print('epoch',epoch,'superresolution training loss',superloss_value)
                        loss_scaler2(super_loss,optimizers[i],optimizer_encoder, parameters=list(decoders[i].parameters())+list(shared_encoder.parameters()),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
                        if (data_iter_step + 1) % accum_iter == 0:
                            optimizers[i].zero_grad()
                            optimizer_encoder.zero_grad()
                        reduce_super = misc.all_reduce_mean(superloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('super_training_loss ', reduce_super, epoch_1000x)
                       
                
                        
            #task3
                elif tasks[i]=='deblurring':
                    if blur_flag==0:
                        if data_iter_step % accum_iter == 0:
          
                            lr_sched.adjust_learning_rate(optimizers[i],optimizer_encoder,data_iter_step / len(data_loader) + epoch,tasks, args)
                        with torch.cuda.amp.autocast():
                            latent,mask,ids_restore=shared_encoder(blur_image,mask_ratio)
                            prediction_deblur=decoders[i](samples,latent,ids_restore,mask,args)
                            
                        
                        blur_loss=prediction_deblur[1]
                        blurloss_value=blur_loss.item()

                        if not math.isfinite(blurloss_value):
                            print("Loss is {}, stopping training".format(blurloss_value))
                            sys.exit(1)
                        blur_loss /= accum_iter
                        
                        loss_scaler3(blur_loss,optimizers[i],optimizer_encoder, parameters=list(decoders[i].module.parameters())+list(shared_encoder.module.parameters()),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
                        if (data_iter_step + 1) % accum_iter == 0:
                            optimizers[i].zero_grad()
                            optimizer_encoder.zero_grad()
                        print('epoch',epoch,'deblurring training loss',blurloss_value)
                        reduce_deblur = misc.all_reduce_mean(blurloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('deblur_trainig_loss ', reduce_deblur, epoch_1000x)
                        
                
            #task4
                elif tasks[i]=='inpainting':
       
                    if inpaint_flag==0:
                        if data_iter_step % accum_iter == 0:
                            lr_sched.adjust_learning_rate(optimizers[i],optimizer_encoder,data_iter_step / len(data_loader) + epoch,tasks, args)
                        with torch.cuda.amp.autocast():
                            latent,mask,ids_restore=shared_encoder(inpaint_mask,mask_ratio)
                            prediction_inpaint=decoders[i](samples,latent,ids_restore,mask,args) 
                            
                        inpaint_loss=prediction_inpaint[1]
                        inpaintloss_value=inpaint_loss.item()
                        if not math.isfinite(inpaintloss_value):
                            print("Loss is {}, stopping training".format(inpaintloss_value))
                            sys.exit(1)
                        inpaint_loss /= accum_iter
                      
                        loss_scaler4(inpaint_loss,optimizers[i],optimizer_encoder, parameters=list(decoders[i].parameters())+list(shared_encoder.parameters()),clip_grad=2,
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
                        if (data_iter_step + 1) % accum_iter == 0:
                            optimizers[i].zero_grad()
                            optimizer_encoder.zero_grad()
                        print('epoch',epoch,'inpainting training loss', inpaintloss_value)
                        reduce_inpaint = misc.all_reduce_mean(inpaintloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('inpainting_training_loss ', reduce_inpaint, epoch_1000x)
    
            #task5
                else:
            
                    if mask_flag==0:
                        if data_iter_step % accum_iter == 0:
                            lr_sched.adjust_learning_rate(optimizers[i], optimizer_encoder,data_iter_step / len(data_loader) + epoch,tasks,args)
                        
                        with torch.cuda.amp.autocast():
                            latent,mask,ids_restore=shared_encoder(patch_mask,mask_ratio)
                            prediction_demask=decoders[i](samples,latent,ids_restore,mask,args) 
                       
                        mask_loss=prediction_demask[1]
                        maskloss_value = mask_loss.item()
                        if not math.isfinite(maskloss_value):
                            print("Loss is {}, stopping training".format(maskloss_value))
                            sys.exit(1)
                        mask_loss /= accum_iter
                        
                        loss_scaler5(mask_loss,optimizers[i],optimizer_encoder, parameters=list(decoders[i].parameters())+list(shared_encoder.parameters()),clip_grad=2,
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
                        if (data_iter_step + 1) % accum_iter == 0:
                            optimizers[i].zero_grad()
                            optimizer_encoder.zero_grad()
                        print('epoch',epoch,'demasking training loss', maskloss_value)
                        reduce_mask = misc.all_reduce_mean(maskloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('demasking_training_loss ', reduce_mask, epoch_1000x)
                        #total_loss_demask+=mask_loss          
            torch.cuda.synchronize()

    #validation  
    shared_encoder.eval()         
    for decoder in decoders:
        decoder.eval()

    val_loss_denoise = 0.0
    val_loss_deblur = 0.0
    val_loss_super = 0.0
    val_loss_demask = 0.0
    val_loss_inpaint = 0.0
    with torch.no_grad():
        for data_iter_step, (samples, _) in enumerate(data_loader_val):
            
            samples = samples.to(device, non_blocking=True)
            imgs_noised,lrimage,blur_image,inpaint_mask,patch_mask=add_distortions(samples,args)
            samples = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(samples)
            
            
            for i in range(num_tasks):
                if tasks[i]=='denoise':
                    if noise_flag==0:
            #task1      
                        
                        
                        with torch.cuda.amp.autocast():        
                            latent,mask,ids_restore=shared_encoder(imgs_noised,mask_ratio)
                            prediction_denoise=decoders[i](samples,latent,ids_restore,mask,args)
                                
                        p=prediction_denoise[0]
                        p=convert.denormalization(p)
                        h=p[1]
                        save_image(h,'/home/ven073/anju/denoised2.png')
                        denoise_loss=prediction_denoise[1]
                        denoiseloss_value = denoise_loss.item()
                        
                        print('epoch',epoch,'denoising validation loss',denoiseloss_value)
                        reduce_denoise = misc.all_reduce_mean(denoiseloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('denoisevalidation loss', reduce_denoise, epoch_1000x)
                if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):  
                           
                        torch.save({
                            'model_state_dict': decoders[i].module.state_dict(),
                           
                            }, f'/scratch3/ven073/decoder_shared/decoder_denoise_epoch_{epoch+1}.pth')
                if(early_stop(val_loss_denoise,len(data_loader_val))):
                            noise_flag=1
                            flags.append(noise_flag)
                      
            #task2
                elif tasks[i]=='superresolution':
          
                    if super_flag==0:
                        with torch.cuda.amp.autocast():
                         
                            latent,mask,ids_restore=shared_encoder(lrimage,mask_ratio)
                            prediction_superresolve=decoders[i](samples,latent,ids_restore,mask,args)
                        p=prediction_superresolve[0]
                        p=convert.denormalization(p)
                        save_image(p[1],'/home/ven073/anju/superresolved2.png')
              
                        super_loss=prediction_superresolve[1]
                        superloss_value = super_loss.item()
                        val_loss_super += super_loss.item()
                        print('epoch',epoch,'superresolution validation loss',superloss_value)
                        reduce_super = misc.all_reduce_mean(superloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('super_validation_loss ', reduce_super, epoch_1000x)
                if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):  
                          
                        torch.save({
                            'model_state_dict': decoders[i].module.state_dict(),
                            
                            }, f'/scratch3/ven073/decoder_shared/decoder_super_epoch_{epoch+1}.pth')
                if(early_stop(val_loss_super,len(data_loader_val))):
                            super_flag=1
                            flags.append(super_flag)
                       

            #task3
                elif tasks[i]=='deblurring':
                    if blur_flag==0:
    
                        with torch.cuda.amp.autocast():
                            
                            latent,mask,ids_restore=shared_encoder(blur_image,mask_ratio)
                            prediction_deblur=decoders[i](samples,latent,ids_restore,mask,args)
                        p=prediction_deblur[0]
                        p=convert.denormalization(p)
                        save_image(p[0],'/home/ven073/anju/deblurred2.png')
              
                        blur_loss=prediction_deblur[1]
                        blurloss_value = blur_loss.item()
                        val_loss_deblur += blur_loss.item()
                        print('epoch',epoch,'deblurring validation loss',blurloss_value)
                        reduce_deblur = misc.all_reduce_mean(blurloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('deblur_validation_loss ', reduce_deblur, epoch_1000x)
                if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):  
                            
                        torch.save({
                            'model_state_dict': decoders[i].module.state_dict(),
                          
                            }, f'/scratch3/ven073/decoder_shared/decoder_deblur_epoch_{epoch+1}.pth')
                if(early_stop(val_loss_deblur,len(data_loader_val))):
                            blur_flag=1
                            flags.append(blur_flag)
                        
                        
                        
            #task4
                elif tasks[i]=='inpainting':
       
                    if inpaint_flag==0:
                        with torch.cuda.amp.autocast():
                            latent,mask,ids_restore=shared_encoder(inpaint_mask,mask_ratio)
                            prediction_inpaint=decoders[i](samples,latent,ids_restore,mask,args) 
                        p=prediction_inpaint[0]
                        p=convert.denormalization(p)
                        save_image(p[1],'/home/ven073/anju/inpainted2.png')
                        inpaint_loss=prediction_inpaint[1]
                        inpaintloss_value = inpaint_loss.item()
                        val_loss_inpaint += inpaint_loss.item()
                        print('epoch',epoch,'inpainting validation loss', inpaintloss_value)
                        reduce_inpaint = misc.all_reduce_mean(inpaintloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('inpainting_validation_loss ', reduce_inpaint, epoch_1000x)
                if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):  
                            
                    torch.save({
                            'model_state_dict': decoders[i].module.state_dict(),
                            
                            }, f'/scratch3/ven073/decoder_shared/decoder_inpainting_epoch_{epoch+1}.pth')
                if(early_stop(val_loss_inpaint ,len(data_loader_val))):
                                inpaint_flag=1
                                flags.append(inpaint_flag)
            #task5
                else:
            
                    if mask_flag==0:
                        with torch.cuda.amp.autocast():
                            latent,mask,ids_restore=shared_encoder(patch_mask,mask_ratio)
                            prediction_demask=decoders[i](samples,latent,ids_restore,mask,args) 
                        p=prediction_demask[0]
                        p=convert.denormalization(p)
                        save_image(p[1],'/home/ven073/anju/demasked2.png')
                        mask_loss=prediction_demask[1]
                        maskloss_value = mask_loss.item()
                        val_loss_demask += mask_loss.item()
                        print('epoch',epoch,'masking validation loss', maskloss_value)
                        reduce_mask = misc.all_reduce_mean(maskloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('masking_validation_loss ', reduce_mask, epoch_1000x)
                if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):  
                           
                         torch.save({
                            'model_state_dict': decoders[i].module.state_dict(),
                            
                            }, f'/scratch3/ven073/decoder_shared/decoder_masking_epoch_{epoch+1}.pth')
                if(early_stop(val_loss_demask ,len(data_loader_val))):
                            mask_flag=1
                            flags.append(mask_flag)
    
      
    return shared_encoder,flags

