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
from metrics_for_eval import psnr_aggregate,batch_PSNR


#from tensorboardX import SummaryWriter
def save_model_earlystop(model,task,epoch):
    torch.save({
                'model_state_dict': model.module.state_dict(),
                           
             }, f"/scratch3/ven073/decoder_noweights/decoder_{task}_best_{epoch}.pth")




def add_distortions(imgs,args):
    batch_size,_,_,_=imgs.shape
    device=args.device
    noise = torch.randn_like(imgs) * args.sigma
    imgs_noised = imgs + noise
    
    
    lrimage=to_low_resolution(imgs,args.downsampling_factor)

    blur_image=blur_input_image(imgs, args.radius)
    
    
    inpaint_mask=mask_to(imgs,device,mask_root=args.mask_root,mask_type=args.mask_type)
    
    patch_mask=patch_generator2(imgs,device)
   
    return imgs_noised,lrimage,blur_image,inpaint_mask,patch_mask


    
def  train_one_epoch(shared_encoder,decoders,data_loader,data_loader_val,device,optimizers,loss_scaler1,
                     loss_scaler2,loss_scaler3,loss_scaler4,loss_scaler5,
                     epoch,tasks,early_stopping_dict,log_writer,optimizer_encoder,args):

    mask_ratio=args.mask_ratio
    
  
    
    accum_iter = args.accum_iter
    for optimizer in optimizers:
        optimizer.zero_grad()
    optimizer_encoder.zero_grad()
    convert=Conversion()

    for _, decoder in decoders.items():
        decoder.train(True)
    shared_encoder.train(True)

    for data_iter_step, (samples, _) in enumerate(data_loader):
            
            samples = samples.to(device, non_blocking=True)

            imgs_noised,lrimage,blur_image,inpaint_mask,patch_mask=add_distortions(samples,args)
            
       
            for i in range(args.num_tasks):

               

                if tasks[i]=='denoise' :

                    if  not(early_stopping_dict['denoise'].early_stop):
                  
                
            #task1      
                        if data_iter_step % accum_iter == 0:
                            lr_sched.adjust_learning_rate(optimizers[i],optimizer_encoder, data_iter_step / len(data_loader) + epoch, tasks,args)
                        
                        with torch.cuda.amp.autocast():
                            latent,mask,ids_restore=shared_encoder(imgs_noised,mask_ratio)
                            prediction_denoise=decoders[tasks[i]](samples,latent,ids_restore,mask,args)
                          
                        denoise_loss=prediction_denoise[1]
                        denoiseloss_value=denoise_loss.item()
                        if not math.isfinite(denoiseloss_value):
                            print("Loss is {}, stopping training".format(denoiseloss_value))
                            sys.exit(1)
                        denoise_loss /= accum_iter
                        print('epoch',epoch,'denoising training loss',denoiseloss_value)
                        loss_scaler1(denoise_loss,optimizers[i],optimizer_encoder, parameters=list(decoders[tasks[i]].parameters())+list(shared_encoder.parameters()),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
                        if (data_iter_step + 1) % accum_iter == 0:
                            optimizers[i].zero_grad()
                            optimizer_encoder.zero_grad()
                        reduce_denoise = misc.all_reduce_mean(denoiseloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('denoisetraining loss', reduce_denoise, epoch_1000x)
                        torch.cuda.empty_cache()
                        #total_loss_denoise+=denoiseloss_value
                    else:
                        continue
            #task2
                elif tasks[i]=='superresolution':

                    if  not(early_stopping_dict['superresolution'].early_stop):
          
                
                        if data_iter_step % accum_iter == 0:
                            lr_sched.adjust_learning_rate(optimizers[i],optimizer_encoder, data_iter_step / len(data_loader) + epoch, tasks,args)
                        with torch.cuda.amp.autocast():
                            latent,mask,ids_restore=shared_encoder(lrimage,mask_ratio)
                            prediction_superresolve=decoders[tasks[i]](samples,latent,ids_restore,mask,args)
                      
                        super_loss=prediction_superresolve[1]
                        superloss_value=super_loss.item()
                        if not math.isfinite(superloss_value):
                            print("Loss is {}, stopping training".format(superloss_value))
                            sys.exit(1)
                        super_loss /= accum_iter
                        print('epoch',epoch,'superresolution training loss',superloss_value)
                        loss_scaler2(super_loss,optimizers[i],optimizer_encoder, parameters=list(decoders[tasks[i]].parameters())+list(shared_encoder.parameters()),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
                        if (data_iter_step + 1) % accum_iter == 0:
                            optimizers[i].zero_grad()
                            optimizer_encoder.zero_grad()
                        reduce_super = misc.all_reduce_mean(superloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('super_training_loss ', reduce_super, epoch_1000x)
                        torch.cuda.empty_cache()
                    else:
                        continue  
                
                        
            #task3
                elif tasks[i]=='deblurring':

                    if not(early_stopping_dict['deblurring'].early_stop):
                   
                        if data_iter_step % accum_iter == 0:
          
                            lr_sched.adjust_learning_rate(optimizers[i],optimizer_encoder,data_iter_step / len(data_loader) + epoch,tasks, args)
                        with torch.cuda.amp.autocast():
                            latent,mask,ids_restore=shared_encoder(blur_image,mask_ratio)
                            prediction_deblur=decoders[tasks[i]](samples,latent,ids_restore,mask,args)
                            
                        
                        blur_loss=prediction_deblur[1]
                        blurloss_value=blur_loss.item()

                        if not math.isfinite(blurloss_value):
                            print("Loss is {}, stopping training".format(blurloss_value))
                            sys.exit(1)
                        blur_loss /= accum_iter
                        
                        loss_scaler3(blur_loss,optimizers[i],optimizer_encoder, parameters=list(decoders[tasks[i]].module.parameters())+list(shared_encoder.module.parameters()),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
                        if (data_iter_step + 1) % accum_iter == 0:
                            optimizers[i].zero_grad()
                            optimizer_encoder.zero_grad()
                        print('epoch',epoch,'deblurring training loss',blurloss_value)
                        reduce_deblur = misc.all_reduce_mean(blurloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('deblur_trainig_loss ', reduce_deblur, epoch_1000x)
                        torch.cuda.empty_cache()
                    else:
                        continue   
                
            #task4
                elif tasks[i]=='inpainting' :

                    if not(early_stopping_dict['inpainting'].early_stop):
       
              
                        if data_iter_step % accum_iter == 0:
                            lr_sched.adjust_learning_rate(optimizers[i],optimizer_encoder,data_iter_step / len(data_loader) + epoch,tasks, args)
                        with torch.cuda.amp.autocast():
                            latent,mask,ids_restore=shared_encoder(inpaint_mask,mask_ratio)
                            prediction_inpaint=decoders[tasks[i]](samples,latent,ids_restore,mask,args) 
                            
                        inpaint_loss=prediction_inpaint[1]
                        inpaintloss_value=inpaint_loss.item()
                        if not math.isfinite(inpaintloss_value):
                            print("Loss is {}, stopping training".format(inpaintloss_value))
                            sys.exit(1)
                        inpaint_loss /= accum_iter
                      
                        loss_scaler4(inpaint_loss,optimizers[i],optimizer_encoder, parameters=list(decoders[tasks[i]].parameters())+list(shared_encoder.parameters()),clip_grad=2,
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
                        if (data_iter_step + 1) % accum_iter == 0:
                            optimizers[i].zero_grad()
                            optimizer_encoder.zero_grad()
                        print('epoch',epoch,'inpainting training loss', inpaintloss_value)
                        reduce_inpaint = misc.all_reduce_mean(inpaintloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('inpainting_training_loss ', reduce_inpaint, epoch_1000x)
                        torch.cuda.empty_cache()
                    else:
                        continue
    
            #task5
                else:
                     if not(early_stopping_dict['masking'].early_stop):
            
                        if data_iter_step % accum_iter == 0:
                            lr_sched.adjust_learning_rate(optimizers[i], optimizer_encoder,data_iter_step / len(data_loader) + epoch,tasks,args)
                        
                        with torch.cuda.amp.autocast():
                            latent,mask,ids_restore=shared_encoder(patch_mask,mask_ratio)
                            prediction_demask = decoders[tasks[i]](samples,latent,ids_restore,mask,args) 
                       
                        mask_loss=prediction_demask[1]
                        maskloss_value = mask_loss.item()
                        if not math.isfinite(maskloss_value):
                            print("Loss is {}, stopping training".format(maskloss_value))
                            sys.exit(1)
                        mask_loss /= accum_iter
                        
                        loss_scaler5(mask_loss,optimizers[i],optimizer_encoder, parameters=list(decoders[tasks[i]].parameters())+list(shared_encoder.parameters()),clip_grad=2,
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
                        if (data_iter_step + 1) % accum_iter == 0:
                            optimizers[i].zero_grad()
                            optimizer_encoder.zero_grad()
                        print('epoch',epoch,'demasking training loss', maskloss_value)
                        reduce_mask = misc.all_reduce_mean(maskloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('demasking_training_loss ', reduce_mask, epoch_1000x)
                        torch.cuda.empty_cache()
                        #total_loss_demask+=mask_loss 
                     else:
                        continue         
            torch.cuda.synchronize()

    #validation  
    shared_encoder.eval()         
    for _, decoder in decoders.items():
        decoder.eval()

    total_noise_loss=0
    total_blur_loss=0
    total_super_loss=0
    total_inpaint_loss=0
    total_mask_loss=0
    with torch.no_grad():
       
    
        for data_iter_step, (samples, _) in enumerate(data_loader_val):
            
            samples = samples.to(device, non_blocking=True)
            imgs_noised,lrimage,blur_image,inpaint_mask,patch_mask=add_distortions(samples,args)
            
            
            
            for i in range(args.num_tasks):

                if tasks[i]=='denoise':
                    
                    if not(early_stopping_dict['denoise'].early_stop):    
                        
                        
                        with torch.cuda.amp.autocast():        
                            latent,mask,ids_restore=shared_encoder(imgs_noised,mask_ratio)
                            prediction_denoise=decoders[tasks[i]](samples,latent,ids_restore,mask,args)
                                
                        p=prediction_denoise[0]
                        p=convert.denormalization(p)
                        h=p[1]
                        save_image(h,'/home/ven073/anju/denoisedcnn.png')
                        denoise_loss=prediction_denoise[1]
                        denoiseloss_value = denoise_loss.item()
                        total_noise_loss += denoiseloss_value
                        print('epoch',epoch,'denoising validation loss',denoiseloss_value)
                        reduce_denoise = misc.all_reduce_mean(denoiseloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('denoisevalidation loss', reduce_denoise, epoch_1000x)
                        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):  
                           
                            torch.save({
                            'model_state_dict': decoders[tasks[i]].module.state_dict(),
                           
                            }, f'/scratch3/ven073/decoder_noweights/decoder_denoise_epoch_{epoch+1}.pth')
                        torch.cuda.empty_cache()

                    else: 

                        save_model_earlystop(decoders[tasks[i]],tasks[i],epoch)
                        continue
                      
            #task2
                elif tasks[i]=='superresolution':
          
                    if  not(early_stopping_dict['superresolution'].early_stop):
                 
                        with torch.cuda.amp.autocast():
                         
                            latent,mask,ids_restore=shared_encoder(lrimage,mask_ratio)
                            prediction_superresolve=decoders[tasks[i]](samples,latent,ids_restore,mask,args)
                        p=prediction_superresolve[0]
                        p=convert.denormalization(p)
                        save_image(p[1],'/home/ven073/anju/superrcnn.png')
              
                        super_loss=prediction_superresolve[1]
                        superloss_value = super_loss.item()
                        total_super_loss+= superloss_value
                        print('epoch',epoch,'superresolution validation loss',superloss_value)
                        reduce_super = misc.all_reduce_mean(superloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('super_validation_loss ', reduce_super, epoch_1000x)
                        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):  
                          
                            torch.save({
                                'model_state_dict': decoders[tasks[i]].module.state_dict(),
                            
                            }, f'/scratch3/ven073/decoder_noweights/decoder_super_epoch_{epoch+1}.pth')
                        torch.cuda.empty_cache()
                    else:
                        save_model_earlystop(decoders[tasks[i]],tasks[i],epoch)
                        continue

            #task3
                elif tasks[i]=='deblurring':
                    if not(early_stopping_dict['deblurring'].early_stop):
                  
    
                        with torch.cuda.amp.autocast():
                            
                            latent,mask,ids_restore=shared_encoder(blur_image,mask_ratio)
                            prediction_deblur=decoders[tasks[i]](samples,latent,ids_restore,mask,args)
                        p=prediction_deblur[0]
                        p=convert.denormalization(p)
                        save_image(p[0],'/home/ven073/anju/deblcnn.png')
              
                        blur_loss=prediction_deblur[1]
                        blurloss_value = blur_loss.item()
                        total_blur_loss+= blurloss_value
                        print('epoch',epoch,'deblurring validation loss',blurloss_value)
                        reduce_deblur = misc.all_reduce_mean(blurloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('deblur_validation_loss ', reduce_deblur, epoch_1000x)
                        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):  
                            
                            torch.save({
                            'model_state_dict': decoders[tasks[i]].module.state_dict(),
                          
                            }, f'/scratch3/ven073/decoder_noweights/decoder_deblur_epoch_{epoch+1}.pth')
                        torch.cuda.empty_cache()
                    else:
                        save_model_earlystop(decoders[tasks[i]],tasks[i],epoch)
                        continue
                        
                        
                        
            #task4
                elif tasks[i]=='inpainting':

                    if not(early_stopping_dict['inpainting'].early_stop):

                        with torch.cuda.amp.autocast():
                            latent,mask,ids_restore=shared_encoder(inpaint_mask,mask_ratio)
                            prediction_inpaint=decoders[tasks[i]](samples,latent,ids_restore,mask,args) 
                        p=prediction_inpaint[0]
                        p=convert.denormalization(p)
                        save_image(p[1],'/home/ven073/anju/inpaicnn.png')
                        inpaint_loss=prediction_inpaint[1]
                        inpaintloss_value = inpaint_loss.item()
                        total_inpaint_loss+=inpaintloss_value
                        print('epoch',epoch,'inpainting validation loss', inpaintloss_value)
                        reduce_inpaint = misc.all_reduce_mean(inpaintloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('inpainting_validation_loss ', reduce_inpaint, epoch_1000x)
                        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):  
                            
                            torch.save({
                            'model_state_dict': decoders[tasks[i]].module.state_dict(),
                            
                            }, f'/scratch3/ven073/decoder_noweights/decoder_inpainting_epoch_{epoch+1}.pth')
                        torch.cuda.empty_cache()
                    else:
                        save_model_earlystop(decoders[tasks[i]],tasks[i],epoch)
                        continue
            #task5
                else:

                    if not(early_stopping_dict['masking'].early_stop):
            
                        with torch.cuda.amp.autocast():
                            latent,mask,ids_restore=shared_encoder(patch_mask,mask_ratio)
                            prediction_demask=decoders[tasks[i]](samples,latent,ids_restore,mask,args) 
                        p=prediction_demask[0]
                        p=convert.denormalization(p)
                        save_image(p[1],'/home/ven073/anju/demascnn.png')
                        mask_loss=prediction_demask[1]
                        maskloss_value = mask_loss.item()
                        total_mask_loss+=maskloss_value
                        print('epoch',epoch,'masking validation loss', maskloss_value)
                        reduce_mask = misc.all_reduce_mean(maskloss_value)
                        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                            log_writer.add_scalar('masking_validation_loss ', reduce_mask, epoch_1000x)
                        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):  
                           
                            torch.save({
                            'model_state_dict': decoders[tasks[i]].module.state_dict(),
                            
                            }, f'/scratch3/ven073/decoder_noweights/decoder_masking_epoch_{epoch+1}.pth')
                        torch.cuda.empty_cache()

                    else:

                        save_model_earlystop(decoders[tasks[i]],tasks[i],epoch)
                        continue

            torch.cuda.synchronize()

        noise_avg   = total_noise_loss/len(data_loader_val)
        blur_avg   =  total_blur_loss/ len(data_loader_val)
        super_avg = total_super_loss/len(data_loader_val)
        inpaint_avg = total_inpaint_loss/len(data_loader_val)
        mask_avg = total_mask_loss/len(data_loader_val)

        task_avg = {'denoise':noise_avg,'superresolution':super_avg,'deblurring':blur_avg,'inpainting':inpaint_avg,'masking':mask_avg}
        
        for task in tasks:
             early_stopping_dict[task](task_avg[task])
        args.all_stopped = all(earlystop.early_stop for earlystop in early_stopping_dict.values())
      
    return shared_encoder

