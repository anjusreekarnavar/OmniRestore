import math
import sys
from typing import Iterable

import torch
import torch.nn as nn

#import util.misc as misc
import util.lr_sched as lr_sched

class CustomCallback():
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, epoch, loss):
        if loss < self.threshold:
            print(f"Training stopped at epoch {epoch} because loss {loss:.4f} is below the threshold {self.threshold}.")
            
            return True
          # Returning True stops the training
        return False
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=10, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    def __call__(self, val_loss):
        
    
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        else:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
     
def callback_For_Threshold(epoch,denoise_loss,mask_loss,super_loss,blur_loss,inpaint_loss,device,args):
    callback = CustomCallback(threshold=0.000001)
    loss=torch.zeros(1,requires_grad=True)
    loss=loss.to(args.device)
    if callback(epoch, denoise_loss):
            
            if callback(epoch, mask_loss):
                loss_value = (super_loss+blur_loss+inpaint_loss)/3
                loss= super_loss+blur_loss+inpaint_loss
                loss=loss/3
                args.mask_flag=1
            elif callback(epoch, super_loss):
                loss_value = (mask_loss+blur_loss+inpaint_loss)/3
                loss= mask_loss+blur_loss+inpaint_loss
                loss=loss/3
                args.super_flag=1
            elif callback(epoch, blur_loss):
                loss_value = (super_loss+mask_loss+inpaint_loss)/3
                loss= mask_loss+super_loss+inpaint_loss
                loss=loss/3
                args.blur_flag=1
            elif callback(epoch, inpaint_loss):
                loss_value = (super_loss+blur_loss+mask_loss)/3
                loss= mask_loss+super_loss+blur_loss
                loss=loss/3
                args.inpaint_flag=1
            else:
                loss_value = (mask_loss+super_loss+blur_loss+inpaint_loss)/4
                loss= mask_loss+super_loss+blur_loss+inpaint_loss
                loss /= 4
                args.denoise_flag=1
    elif callback(epoch, mask_loss):
            if callback(epoch, denoise_loss):
                loss_value = (super_loss+blur_loss+inpaint_loss)/3
                loss= super_loss+blur_loss+inpaint_loss
                loss=loss/3
                args.denoise_flag=1
            elif callback(epoch, super_loss):
                loss_value = (denoise_loss+blur_loss+inpaint_loss)/3
                loss= denoise_loss+blur_loss+inpaint_loss
                loss=loss/3
                args.super_flag=1
            elif callback(epoch, blur_loss):
                loss_value = (super_loss+denoise_loss+inpaint_loss)/3
                loss= denoise_loss+super_loss+inpaint_loss
                loss=loss/3
                args.blur_flag=1
            elif callback(epoch, inpaint_loss):
                loss_value = (super_loss+blur_loss+denoise_loss)/3
                loss= denoise_loss+super_loss+blur_loss
                loss=loss/3
                args.inpaint_flag=1
            else:
                loss_value = (denoise_loss+super_loss+blur_loss+inpaint_loss)/4
                loss=denoise_loss+super_loss+blur_loss+inpaint_loss
                loss /= 4
                args.mask_flag=1
    elif callback(epoch, super_loss):
            if callback(epoch, denoise_loss):
                loss_value = (mask_loss+blur_loss+inpaint_loss)/3
                loss= mask_loss+blur_loss+inpaint_loss
                loss=loss/3
                args.denoise_flag=1
            elif callback(epoch, mask_loss):
                loss_value = (denoise_loss+blur_loss+inpaint_loss)/3
                loss= denoise_loss+blur_loss+inpaint_loss
                loss=loss/3
                args.mask_flag=1
            elif callback(epoch, blur_loss):
                loss_value = (denoise_loss+blur_loss+inpaint_loss)/3
                loss= mask_loss+denoise_loss+inpaint_loss
                loss=loss/3
                args.blur_flag=1
            elif callback(epoch, inpaint_loss):
                loss_value = (denoise_loss+blur_loss+mask_loss)/3
                loss= mask_loss+denoise_loss+blur_loss
                loss=loss/3
                args.inpaint_flag=1
            else:
                loss_value = (mask_loss+denoise_loss+blur_loss+inpaint_loss)/4
                loss =mask_loss+denoise_loss+blur_loss+inpaint_loss
                loss /= 4
                args.super_flag=1
    elif callback(epoch, blur_loss):
            if callback(epoch, denoise_loss):
                loss_value = (mask_loss+super_loss+inpaint_loss)/3
                loss= super_loss+mask_loss+inpaint_loss
                loss=loss/3
                args.denoise_flag=1
            elif callback(epoch, mask_loss):
                loss_value = (denoise_loss+super_loss+inpaint_loss)/3
                loss= denoise_loss+blur_loss+inpaint_loss
                loss=loss/3
                args.mask_flag=1
            elif callback(epoch, super_loss):
                loss_value = (mask_loss+denoise_loss+inpaint_loss)/3
                loss= mask_loss+denoise_loss+inpaint_loss
                loss=loss/3
                args.super_flag=1
            elif callback(epoch, inpaint_loss):
                loss_value = (mask_loss+super_loss+denoise_loss)/3
                loss= mask_loss+super_loss+denoise_loss
                loss=loss/3
                args.inpaint_flag=1
            else:
                loss_value = (mask_loss+denoise_loss+super_loss+inpaint_loss)/4
                loss =mask_loss+denoise_loss+super_loss+inpaint_loss
                loss /= 4
                args.blur_flag=1
    elif callback(epoch, inpaint_loss):
            if callback(epoch, denoise_loss):
                loss_value = (mask_loss+super_loss+blur_loss)/3
                loss= super_loss+blur_loss+mask_loss
                loss=loss/3
                args.denoise_flag=1
            elif callback(epoch, mask_loss):
                loss_value = (denoise_loss+super_loss+blur_loss)/3
                loss=denoise_loss+blur_loss+blur_loss
                loss=loss/3
                args.mask_flag=1
            elif callback(epoch, super_loss):
                loss_value = (mask_loss+denoise_loss+blur_loss)/3
                loss= mask_loss+blur_loss+denoise_loss
                loss=loss/3
                args.super_flag=1
            elif callback(epoch, blur_loss):
                loss_value = (mask_loss+super_loss+denoise_loss)/3
                loss= mask_loss+super_loss+denoise_loss
                loss=loss/3
                args.blur_flag=1
            else:
                loss_value = (mask_loss+denoise_loss+super_loss+blur_loss)/4
                loss =mask_loss+denoise_loss+super_loss+blur_loss
                loss /= 4
                args.inpaint_flag=1
    elif callback(epoch, inpaint_loss):
            if callback(epoch, denoise_loss):
                loss_value = (mask_loss+super_loss+blur_loss)/3
                loss= super_loss+blur_loss+mask_loss
                loss=loss/3
                args.denoise_flag=1
            elif callback(epoch, mask_loss):
                loss_value = (denoise_loss+super_loss+blur_loss)/3
                loss=denoise_loss+blur_loss+blur_loss
                loss=loss/3
                args.mask_flag=1
            elif callback(epoch, super_loss):
                loss_value = (mask_loss+denoise_loss+blur_loss)/3
                loss= mask_loss+blur_loss+denoise_loss
                loss=loss/3
                args.super_flag=1
            elif callback(epoch, blur_loss):
                loss_value = (mask_loss+super_loss+denoise_loss)/3
                loss= mask_loss+super_loss+denoise_loss
                loss=loss/3
                args.blur_flag=1
            else:
                loss_value = (mask_loss+denoise_loss+super_loss+blur_loss)/4
                loss =mask_loss+denoise_loss+super_loss+blur_loss
                loss /= 4
                args.inpaint_flag=1
    elif callback(epoch, denoise_loss) and callback(epoch, super_loss):
            if callback(epoch, mask_loss):
                loss_value = (inpaint_loss+blur_loss)/2
                loss=blur_loss+inpaint_loss
                loss=loss/2
                args.mask_flag=1
            
            elif callback(epoch, inpaint_loss):
                loss_value = (mask_loss+blur_loss)/2
                loss=blur_loss+mask_loss
                loss=loss/2
                args.inpaint_flag=1
            elif callback(epoch, blur_loss):
                loss_value = (mask_loss+inpaint_loss)/2
                loss=mask_loss+inpaint_loss
                loss=loss/2
                args.blur_flag=1
            else: 
                loss_value = (mask_loss+inpaint_loss+blur_loss)/3
                loss=mask_loss+inpaint_loss+blur_loss
                loss=loss/3
                args.denoise_flag=1
                args.super_flag=1
    elif callback(epoch, mask_loss) and callback(epoch, blur_loss):
            if callback(epoch, denoise_loss):
                loss_value = (inpaint_loss+super_loss)/2
                loss=blur_loss+inpaint_loss
                loss=loss/2
                args.denoise_flag=1
            
            elif callback(epoch, inpaint_loss):
                loss_value = (denoise_loss+super_loss)/2
                loss=denoise_loss+super_loss
                loss=loss/2
                args.inpaint_flag=1
            elif callback(epoch, super_loss):
                loss_value = (denoise_loss+inpaint_loss)/2
                loss=denoise_loss+inpaint_loss
                loss=loss/2
                args.super_flag=1
            else: 
                loss_value = (denoise_loss+inpaint_loss+super_loss)/3
                loss=mask_loss+inpaint_loss+blur_loss
                loss=loss/3
                args.mask_flag=1
                args.blur_flag=1
    elif callback(epoch, inpaint_loss) and callback(epoch, denoise_loss):
            if callback(epoch, super_loss):
                loss_value = (mask_loss+blur_loss)/2
                loss=mask_loss+blur_loss
                loss=loss/2
                args.super_flag=1
            
            elif callback(epoch, mask_loss):
                loss_value = (blur_loss+super_loss)/2
                loss=blur_loss+super_loss
                loss=loss/2
                args.mask_flag=1
            elif callback(epoch, blur_loss):
                loss_value = (super_loss+mask_loss)/2
                loss=super_loss+mask_loss
                loss=loss/2
                args.blur_flag=1
            else: 
                loss_value = (mask_loss+blur_loss+super_loss)/3
                loss=mask_loss+blur_loss+super_loss
                loss=loss/3
                args.denoise_flag=1
                args.inpaint_flag=1
    else:
            loss_value = (mask_loss.item()+denoise_loss.item()+super_loss.item()+blur_loss.item()+inpaint_loss.item())/5
            loss =mask_loss+denoise_loss+super_loss+blur_loss+inpaint_loss
            
    return loss,loss_value
