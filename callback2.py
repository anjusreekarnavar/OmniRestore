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
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        
        self.trace_func = trace_func
    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
          
            self.counter = 0
     
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
