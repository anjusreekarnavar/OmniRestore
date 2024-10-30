# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(optimizer_dec,optimizer_enc, epoch,task, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    
    if epoch < args.warmup_epochs:
        if task=='denoise':
            lr=args.dlr*epoch / args.warmup_epochs 
        elif task=='superresolution':
            lr=args.slr*epoch / args.warmup_epochs 
        elif task=='deblurring':
            lr=args.blur*epoch / args.warmup_epochs 
        elif task=='inpainting':
            lr=args.ilr*epoch / args.warmup_epochs
        else:
            lr=args.mlr*epoch / args.warmup_epochs
       
        lr_2=args.elr*epoch / args.warmup_epochs 
       
    else:
        if task=='denoise':
            lr = args.min_lr + (args.dlr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        elif task=='superresolution':
            lr = args.min_lr + (args.slr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        elif task=='deblurring':
            lr = args.min_lr + (args.blur - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        elif task=='inpainting':
            lr = args.min_lr + (args.ilr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        else:
            lr = args.min_lr + (args.mlr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        
        
        lr_2 = args.min_lr + (args.elr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
       
    for param_group in optimizer_enc.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr_2 * param_group["lr_scale"]
        else:
            param_group["lr"] = lr_2
    for param_group in optimizer_dec.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr