import argparse
import datetime
import json
import numpy as np
import os
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter, ImageOps
from util import misc
import timm.optim.optim_factory as optim_factory


def create_optimizer(args, model, eff_batch_size):

    optimizers = {}

    if args.noiselr is None:  # only base_lr is specified
        args.noiselr = args.blr * eff_batch_size / 256

    if args.blurlr is None:
        args.blurlr = args.blr * eff_batch_size / 256

    if args.superlr is None:
        args.superlr = args.blr * eff_batch_size / 256

    if args.inpaintlr is None:
        args.inpaintlr = args.mlr * eff_batch_size / 256

    if args.masklr is None:
        args.masklr = args.mlr * eff_batch_size / 256

    param_enc = optim_factory.add_weight_decay(model.encoder, args.weight_decay)

    # List of decoders with corresponding learning rates and task names
    decoders = {
        "denoising": (model.noise_decoder, args.noiselr),
        "deblurring": (model.blur_decoder, args.blurlr),
        "super_resolution": (model.super_decoder, args.superlr),
        "inpainting": (model.inpaint_decoder, args.inpaintlr),
        "demasking": (model.mask_decoder, args.masklr),
    }

    # Create optimizers for each task
    optimizers = {}
    for task, (decoder, lr) in decoders.items():
        param_dec = optim_factory.add_weight_decay(decoder, args.weight_decay)
        optimizer = torch.optim.AdamW(
            list(param_enc) + list(param_dec), lr=lr, betas=(0.9, 0.95)
        )
        optimizers[task] = optimizer

    return optimizers
