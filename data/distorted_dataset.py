import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import datasets, transforms, models
from temporary import Conversion
from inpaint_mask_generator import generate_mask, patch_generator
import PIL
import random
from perceptualloss import LossNetwork
from torchvision import models
import torch.nn.functional as F


def random_generator():

    n = random.randint(1, 5)
    return n


def new_distorted_dataset(imgs, args):
    total_distortions = random_generator()
    available_distortions = [
        "denoising",
        "deblurring",
        "super-resolution",
        "inpainting",
        "demasking",
    ]
    chosen_distortion = random.sample(available_distortions, total_distortions)
    to_distort = imgs
    batch_size, _, _, _ = imgs.shape

    for i in chosen_distortion:
        if i == "denoising":
            noise = torch.randn_like(to_distort) * args.sigma
            imgs_noised = to_distort + noise
            new_image = imgs_noised
        elif i == "deblurring":
            blur_image = blur_input_image(to_distort, args.radius)
            new_image = blur_image
        elif i == "super-resolution":
            lrimage = converto_low_resolution(to_distort, args.downsampling_factor)
            new_image = lrimage
        elif i == "inpainting":
            inpaint_mask = generate_mask(
                batch_size,
                args.input_size,
                args.percentage,
                args.max_vertices,
                args.mask_radius,
                args.num_lines,
            )
            inpaint_mask = inpaint_mask.to(args.device)
            new_image = to_distort * inpaint_mask
        else:
            patch_mask = patch_generator(imgs, args.num_patches, args.patch_size)
            patch_mask = patch_mask.to(args.device)
            new_image = to_distort * (1 - patch_mask)
        to_distort = new_image
    final_distorted_image = to_distort
    final_distorted_image = final_distorted_image.to(args.device, non_blocking=True)
    return final_distorted_image
