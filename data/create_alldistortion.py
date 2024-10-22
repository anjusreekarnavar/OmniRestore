import os 
import glob 
import numpy as np
from PIL import Image 
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import random
import argparse
import cv2
import os
import numpy as np



def get_args_parser():
    parser = argparse.ArgumentParser('eval dataset creation', add_help=False)
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='images input size')
    parser.add_argument('--sigma', default=0.25, type=float,
                        help='Std of Gaussian noise')
    parser.add_argument('--radius', default=1, type=int,
                        help='blurring radius')
    parser.add_argument('--downsampling_factor', default=4, type=int,
                        help='downsampling')
    parser.add_argument('--mask_shape',type=str,default='ellipse',help='for inpaint mask type')
    parser.add_argument('--percentage',type=int,default=0.25,help='for percentage to mask')
    parser.add_argument('--max_vertices',type=int,default=10,help='maximum vertices for irregular mask')
    parser.add_argument('--mask_radius',type=int,default=5,help='radius for mask')
    parser.add_argument('--num_lines',type=int,default=10,help='lines for mask')
    parser.add_argument('--num_patches',type=int,default=10,help='number of patches in the mask')
    parser.add_argument('--patch_size',type=int,default=16,help='size of each patch')
    parser.add_argument('--num_masks',type=int,default=6,help='size of each patch')
    
    return parser
    
def random_generator():

    n=random.randint(1,5)
    return n

def create_dataset(imgs,args):
    total_distortions=random_generator()
    available_distortions=['denoising','deblurring','super-resolution','inpainting','demasking']
    chosen_distortion=random.sample(available_distortions,total_distortions)
    to_distort=imgs

    for i in chosen_distortion:
   
        if i=='denoising': 
            noise = torch.randn_like(to_distort) * args.sigma
            imgs_noised = imgs + noise
            new_image=imgs_noised
            
        
        elif i=='deblurring':
            blur_image=blur_input_image(to_distort,args.radius)
            new_image=blur_image
       
       
        elif i=='super-resolution':
            lrimage=converto_low_resolution(to_distort,args.downsampling_factor)
            new_image=lrimage
           
 
        elif i=='inpainting': 
             inpaint_mask=generate_mask(to_distort,args.input_size,args.percentage,args.max_vertices,args.mask_radius,args.num_lines)
             #inpaint_mask=inpaint_mask.to(args.device)
             new_image = inpaint_mask
          
 
        else:
            patch_mask=patch_generator(to_distort,args.num_patches,args.patch_size)
            #patch_mask=patch_mask.to(args.device)
            new_image=patch_mask
        
      
        to_distort=new_image
    
    final_distorted_image=to_distort
   
    #inal_distorted_image = final_distorted_image.to(args.device, non_blocking=True)
    return final_distorted_image

def directory_process(source_dir,dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])
    to_pil = transforms.ToPILImage()
    #with open(log_file, 'w') as log:
    for root, _, files in os.walk(source_dir):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Read the image
                    image_path = os.path.join(root, filename)
                    image = Image.open(image_path).convert('RGB')
                    image_tensor = data_transform(image)
                   
                    distorted_image=create_dataset(image_tensor,args)
                    image_tensor = data_transform(image)
                   
                    #relative_path = os.path.relpath(root, source_dir)
                    output_path = os.path.join(dest_dir, filename)
                    #os.makedirs(output_subdir, exist_ok=True)
                    #output_image_path = os.path.join(output_subdir, filename)
                    distorted_image=distorted_image.squeeze(0)
                    #distorted_image = to_pil(distorted_image)
                    save_image(distorted_image,output_path)
                    #log.write(f"{filename}:-->{output_image_path}:: {distortion_type}\n")
    print('process finished')

def main(args):
    # Define the source and destination directories
    source_dir='/scratch3/ven073/new_data/val/groundtruth'
    dest_dir='/scratch3/ven073/new_data/val/input'
 
    directory_process(source_dir,dest_dir)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    #if args.output_dir:
        #Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

