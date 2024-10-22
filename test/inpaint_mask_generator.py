import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import cv2
from torchvision import datasets, transforms
from PIL import ImageDraw, Image,ImageColor
def mask_decision(start,stop):

    n= random.randint(start,stop)
    return n
def generate_mask(batch_size, img_size,per,max_vertices,radius,num_lines):
    # Initialize mask tensor
    mask = torch.ones(batch_size, 1, img_size, img_size)
    random_decision=mask_decision(1,4)
    if random_decision==1:
        mask_shape='square'
    elif random_decision==2:
        mask_shape='circle'
    elif random_decision==3:
        mask_shape='line'
    else:
        mask_shape='ellipse'

    if mask_shape=='square':
    
    
    # Randomly select a square region to inpaint (here, using 25% of the image size)
        hole_size = int(per * img_size)
        x = torch.randint(0, img_size - hole_size, (batch_size,))
        y = torch.randint(0, img_size - hole_size, (batch_size,))
    
    # Set the selected region to zero (black)
        for i in range(batch_size):
            mask[i, :, y[i]:y[i] + hole_size, x[i]:x[i] + hole_size] = 0.0
    if mask_shape=='circle':
    
        center_x = torch.randint(radius, img_size - radius, (batch_size,))
        center_y = torch.randint(radius, img_size - radius, (batch_size,))
    
        for i in range(batch_size):
            y, x = torch.meshgrid(torch.arange(img_size), torch.arange(img_size))
            dist_from_center = torch.sqrt((x - center_x[i])**2 + (y - center_y[i])**2)
            mask[i, :, dist_from_center <= radius] = 0.0
    if mask_shape=='line':
        
        line_length=(20, 50)
        line_width=(1, 3)
        mask = Image.new('L',  (img_size, img_size), color=255)  # Initialize mask with white (255)

        draw = ImageDraw.Draw(mask)

        for _ in range(num_lines):
            start_point = (random.randint(0, img_size), random.randint(0, img_size))
            end_point = (start_point[0] + random.randint(line_length[0], line_length[1]),
                     start_point[1] + random.randint(line_length[0], line_length[1]))
            line_width_val = random.randint(line_width[0], line_width[1])
            draw.line([start_point, end_point], fill=0, width=line_width_val)

        mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0
    
    if mask_shape=='ellipse':
    
        mask = Image.new('L', (img_size, img_size), 255)  
        red = ImageColor.getrgb("red")
        draw = ImageDraw.Draw(mask)
        num_vertices = torch.randint(3, max_vertices + 1, (1,)).item()

        for _ in range(num_vertices):
            x = torch.randint(img_size, (1,)).item()
            y = torch.randint(img_size, (1,)).item()
            draw.ellipse([(x, y), (x + torch.randint(20, 100, (1,)).item(), y + torch.randint(20, 100, (1,)).item())], fill=50,outline ="green")

        mask = transforms.ToTensor()(mask).unsqueeze(0)
    
    return mask


def patch_generator(image,num_patches, patch_size):
  batch_size, channels, height, width = image.shape
  # Create a mask of zeros with the same shape as the image.
  mask = torch.zeros((batch_size, channels, height, width))
  # Generate random coordinates for the masks.
  coords = torch.randint(0, height - patch_size + 1, (batch_size, num_patches, 2))
  # Create the masks by setting the corresponding pixels to 1.
  for i in range(batch_size):
    for j in range(num_patches):
      x, y = coords[i, j]
      mask[i, :, x:x + patch_size, y:y + patch_size] = 1

  # Apply the mask to the images.


  return mask
def patch_generator2(image,device):
  batch_size, channels, height, width = image.shape
  # Create a mask of zeros with the same shape as the image.
  mask_batch_tensor = torch.zeros((batch_size,height, width),dtype=torch.float32)
  # Generate random coordinates for the masks.
  rows=6
  cols=6
  cell_height=height//rows
  cell_width=width//cols
  #coords = torch.randint(0, height - patch_size + 1, (batch_size, num_patches, 2))
  # Create the masks by setting the corresponding pixels to 1.
  for i in range(batch_size):
    for row in range(rows):
        for col in range(cols):
            top_left_y=row*cell_height
            top_left_x=col*cell_width
            bottom_right_y=top_left_y+cell_height
            bottom_right_x=top_left_x+cell_width
            if (row+col)%2==0:
                mask_batch_tensor[i,top_left_y:bottom_right_y,top_left_x:bottom_right_x]=1
  white_color_tensor=torch.ones_like(image)
  white_color_tensor=white_color_tensor.to(device)

  mask_batch_tensor=mask_batch_tensor.unsqueeze(1).expand(batch_size,channels,height,width)
  mask_batch_tensor=mask_batch_tensor.to(device)
  mask_batch_tensor=torch.where(mask_batch_tensor==1,image,white_color_tensor)
  # Apply the mask to the images.


  return mask_batch_tensor

def mask_to(tensor,device,mask_root,mask_type,mask_id=-1,n=100):
    batch = tensor.shape[0]
    mask_root2=os.path.join(mask_root,mask_type)
    if mask_id < 0:
        mask_id = np.random.randint(0, n, batch)
        masks = []
        for i in range(batch):
            masks.append(cv2.imread(os.path.join(mask_root2, f'{mask_id[i]:06d}.png'))[None, ...] / 255.)
        mask = np.concatenate(masks, axis=0)
    else:
        mask = cv2.imread(os.path.join(mask_root2, f'{mask_id:06d}.png'))[None, ...] / 255.

    mask = torch.tensor(mask).permute(0, 3, 1, 2).float()
    # for images are clipped or scaled
    mask = F.interpolate(mask, size=tensor.shape[2:], mode='nearest')
    mask=mask.to(device)
    masked_tensor = mask * tensor
    return masked_tensor + (1. - mask)
        
     
