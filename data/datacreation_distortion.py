import torch
import torch.nn.functional as F
import numpy as np
import random
from torchvision import datasets, transforms
from PIL import ImageDraw, Image,ImageColor
from torchvision.utils import save_image
def mask_decision(start,stop):

    n= random.randint(start,stop)
    return n
def generate_mask(image,img_size,per,max_vertices,radius,num_lines):
    # Initialize mask tensor
    mask = torch.zeros_like((image))
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
        _, height, width = image.shape
        hole_size = int(per * img_size)
        top = np.random.randint(0, height - hole_size + 1)
        left = np.random.randint(0, width - hole_size + 1)
    # Set the selected region to zero (black)
     
        bottom = top + hole_size
        right = left + hole_size
        image[:, top:bottom, left:right] = 0
        image=image

  

    if mask_shape=='circle':
        _, height, width = image.shape
        center_x = np.random.randint(radius, width - radius)
        center_y = np.random.randint(radius, height - radius)
        center = (center_x, center_y)
    
        
        y, x = np.ogrid[:height, :width]
    
    # Create a circular mask
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    
    # Convert mask to tensor
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
    
    # Apply the mask: set circular region to black
        for c in range(image.size(0)):
            image[c][mask_tensor] = 0
        image=image


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
        image=mask*image
        
      
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
        image=mask*image
       
    
    return image


def patch_generator(image,num_masks,max_size):

  image=image.squeeze(0)
  channels, height, width = image.shape
  
  mask_size = (max_size, max_size)

  mask_height, mask_width = mask_size

  for _ in range(num_masks):
        # Ensure the mask fits within the image dimensions
        top = np.random.randint(0, height - mask_height + 1)
        left = np.random.randint(0, width - mask_width + 1)
        
        # Define the mask region
        bottom = top + mask_height
        right = left + mask_width
        
        # Apply the mask by setting the specified region to black (0)
        image[:, top:bottom, left:right] = 0
        
        # Apply the mask by setting the specified region to black (0)
        newimage=image
        

  return newimage
        
     
