import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    return img

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir,transform=None):
        super(DataLoaderTrain, self).__init__()

        #self.transform = target_transform

        gt_dir = 'groundtruth'
        input_dir = 'input'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))

        self.transform=transform
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files ]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files ]
        

        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        
                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        clean_img = load_img(self.clean_filenames[tar_index])
        noisy_img = load_img(self.noisy_filenames[tar_index])

        # Convert to float tensors
        clean_tensor = torch.from_numpy(np.float32(clean_img)).permute(2, 0, 1)  # Shape (C, H, W)
        noisy_tensor = torch.from_numpy(np.float32(noisy_img)).permute(2, 0, 1)

        # Optionally print shapes for debugging
        print(f"Clean tensor shape: {clean_tensor.shape}")
        print(f"Noisy tensor shape: {noisy_tensor.shape}")

        # Apply transformations if provided
        if self.transform:
            clean = self.transform(clean_tensor)  # Transform is applied to tensor
            noisy = self.transform(noisy_tensor)
        
        return clean, noisy, clean_filename, noisy_filename

class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir,transform=None):
        super(DataLoaderVal, self).__init__()

        #self.transform = target_transform

        gt_dir = 'groundtruth'
        input_dir = 'input'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))


        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files ]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files ]
        
        self.transform=transform
        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))


        
                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

       
    
        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)
        
        
        
        
        return clean, noisy, clean_filename, noisy_filename

