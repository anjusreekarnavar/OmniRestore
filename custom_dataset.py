import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2

class TestDataset(Dataset):
    def __init__(self, image_folder, transform):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg','.JPEG'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image  

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    return img

class ImageRestorationDataset(Dataset):
    def __init__(self, noisy_folder, clean_folder, transform):
        self.noisy_dir = noisy_folder
        self.clean_dir = clean_folder
        self.transform = transform
        self.clean_images=sorted(os.listdir(self.clean_dir))
        self.noisy_images=sorted(os.listdir(self.noisy_dir))
     

        assert self.clean_images==self.noisy_images, "mismatch in filenames"
    def __len__(self):
        return len(self.clean_images)



    def __getitem__(self, idx):
        clean_filename= self.clean_images[idx]
        noisy_filename=self.noisy_images[idx]
        assert clean_filename==noisy_filename , "filenames do not match"

        clean_image_path=os.path.join(self.clean_dir,clean_filename)
        noisy_image_path=os.path.join(self.noisy_dir,noisy_filename)
        
        noisy_img = Image.open(noisy_image_path).convert('RGB')
        clean_img = Image.open(clean_image_path).convert('RGB')

        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)
        
        return clean_img,noisy_img
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir):
        super(DataLoaderVal, self).__init__()

        #self.transform = target_transform

        gt_dir = 'groundtruth'
        input_dir = 'input'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))


        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files ]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files ]
        

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



class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image
class CustomDataset(Dataset):
    def __init__(self, original_folder, noisy_folder, transform):
        self.original_folder = original_folder
        self.noisy_folder = noisy_folder
        self.transform = transform
        self.original_images = os.listdir(original_folder)
        self.noisy_images = os.listdir(noisy_folder)

    def __len__(self):
        return min(len(self.original_images), len(self.noisy_images))

    def __getitem__(self, idx):
        img_name_orig = os.path.join(self.original_folder, self.original_images[idx])
        img_name_noisy = os.path.join(self.noisy_folder, self.noisy_images[idx])

        image_orig = Image.open(img_name_orig)
        image_noisy = Image.open(img_name_noisy)

        if self.transform:
            image_orig = self.transform(image_orig)
            image_noisy = self.transform(image_noisy)

        return image_orig, image_noisy
# Define paths to original and noisy data folders
#original_folder = "path_to_original_data_folder"
#noisy_folder = "path_to_noisy_data_folder"

# Create custom dataset instance
#custom_dataset = CustomDataset(original_folder, noisy_folder, transform=None)

# Create data loader for evaluation
#batch_size = 32
#evaluation_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
