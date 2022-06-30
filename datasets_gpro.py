import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image, ImageOps
from skimage import io, transform
import os
import numpy as np
import random

class MRIDataset(Dataset):
    def __init__(self, flair_image_files, t2_image_files, root_dir, crop=False, crop_size=256, multi_scale=False, rotation=False, color_augment=False, mirror=False, transform=None):
        """
        Args:
             split_file: Path to the split file
             root_dir: Directory with all the images
             transform: Optional transform to be appeared on a sample
        """
        flair_file = open(flair_image_files, 'r')
        self.flair_image_files = flair_file.readlines()
        t2_file = open(t2_image_files, 'r')
        self.t2_image_files = t2_file.readlines()
        self.root_dir = root_dir
        self.transform = transform        
        self.crop = crop
        self.crop_size = crop_size
        self.multi_scale = multi_scale
        self.rotation = rotation
        self.color_augment = color_augment
        self.mirror=mirror
        self.rotate90 = transforms.RandomRotation(90)  
        self.rotate45 = transforms.RandomRotation(45)    

    def __len__(self):
        return len(self.flair_image_files)

    def __getitem__(self, idx):
        image_name = self.flair_image_files[idx][0:-1].split('/')
        flair_image = Image.open(os.path.join(self.root_dir, image_name[2], image_name[3], image_name[4])).convert('RGB')
        k_space = Image.open(os.path.join(self.root_dir, image_name[2], 'flair_k_space', image_name[4]))
        #t1 modality
        t1_image = Image.open(os.path.join(self.root_dir, image_name[2], 't1', image_name[4])).convert('RGB')
        ##t1ce modality
        t1ce_image = Image.open(os.path.join(self.root_dir, image_name[2], 't1ce', image_name[4])).convert('RGB')

        # t2 modality - target
        t2_image_name = self.t2_image_files[idx][0:-1].split('/')
        t2_image = Image.open(os.path.join(self.root_dir, t2_image_name[2], t2_image_name[3],t2_image_name[4])).convert('RGB')

        if self.rotation:
            degree = random.choice([90, 180, 270])
            flair_image = transforms.functional.rotate(flair_image, degree)
            t2_image = transforms.functional.rotate(t2_image, degree)
            k_space = transforms.functional.rotate(k_space, degree)
            t1_image = transforms.functional.rotate(t1_image, degree)
            t1ce_image = transforms.functional.rotate(t1ce_image, degree)
        if self.color_augment:
            flair_image = transforms.functional.adjust_gamma(flair_image, 1)
            t2_image = transforms.functional.adjust_gamma(t2_image, 1)
            k_space = transforms.functional.adjust_gamma(k_space, 1)
            t1_image = transforms.functional.adjust_gamma(t1_image, 1)
            t1ce_image = transforms.functional.adjust_gamma(t1ce_image, 1)
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            flair_image = transforms.functional.adjust_saturation(flair_image, sat_factor)
            t2_image = transforms.functional.adjust_saturation(t2_image, sat_factor)
            k_space = transforms.functional.adjust_saturation(k_space, sat_factor)
            t1_image = transforms.functional.adjust_saturation(t1_image, sat_factor)
            t1ce_image = transforms.functional.adjust_saturation(t1ce_image, sat_factor)
        if self.mirror:
            random_seed = random.randint(0, 1)
            if random_seed==1:
                flair_image = ImageOps.mirror(flair_image)
                t2_image = ImageOps.mirror(t2_image)
                k_space = ImageOps.mirror(k_space)
                t1_image = ImageOps.mirror(t1_image)
                t1ce_image = ImageOps.mirror(t1ce_image)
        if self.transform:
            flair_image = self.transform(flair_image)
            t2_image = self.transform(t2_image)
            k_space = self.transform(k_space)
            t1_image = self.transform(t1_image)
            t1ce_image = self.transform(t1ce_image)
        if self.crop:
            W = flair_image.size()[1]
            H = flair_image.size()[2]
            Ws = np.random.randint(0, W-self.crop_size-1, 1)[0]
            Hs = np.random.randint(0, H-self.crop_size-1, 1)[0]
            
            flair_image = flair_image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
            t2_image = t2_image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
            k_space = k_space[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
            t1_image = t1_image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
            t1ce_image = t1ce_image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
        if self.multi_scale:
            H = t2_image.size()[1]
            W = t2_image.size()[2]
            flair_image_s1 = transforms.ToPILImage()(flair_image)
            t2_image_s1 = transforms.ToPILImage()(t2_image)
            flair_image_s2 = transforms.ToTensor()(transforms.Resize([H/2, W/2])(flair_image_s1))
            t2_image_s2 = transforms.ToTensor()(transforms.Resize([H/2, W/2])(t2_image_s1))
            flair_image_s3 = transforms.ToTensor()(transforms.Resize([H/4, W/4])(flair_image_s1))
            t2_image_s3 = transforms.ToTensor()(transforms.Resize([H/4, W/4])(t2_image_s1))
            flair_image_s1 = transforms.ToTensor()(flair_image_s1)
            t2_image_s1 = transforms.ToTensor()(t2_image_s1)
            return {'flair_image_s1': flair_image_s1, 'flair_image_s2': flair_image_s2, 'flair_image_s3': flair_image_s3, 't2_image_s1': t2_image_s1, 't2_image_s2': t2_image_s2, 't2_image_s3': t2_image_s3}
        else:
            return {'flair_image': flair_image,'k_space':k_space ,'t1_image':t1_image ,'t1ce_image':t1ce_image ,'t2_image': t2_image}
        
