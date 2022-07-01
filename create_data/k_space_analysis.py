import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image, ImageOps
from skimage import io, transform
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.io import imsave,imread

def save_image(img, path):
    print('img shape',img.shape)
    img *= 255
    np.clip(img, 0, 255, out=img)

modality='t1ce'
path='paired_brats/val'+modality+'/'
dirs=os.listdir(path)
dirs.sort()
tar_path='paired_brats/val'+modality+'_k_space'+'/'
if not os.path.exists(tar_path):
    os.makedirs(tar_path)

for i in range(len(dirs)):
    print(i)
    img_original = Image.open(path+dirs[i])
    f = np.fft.fft2(img_original)
    fshift = np.fft.fftshift(f)
    a_fshift = np.log(np.abs(fshift))
    ph_fshift = np.angle(fshift)

    imsave(tar_path+dirs[i],a_fshift)
