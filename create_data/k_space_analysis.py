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
# dicomfile_ge = pydicom.read_file('/home/ray/Downloads/ge_head_1.dcm')

def save_image(img, path):
    print('img shape',img.shape)
    img *= 255
    np.clip(img, 0, 255, out=img)
    #img=Image.fromarray(img)
#    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    cv2.imwrite('ex.png',img)
    #img.convert('I').save('example.png')
    #Image.fromarray(img.astype('uint8'), 'L').save(path)
modality='flair'
path='./test/'+modality+'/'
dirs=os.listdir(path)
dirs.sort()
tar_path='./test/'+modality+'_k_space'+'/'
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
