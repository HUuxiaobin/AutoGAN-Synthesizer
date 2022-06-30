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





#    plt.imshow(a_fshift, 'gray')
#    plt.savefig('example.png')
#    save_image(a_fshift,'exampe.png')
#    print(a_fshift.shape)
#    target=Image.fromarray(a_fshift)
#    target.convert('L').save('example.png')
				
				
#    plt.imshow(a_fshift, 'gray')
#    plt.savefig('example.png')

    #plt.savefig('example.png')
    #a_fshift.save('example.png')

# plt.subplot(131), plt.imshow(img_original, 'gray'), plt.title('original')
# plt.subplot(132), plt.imshow(a_fshift, 'gray'), plt.title('amp')
# plt.subplot(133), plt.imshow(ph_fshift, 'gray'), plt.title('phase')
# plt.show()


#s_real = np.exp(a_fshift)*np.cos(ph_fshift)
#s_imag = np.exp(a_fshift)*np.sin(ph_fshift)
#
#s = np.zeros([240, 240], dtype=complex)
#s.real = np.array(s_real)
#s.imag = np.array(s_imag)
#
#fshift = np.fft.ifftshift(s)
#img_back = np.fft.ifft2(fshift)
#
#img_back = np.abs(img_back)
#plt.subplot(141), plt.imshow(img_original, 'gray'), plt.title('original')
#plt.subplot(142), plt.imshow(a_fshift, 'gray'), plt.title('amp')
#plt.subplot(143), plt.imshow(ph_fshift, 'gray'), plt.title('phase')
#plt.subplot(144), plt.imshow(img_back, 'gray'), plt.title('inverse fourier')
#plt.show()