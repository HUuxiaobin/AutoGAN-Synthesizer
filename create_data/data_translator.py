import numpy as np
import glob
import os
import warnings
import shutil
import SimpleITK as sitk
import numpy as np
import shutil
from matplotlib import pyplot as plt
from skimage.io import imsave,imread
from scipy import ndimage

def split_single_images(stack_images,zone,seg_images,file_loc,file_name,stack_images_t2_loop,stack_images_flair_loop,stack_images_t1_loop):
    #lesion domain
#    for j in range(zone[0]+2,zone[-1]-2,2):
#        print('j num',j)
        #print(int((slice_num-Num)/2)+i)
    ##sample from slice_patient=25-125
    for j in range(25,125,1):
        print('j num',j)

#        print(stack_images.shape)
#        print(seg_images.shape)
        single_image=stack_images[j,:,:]
        single_seg_images=seg_images[j,:,:]

        single_t2=stack_images_t2_loop[j,:,:]
        single_t1=stack_images_t1_loop[j,:,:]
        single_flair=stack_images_flair_loop[j,:,:]

        imsave(file_loc+'t1ce/'+file_name+str(j)+'.png',single_image)
        imsave(file_loc+'seg/'+file_name+str(j)+'.png',single_seg_images)
        imsave(file_loc+'t2/'+file_name+str(j)+'.png',single_t2)
        imsave(file_loc+'flair/'+file_name+str(j)+'.png',single_flair)
        imsave(file_loc+'t1/'+file_name+str(j)+'.png',single_t1)
#step1:copy into the dataset file
# subject_list=[]

stack_images_seg = np.ndarray((155, 240, 240), dtype=np.float32)
stack_images_t1ce = np.ndarray((155, 240, 240), dtype=np.float32)

file_loc='paired_brats/train'
soure_loc='MICCAI_BraTS_2018_Data_Training/LGG'

if not os.path.exists(file_loc):
    os.makedirs(file_loc)

if not os.path.exists(file_loc+'t1/'):
    os.makedirs(file_loc+'t1/')

if not os.path.exists(file_loc+'seg/'):
    os.makedirs(file_loc+'seg/')

if not os.path.exists(file_loc+'t2/'):
    os.makedirs(file_loc+'t2/')

if not os.path.exists(file_loc+'flair/'):
    os.makedirs(file_loc+'flair/')

if not os.path.exists(file_loc+'t1ce/'):
    os.makedirs(file_loc+'t1ce/')
##step2: read all flair images and picp up 100 patient slice and name them in sequence

patient_num=0
slice_patient=25
print('this is  current work')
i=1
for subject_id in glob.glob(soure_loc+'/*/*'):
    print(subject_id)
    if 'seg' in subject_id:
    #print(subject_id)
        stack_images_seg=sitk.ReadImage(subject_id)
        stack_images_seg=sitk.GetArrayFromImage(stack_images_seg)
        #print(stack_images_seg)

        #attention lesion zone
        #print(stack_images.shape) ###[155,240,240]
        lesion_zone=np.where(stack_images_seg!=0)[0]
        print('seg shape', np.shape(stack_images_seg))
        #print(lesion_zone,lesion_zone[0],lesion_zone[-1])
    if 't1.nii' in subject_id:
        #print(subject_id)
        stack_images_t1=sitk.ReadImage(subject_id)
        stack_images_t1=sitk.GetArrayFromImage(stack_images_t1)
    ##sigle modality
								
    if 't1ce' in subject_id:
        #print(subject_id)
        stack_images_t1ce=sitk.ReadImage(subject_id)
        stack_images_t1ce=sitk.GetArrayFromImage(stack_images_t1ce)
    ##sigle modality

    if 't2' in subject_id:
        #print(subject_id)
        stack_images_t2=sitk.ReadImage(subject_id)
        stack_images_t2=sitk.GetArrayFromImage(stack_images_t2)

    if 'flair' in subject_id:
        #print(subject_id)
        stack_images_flair=sitk.ReadImage(subject_id)
        stack_images_flair=sitk.GetArrayFromImage(stack_images_flair)


    a,b=os.path.splitext(subject_id)
    a,b=os.path.splitext(a)
    a,b=os.path.split(a)
    a,b=os.path.split(a)
    print(b)
    #print(i)
    if i%5==0:
        split_single_images(stack_images_t1ce,lesion_zone,stack_images_seg,file_loc,b,stack_images_t2,stack_images_flair,stack_images_t1)

    i=i+1
