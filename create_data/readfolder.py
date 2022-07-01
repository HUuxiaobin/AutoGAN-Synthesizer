import numpy as np
import os
import glob

modality='flair'
stage='val'

path='paired_brats/'+stage+modality
with open('paired_brats/'+stage+'_'+modality+'.txt','w') as f:
	for file in glob.glob(path+'/*'):
		f.write(file+'\n')



