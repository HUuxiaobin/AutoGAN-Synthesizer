import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
##write text

def write_text(five_points,name):
	with open(name, "w") as f:
		f.write(str(five_points[0][0]))
		f.write(' ')
		f.write(str(five_points[0][1]))
		f.write('\n')
		f.write(str(five_points[1][0]))
		f.write(' ')
		f.write(str(five_points[1][1]))
		f.write('\n')
		f.write(str(five_points[2][0]))
		f.write(' ')
		f.write(str(five_points[2][1]))
		f.write('\n')
		f.write(str(five_points[3][0]))
		f.write(' ')
		f.write(str(five_points[3][1]))
		f.write('\n')
		f.write(str(five_points[4][0]))
		f.write(' ')
		f.write(str(five_points[4][1]))
		f.close()


modality='flair'
stage='train'

path='./'+


