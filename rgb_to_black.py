import io
import sys
import matplotlib.pyplot as plt
import sys
import dlib
import numpy as np
from skimage import io
import openface
import cv2
import os

imgDir=sys.argv[1]
dst=sys.argv[2]
for file in os.listdir(imgDir):
    if (file.split('.')[1]=='jpg'):
        img_name=imgDir+file
        name=file.split('.')[0]
        I=io.imread(img_name, as_grey=True)
        io.imsave("{}black_{}.jpg".format(dst,name), I)