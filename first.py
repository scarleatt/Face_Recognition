import matplotlib.pyplot as plt
import sys
import dlib
import numpy as np
from skimage import io
import openface
import cv2
import os

predictor_model = "shape_predictor_68_face_landmarks.dat"

face_detector = dlib.get_frontal_face_detector()
face_aligner = openface.AlignDlib(predictor_model)

if sys.argv[1].split('.')[1]=='jpg':
    file_name = sys.argv[1]
    image = io.imread(file_name)
    detected_faces = face_detector(image, 0)
    print("I found {} faces in the file {}".format(len(detected_faces), file_name))

    for i, face_rect in enumerate(detected_faces):
        alignedFace = face_aligner.align(534, image, face_rect,
                                         landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        name = "./img_rect_cuted/{}_img_rect_cuted_{}.jpg".format(file_name.split('.')[0], i)
        cv2.imwrite(name, alignedFace)

        I = io.imread(name, as_grey=True)
        io.imsave(name, I)
        print(name)
else:
    imgDir=sys.argv[1]
    if sys.argv[2]:
        dst=sys.argv[2]
    else:
        dst='./'
    for file in os.listdir(imgDir):
        if (file.split('.')[1]=='jpg'):
            file_name=imgDir+file
            image = io.imread(file_name)
            detected_faces = face_detector(image, 0)
            print("I found {} faces in the file {}".format(len(detected_faces), file_name))

            img=plt.imread(file_name)
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray')
            for i, face_rect in enumerate(detected_faces):
                l=face_rect.left()
                r=face_rect.right()
                t=face_rect.top()
                b=face_rect.bottom()
                plt.vlines(l, b, t, csolors="b", linewidth=2)
                plt.hlines(t, l, r, colors="b", linewidth=2)
                plt.hlines(b, l, r, colors="b", linewidth=2)
                plt.vlines(r, b, t, colors="b", linewidth=2)
                print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, l, t, r, b))

            plt.axis('off')
            
            fig.savefig('{}img_{}_rect_all.jpg'.format(dst,file.split('.')[0]),
                        dpi=200,bbox_inches='tight')
            plt.close(fig)