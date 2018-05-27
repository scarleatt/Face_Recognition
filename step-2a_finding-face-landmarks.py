import sys
import dlib
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import openface
import cv2
import os

predictor_model = "shape_predictor_68_face_landmarks.dat"

face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)

imgDir=sys.argv[1]
for file in os.listdir(imgDir):
	if (file.split('.')[1]=='jpg'):
		file_name=imgDir+file
		name=file_name.split('.')[0]
		image = io.imread(file_name)

		detected_faces = face_detector(image, 1)
		print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

		for i, face_rect in enumerate(detected_faces):
			print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}"
				.format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
			pose_landmarks = face_pose_predictor(image, face_rect)

			img = plt.imread(file_name)
			fig, ax = plt.subplots()
			ax.imshow(img, cmap='gray')

			x = np.zeros(68); y = np.zeros(68)
			for j in range(0, 68):
				# print("{}, {}".format(j, pose_landmarks.part(j)))
				x[j] = pose_landmarks.part(j).x
				y[j] = pose_landmarks.part(j).y
			plt.scatter(x, y, marker='.', c='b')

			plt.axis('off')
			fig.savefig('./img_rect_cuted_dot/{}_dot.jpg'.format(file.split('.')[0]), dpi=200,
						bbox_inches='tight')
			# plt.show()
			plt.close(fig)
			# print("Saved img_rect_cuted_dot_{}_{}.jpg".format(i,name))