import cv2 
import sys 
import dlib
import os
import time
from skimage import io
import logging as log 
import datetime as dt 
from time import sleep 
import numpy as np
# from keras_neural_style_transfer import *
from keras.models import load_model
model = load_model('model.h5')
model.load_weights('weights.h5')
emotion_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import glob
from PIL import Image

def get_one_pred(img):
    img = Image.open(img)
    reimg = img.resize((200,200))
    x = np.array(reimg)
    x = np.expand_dims(x, axis=0)
    test_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=True)
    test_datagen.fit(x)
    test_data = test_datagen.flow(x)
    preds = model.predict(test_data[0])
    return preds

video_capture = cv2.VideoCapture('angry.avi') 
predictor_path = "shape_predictor_68_face_landmarks.dat"
count = 1
history=20

while True:
	start_time = time.time()

	if not video_capture.isOpened():
		print('Unable to load camera.')
		sleep(5)
		pass

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictor_path)

	def get_landmarks(im):
		try:
			rects = detector(im, 1)
			tmp=np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
		except:
			tmp=[]
		return tmp

	ret, frame = video_capture.read()
	try:
		dets = detector(frame, 1)
	except:
		continue

	for k, d in enumerate(dets):
		frame_20 = cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()),(0, 255, 0), 2)
		shape = predictor(frame, d)
		tmp = get_landmarks(frame)
		if len(tmp) != 0:
			for i in range(len(tmp)):
				tmp1 = np.array(tmp[i])
				tmp2 = tmp1[0]
				tmp3 = tuple(tmp2)
				frame_20 = cv2.circle(frame, tmp3, 1, (255,0,0))

	cv2.imwrite('frame.jpg', frame)

	a = get_one_pred('frame.jpg')
	b = a[0].tolist()
	c = b.index(max(b))
	
	# img_nrows, img_ncols = re_img_nrows_img_ncols('frame.jpg')
	# outputs,combination_image = re_outputs_combination_image('frame.jpg','fangao.jpg',0.025,1.0,1.0)
	# f_outputs = K.function([combination_image], outputs)
	# style_img = gan_img('frame.jpg', 'F:/ingnew3/图像人脸定位/keras_gan_result/results', 5)

	font = cv2.FONT_HERSHEY_SIMPLEX
	frame_20 = cv2.putText(frame_20, emotion_list[0]+':'+str(b[0]), (5, 10), font, 0.5, (255, 255, 255), 2)
	frame_20 = cv2.putText(frame_20, emotion_list[1]+':'+str(b[1]), (5, 30), font, 0.5, (255, 255, 255), 2)
	frame_20 = cv2.putText(frame_20, emotion_list[2]+':'+str(b[2]), (5, 50), font, 0.5, (255, 255, 255), 2)
	frame_20 = cv2.putText(frame_20, emotion_list[3]+':'+str(b[3]), (5, 70), font, 0.5, (255, 255, 255), 2)
	frame_20 = cv2.putText(frame_20, emotion_list[4]+':'+str(b[4]), (5, 90), font, 0.5, (255, 255, 255), 2)
	frame_20 = cv2.putText(frame_20, emotion_list[5]+':'+str(b[5]), (5, 110), font, 0.5, (255, 255, 255), 2)
	frame_20 = cv2.putText(frame_20, emotion_list[6]+':'+str(b[6]), (5, 130), font, 0.5, (255, 255, 255), 2)
	cv2.imshow('Video', frame_20) 
	# cv2.imshow('style_img', style_img)

	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
	
	end_time = time.time()
	print(end_time - start_time)

video_capture.release() 
cv2.destroyAllWindows()