import cv2 
import sys 
import time
import logging as log 
import datetime as dt 
from time import sleep 
from keras.models import load_model
model = load_model('model_1.h5')
model.load_weights('weights_1.h5')

emotion_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

model_point = load_model('model_point.h5')
model_point.load_weights('weights_point.h5')

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import glob
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_one_pred(img):
  img = Image.open(img)
  reimg = img.resize((100,100))
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


def get_one_point(x):
    x = images[0]
    x = np.expand_dims(x, axis=2)
    x = np.expand_dims(x, axis=0)
    preds = model_point.predict(x)
    return preds
#cascPath = "haarcascade_frontalface_default.xml"
#faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
video_capture = cv2.VideoCapture()
#img_path = 'F:/ingnew3/图像人脸定位/rgb.jpg'
start_time = time.time()

while True:
  start_time = time.time()
  # if not video_capture.isOpened():
  #   print('Unable to load camera.') 
  #   sleep(5) 
  #   pass
  ret, frame = video_capture.read() 
  #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
  #frame = cv2.imread('ir.jpg')
  # if frame.any() == None or frame == None :
  #   sleep(1)
  #   pass
  # faces = faceCascade.detectMultiScale( 
  #   frame, 
  #   scaleFactor=1.1, 
  #   minNeighbors=5, 
  #   minSize=(30, 30), 
  # #flags=cv2.cv.CV_HAAR_SCALE_IMAGE 
  # )
  # for (x, y, w, h) in faces:
  #   cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
  #   #cv2.imwrite('test.jpg', frame)
  #   eyes = eye_cascade.detectMultiScale(frame, 1.2, 3)
  #   for (ex,ey,ew,eh) in eyes:
  #     cv2.rectangle(frame, (ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2)


  cv2.imwrite('test.png', frame)
  a = get_one_pred('test.png')
  b = a[0].tolist()
  c = b.index(max(b))

  d = get_one_point(frame)
  e = d[0].tolist()

  font = cv2.FONT_HERSHEY_SIMPLEX
  frame = cv2.putText(frame, emotion_list[0]+':'+str(b[0]), (5, 10), font, 0.5, (255, 255, 255), 2)
  frame = cv2.putText(frame, emotion_list[1]+':'+str(b[1]), (5, 30), font, 0.5, (255, 255, 255), 2)
  frame = cv2.putText(frame, emotion_list[2]+':'+str(b[2]), (5, 50), font, 0.5, (255, 255, 255), 2)
  frame = cv2.putText(frame, emotion_list[3]+':'+str(b[3]), (5, 70), font, 0.5, (255, 255, 255), 2)
  frame = cv2.putText(frame, emotion_list[4]+':'+str(b[4]), (5, 90), font, 0.5, (255, 255, 255), 2)
  frame = cv2.putText(frame, emotion_list[5]+':'+str(b[5]), (5, 110), font, 0.5, (255, 255, 255), 2)
  frame = cv2.putText(frame, emotion_list[6]+':'+str(b[6]), (5, 130), font, 0.5, (255, 255, 255), 2)
  for i in range(15):
    frame = cv2.circle(frame,(int(e[i*2]),int(e[i*2+1])),5,(55,255,155),1)

  cv2.imshow('Video', frame)


  if cv2.waitKey(1) & 0xFF == ord('q'): 
    break

  end_time = time.time()
  print(end_time-start_time)

video_capture.release() 
cv2.destroyAllWindows()