import dlib
import numpy as np
import os
from skimage import io
#import cv2 
#video_cap = cv2.VideoCapture('brush_hair_57.avi')

history=20


predictor_path = "shape_predictor_68_face_landmarks.dat"
faces_path = "pos10464.jpg"


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

win = dlib.image_window()
img = io.imread(faces_path)
win.clear_overlay()
win.set_image(img)

dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))

for k, d in enumerate(dets):
    print("dets{}".format(d))
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
    k, d.left(), d.top(), d.right(), d.bottom()))
    shape = predictor(img, d)
    print("Part 0: {}, Part 1: {} ...".format(shape.part(0),  shape.part(1)))
    win.add_overlay(shape)
win.add_overlay(dets)
print(type(win))
def get_landmarks(im):
    rects = detector(im, 1)
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def get_landmarks_m(im):
    dets = detector(im, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i in range(len(dets)):
        facepoint = np.array([[p.x, p.y] for p in predictor(im, dets[i]).parts()])
        for i in range(68):
            im[facepoint[i][1]][facepoint[i][0]] = [232,28,8]        
    return im    

print("face_landmark:")
print(get_landmarks(img))
print(type(get_landmarks(img)))
print(len(get_landmarks(img)))
dlib.hit_enter_to_continue()


camera.release()
cv2.destroyAllWindows()