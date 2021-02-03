import os
import shutil
from config import *
from utils import *
import cv2

if os.path.exists(PREDICTOR_PATH) == False or os.path.exists(PREDICTOR_FINE_PATH) == False:
    print('Missing shape predictor model files.\n\nRun download.py please...')
    print('\nThen ensure the files are decompressed.')


cap = cv2.VideoCapture(0)
cap.set(3,WIDTH)
cap.set(4,HEIGHT)

D = Detector()
P5 = ShapePredictor(PREDICTOR_PATH)
P68 = ShapePredictor(PREDICTOR_FINE_PATH)

day_info, img_info = time()
path = os.path.join(FILE_PATH,day_info)
if os.path.exists(path) == True:
    shutil.rmtree(path)
    os.mkdir(path)
else:
    os.mkdir(path)


while True:
    ret,frame = cap.read()
    origin = frame.copy()
    rects = D.detect(frame)
    landmarks_5pts = P5.predict(frame,rects)
    draw_rects(frame,rects,color='GREEN')
    draw_landmarks(frame,landmarks_5pts,color='RED')
    cv2.imshow('Live',frame)
    if cv2.waitKey(10) != -1:
        landmarks_68pts = P68.predict(frame,rects)
        draw_landmarks(frame,landmarks_68pts,color='BLUE')
        cv2.imshow('Live',frame)
        assure = cv2.waitKey()
        if assure != 13:
            continue
        else:
            if len(landmarks_5pts) != 1:
                print('Error: number of faces != 1')
                continue
            np.save(os.path.join(path,'5pts.npy'),landmarks_5pts[0])
            np.save(os.path.join(path,'68pts.npy'),landmarks_68pts[0])
            cv2.imshow('Live',origin)
            cv2.waitKey(0)
            cv2.imwrite(os.path.join(path,img_info+'.jpg'),origin)
            print('Image saved!')
            break