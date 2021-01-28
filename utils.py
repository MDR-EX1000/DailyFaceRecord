from config import *
import dlib
import datetime
import numpy as np
import cv2

class Detector():
    def __init__(self):
        super(Detector,self).__init__()
        self.detector = dlib.get_frontal_face_detector()

    def detect(self,img):
        return self.detector(img,1)

class ShapePredictor():
    def __init__(self,model_path):
        super(ShapePredictor,self).__init__()
        self.predictor = dlib.shape_predictor(model_path)
        self.n = int(model_path.split('_')[2])

    def shape2array(self,shape):
        coords = np.zeros((self.n,2),dtype='int')
        for i in range(self.n):
            coords[i] = (shape.part(i).x,shape.part(i).y)
        return coords

    def predict(self,img,rects):
        shapes = []
        for (i,rect) in enumerate(rects):
            shape = self.shape2array(self.predictor(img,rect))
            shapes.append(shape)
        return shapes

def time():
    day_info = datetime.datetime.now().strftime("%B-%d-%Y")
    img_info = datetime.datetime.now().strftime("%I-%M-%p-%B-%d-%Y")
    return day_info,img_info

def draw_rects(img,rects,color):
    assert(color in COLOR2RGB.keys())
    for (i,rect) in enumerate(rects):
        cv2.rectangle(img,(rect.left(),rect.top()),(rect.right(),rect.bottom()),COLOR2RGB[color],2)
    return img

def draw_landmarks(img,shapes,color):
    assert(color in COLOR2RGB.keys())
    for shape in shapes:
        for (x,y) in shape:
            cv2.circle(img,(x,y),2,COLOR2RGB[color],-1)
    return img


def align_crop(img,src_landmarks,dst_landmarks):
    tform = cv2.estimateAffinePartial2D(src_landmarks,dst_landmarks, ransacReprojThreshold=np.Inf)[0]
    img_align = cv2.warpAffine(img,tform,(CROP_W,CROP_H),flags=INTERPOLATE_TYPE,borderMode=PAD_TYPE)
    return img_align

