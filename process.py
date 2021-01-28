import os
import functools
import numpy as np
import imageio
from config import *
from utils import *
import cv2

font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
pos = (CROP_W - 128,CROP_H - 10)
flag_text = True
size_text = 0.55
width_text = 2
flag_gif = True
duration = 0.2


MONTH2ID = {
    'January': '01',
    'February': '02',
    'March': '03',
    'April': '04',
    'May': '05',
    'June': '06',
    'July': '07',
    'August': '08',
    'September': '09',
    'October': '10',
    'November': '11',
    'December': '12'
}


def cmp(x,y):
    def f(a,b):
        if a < b:
            return -1
        return 1
    mx,dx,yx = x.split('-')
    my,dy,yy = y.split('-')
    yx = int(yx)
    yy = int(yy)
    if yx == yy:
        mx = int(MONTH2ID[mx])
        my = int(MONTH2ID[my])
        if mx == my:
            dx = int(dx)
            dy = int(dy)
            return f(dx,dy)
        else:
            return f(mx,my)
    else:
        return f(yx,yy)

def timestr(x):
    m,d,y = x.split('-')
    return y + '-' + MONTH2ID[m] + '-' + d

dirs = os.listdir(FILE_PATH)
dirs = sorted(dirs,key=functools.cmp_to_key(cmp))
dst_landmarks = np.load(STD_LANDMARK_PATH)
dst_landmarks *= FACE_SCALE
dst_landmarks[:,0] += OFFSET_W
dst_landmarks[:,1] += OFFSET_H

idx = 0
img_list = []
for dir in dirs:
    print(timestr(dir))
    idx += 1
    if '.py' in dir:
        continue
    files = os.listdir(os.path.join(FILE_PATH,dir))
    for file in files:
        if '.jpg' in file:
            img_path = os.path.join(FILE_PATH,dir,file)
    img = cv2.imread(img_path)
    src_landmarks = np.load(os.path.join(FILE_PATH,dir,'68pts.npy'))
    img_out = align_crop(img,src_landmarks,dst_landmarks)
    if flag_text == True:
        cv2.putText(img_out,timestr(dir),pos,font,size_text,COLOR2RGB['SP'],width_text)
    out_path = os.path.join('crop',str(idx) + '.jpg')
    cv2.imwrite(out_path,img_out,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
    if flag_gif == True:
        img_list.append(imageio.imread(out_path))

if flag_gif == True:
    out_path = 'FaceRecord.gif'
    imageio.mimsave(out_path,img_list,'GIF',duration=duration)

