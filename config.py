import cv2
#COLORMAP
COLOR2RGB = {'BLUE':(0xDD,0xB8,0x22),'GREEN':(0x92,0xDD,0x22),'RED':(0x6D,0x22,0xDD),'SP':(0x2D,0x50,0xDF)}
#RESOULUTIONMAP
RESOLUTION2SIZE = {'720p':[1280,720],'480p':[640,480]}
#PAD_TYPE
PAD_TYPE = cv2.BORDER_REPLICATE
INTERPOLATE_TYPE = cv2.INTER_CUBIC

#DAT PATH
STD_LANDMARK_PATH = './dat/standard_landmark_68pts.npy'
PREDICTOR_PATH = './dat/shape_predictor_5_face_landmarks.dat'
PREDICTOR_FINE_PATH = './dat/shape_predictor_68_face_landmarks.dat'

#FILE_PATH
FILE_PATH = './files'

######SETTING PARAM######
RESOLUTION = '480p'
FACE_FACTOR = 0.55
CROP_FACTOR_W = 0.75
CROP_FACTOR_H = 1.0
OFFSET_FACTOR_W = 0.5
OFFSET_FACTOR_H = 0.65
#########################

WIDTH = RESOLUTION2SIZE[RESOLUTION][0]
HEIGHT = RESOLUTION2SIZE[RESOLUTION][1]
CROP_W = int(CROP_FACTOR_W * WIDTH)
CROP_H = int(CROP_FACTOR_H * HEIGHT)
FACE_SCALE = int(max(CROP_H,CROP_W) * FACE_FACTOR)
OFFSET_W = int(CROP_W * OFFSET_FACTOR_W)
OFFSET_H = int(CROP_H * OFFSET_FACTOR_H)