import urllib.request
from config import *
import os

def download(url,path):
    f = urllib.request.urlopen(url)
    data = f.read()
    bz = path + '.bz2'
    with open(bz,'wb') as datbz:
        datbz.write(data)

if os.path.exists(PREDICTOR_PATH) == False or os.path.exists(PREDICTOR_FINE_PATH) == False:
    print('Missing shape predictor...Waiting for download...\n')
    url_5pts = 'http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2'
    url_68pts = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    print('You can download manually from given urls. \nPlease refer to readme document for details.\n')
    print('Downloading 5-points shape predictor model...')
    download(url_5pts,PREDICTOR_PATH)
    print('Downloading 68-points shape predictor model...')
    download(url_68pts,PREDICTOR_FINE_PATH)
    print('\nFinished. Please decompress the files.')
else:
    print('Shape predictor model files already exists...')
