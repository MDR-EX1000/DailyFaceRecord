## Background

**Generate GIF file that records the daily variations on your face in a period of time.**  
Detection and alignment are appied to calibrate the pose and position of faces.

## Requirements
```opencv-python```  
```dlib```  
```imageio```  
```urllib```

## Usage
1. **Download model**  
```
python download.py
```  
or **manually download** with the following url and move to ```./dat/```  

[dlib-5pts_shape_predictor](http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2)  
[dlib-68pts_shape_predictor](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

2. **Decompress the .bz2 files**  

3. **Daily running to record face image.**  
```
python run.py
```
4. **Crop faces and generate GIF file with the recorded faces**  

```
python process.py
```  

**Some important parameters in** ```config.py```  
```
FACE_FACTOR = 0.55 #face size in the cropped image 
CROP_FACTOR_W = 0.75 #width of the cropped image
CROP_FACTOR_H = 1.0 #height of the cropped image
OFFSET_FACTOR_W = 0.5 #face center offset_x
OFFSET_FACTOR_H = 0.65 #face center offset_y
```
