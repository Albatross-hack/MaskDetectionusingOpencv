import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.layers import *
import cv2


labels_dict={0:'without_mask',1:'with_mask'}
color_dict={0:(0,0,255),1:(0,255,0)}

size = 4
cv2.namedWindow("COVID Mask Detection Video Feed")
webcam = cv2.VideoCapture(0) 
load_model = keras.models.load_model('densenet121_detection_model.h5')
classifier = cv2.CascadeClassifier('facemodel.xml')

while True:
    rval, im = webcam.read()
    im=cv2.flip(im,1,1)
    
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
 
    faces = classifier.detectMultiScale(mini)

    for f in faces:
        (x, y, w, h) = [v * size for v in f] 
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(224,224))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,224,224,3))
        reshaped = np.vstack([reshaped])
        result=load_model.predict(reshaped)
        if result[0][0] > result[0][1]:
            percent = round(result[0][0]*100,2)
        else:
            percent = round(result[0][1]*100,2)
        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(im, labels_dict[label] + " " + str(percent) + "%", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    if im is not None:   
        cv2.imshow('COVID Mask Detection Video Feed', im)
    key = cv2.waitKey(10)
    
    # Exit
    if key == 113: #The q key
        break
        
# Stop video
webcam.release()

# Close all windows