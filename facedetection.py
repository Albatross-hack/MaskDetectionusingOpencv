import cv2
detector=cv2.CascadeClassifier('facemodel.xml')
camera=cv2.VideoCapture(0)
while(1):
    success,b=camera.read()
    frame=cv2.flip(b,1)
    indata=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=detector.detectMultiScale(indata,minNeighbors=3,minSize=[15,15],maxSize=[400,400])
    for index,(x,y,w,h) in enumerate(face):
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame, 'person '+str(index+1), (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        c=cv2.blur(frame[y:y+w,x:x+h],[50,50])
        frame[y:y+w,x:x+h]=c
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #converts the binary key press value to decimal which is 113
        break