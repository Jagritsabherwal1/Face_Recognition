import cv2
import numpy as np
from os import listdir
from os.path import join,isfile

data ='D:/Face_Samples/'
training=[f for f in listdir(data) if isfile (join(data,f))]
train,label=[],[]
for i,files in enumerate(training):
    img_path=data+training[i]
    images=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    train.append(np.asarray(images,dtype=np.uint8))
    label.append(i)

label=np.asarray(label,dtype=np.int32)
model=cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(train) , np.asarray(label))

print('Training Successful')




# Final Running


face_classifier= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def face_detector(img,size=0.5):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)

    if faces is ():
        return img,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi=img[y:y+h,x:x+w]
        roi=cv2.resize(roi,(200,200))
    return img,roi


cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    image,face=face_detector(frame)

    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result=model.predict(face)

        if result[1]<500:
            confidence=int(100*(1-(result[1])/300))
            display=str(confidence)+'% Face Match'
        cv2.putText(image,display,(10,150),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255))


        if confidence>75:
            cv2.putText(image, "UNLOCKED", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255 ,255 ))
            cv2.imshow('face Croper',image)
        else:
            cv2.putText(image, "LOCKED", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
            cv2.imshow("face croppper",image)

    except:
        cv2.putText(image, "FACE NOT FOUND", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
        cv2.imshow('face Croper',image)
        pass
    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()