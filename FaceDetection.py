import cv2
import numpy as np

face_classifier= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_extract(img):
    #gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(img,2
                                           ,5)
    if faces is ():
        return None
    for(x,y,w,h) in faces:
        crop=img[y:y+h, x:x+w]
    return crop



cap=cv2.VideoCapture(0);
count=0
count1=0

while True:
    ret,frame=cap.read()
    if face_extract(frame) is not None:
        count+=1
        face=cv2.resize(face_extract(frame),(300,300))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        file_path='D:/Face_Samples/user'+str(count)+'.jpg'
        cv2.imwrite(file_path,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow("Faces",face)

    else:
        print("Face Not Found")
        count1+=1
        pass

    if cv2.waitKey(1)==13 or count==100 :
        break
    elif count1==200:
        break
cap.release()
cv2.destroyAllWindows()
print("Samples Collected Successfully")

