import streamlit as st
import numpy as np
import cv2
import tempfile
from keras.models import load_model
from keras.utils import load_img,img_to_array
facemodel=cv2.CascadeClassifier("face.xml")
maskmodel=load_model('mask.h5')
st.title("Face Mask Detection System ")
choice=st.sidebar.selectbox("MY MENU",("HOME","IMAGE","VIDEO","WEB CAMERA","URL"))
if(choice=="HOME"):
    st.header("welcome to mask detection application")
elif(choice=="IMAGE"):
    file=st.file_uploader("upload image")
    if file:
        b=file.getvalue()
        d=np.frombuffer(b,np.uint8)
        frame=cv2.imdecode(d,cv2.IMREAD_COLOR)
        face=facemodel.detectMultiScale(frame)
        for (x,y,l,w) in face :
            crop_face1=frame[y:y+w,x:x+l]
            crop_face=cv2.imwrite("temp.jpg",crop_face1)
            crop_face=load_img("temp.jpg",target_size=(150,150))
            crop_face=img_to_array(crop_face)
            crop_face=np.expand_dims(crop_face,axis=0)
            pred=maskmodel.predict(crop_face)[0][0]
            if(pred==1):
                cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),3)
            else:
                cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),3)
        st.image(frame,channels="BGR",width=400)
elif(choice=="VIDEO"):
    file=st.file_uploader("upload video")
    window=st.empty()
    if file:
        tfile=tempfile.NamedTemporaryFile()
        tfile.write(file.read())  
        vid=cv2.VideoCapture(tfile.name)
        i=1
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                face=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in face :
                    crop_face1=frame[y:y+w,x:x+l]
                    crop_face=cv2.imwrite("temp.jpg",crop_face1)
                    crop_face=load_img("temp.jpg",target_size=(150,150))
                    crop_face=img_to_array(crop_face)
                    crop_face=np.expand_dims(crop_face,axis=0)
                    pred=maskmodel.predict(crop_face)[0][0]
                    if(pred==1):
                       cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),3)
                       path="F:/my_project/FMD/Scripts/data/unmaks/"+"str(i)"+".jpg"
                       cv2.imwrite(path,crop_face)
                       i=i+1
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0))
                window.image(frame,channels="BGR")
elif(choice=="WEB CAMERA"):
    btn=st.button("start camera")
    window=st.empty()
    btn2=st.button("stop camera")
    if btn2:
        st.rerun()  
    if btn: 
        vid=cv2.VideoCapture(0)
        i=1
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                maskmodel=load_model('mask.h5')
                face=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in face :
                    crop_face1=frame[y:y+w,x:x+l]
                    crop_face=cv2.imwrite("temp.jpg",crop_face1)
                    crop_face=load_img("temp.jpg",target_size=(150,150))
                    crop_face=img_to_array(crop_face)
                    crop_face=np.expand_dims(crop_face,axis=0)
                    pred=maskmodel.predict(crop_face)[0][0]
                    if(pred==1):
                       cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),3)
                       path="F:/my_project/FMD/Scripts/data/unmaks/"+"str(i)"+".jpg"
                       cv2.imwrite(path,crop_face)
                       i=i+1
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),3)
                window.image(frame,channels="BGR")                
        btn2=st.button("stop camera")
             

elif(choice=="URL"):
    a=st.text_input("enter the url ")
    btn=st.button("start camera")
    window=st.empty()
    if btn: 
        vid=cv2.VideoCapture(a)
        i=1
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                face=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in face :
                    crop_face1=frame[y:y+w,x:x+l]
                    crop_face=cv2.imwrite("temp.jpg",crop_face1)
                    crop_face=load_img("temp.jpg",target_size=(150,150))
                    crop_face=img_to_array(crop_face)
                    crop_face=np.expand_dims(crop_face,axis=0)
                    pred=maskmodel.predict(crop_face)[0][0]
                    if(pred==1):
                       cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),3)
                       path="F:/my_project/FMD/Scripts/data/unmaks/"+"str(i)"+".jpg"
                       cv2.imwrite(path,crop_face)
                       i=i+1
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),3)
                window.image(frame,channels="BGR")                
            btn2=st.button("stop camera")
            if btn2:
                vid.close()
                st.experimental_rerun()                       