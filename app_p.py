#all imports 
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import time 
import multiprocessing

import numpy as np
from utils import *


'''
multi-processing 

def recoring () -> reads the image takes in input_1.send()
def detection() -> takes the input and process and detects the faces  output_1.recv() input_2.sent()
def show() -> shows fps and image output_2.recv()

'''





fps = 0



folder = './models'

face_extractor_model = "ssd.caffemodel"
face_encoder_model  = "deploy.prototxt"
face_identifier_model = "classifier.pickle"
face_labels = "encoded_label.pickle"


clf = pk.load(open(face_identifier_model,"rb"))
label = pk.load(open(face_labels,'rb'))

model_path = os.path.join(folder,face_extractor_model)
proto_path = os.path.join(folder,face_encoder_model)

dnet = cv2.dnn.readNetFromCaffe(proto_path,model_path)

#face net model
enet = cv2.dnn.readNetFromTorch('./models/openface_nn4.small2.v1.t7')


cam  = cv2.VideoCapture(0)


    
            
def detect(q1,q2):
   
    

    while True:

        frame = q1.get()

        # print(frame.shape ,"in detect")

        blob_img = cv2.dnn.blobFromImage(frame,1.0,size = (300,300),mean = (123.68,116.78,103.94),swapRB = False,crop = False)

        #getting  w and h
        h,w,_ = frame.shape

        #set input 
        dnet.setInput(blob_img)

        result = dnet.forward()
        
        #preprocessing results 
        res = process(result,w,h,conf = 0.9)

        # print("no of detection ",len(res))

        if len(res) > 0:

            

            for i in range(len(res)):
                
                

                vec = get_embedding(frame,res[i],enet)
                preds = clf.predict_proba(vec)[0]
                

                # print(preds)

                j = np.argmax(preds)
                proba = preds[j]
                name = label.classes_[j]

                if proba < 0.5:
                    color = (0,0,255)
                else:
                    color = (0,255,0)

                cv2.rectangle(frame,(res[i][0],res[i][1]),(res[i][2],res[i][3]),color,5)

                    
                cv2.putText(frame,name,(res[i][0],res[i][1]),cv2.FONT_HERSHEY_SIMPLEX,4,color,2,cv2.LINE_AA)


                if len(res) == 2:

                    print(name,proba,i,preds)




        q2.put_nowait(frame)




   

if __name__ == "__main__":
    
    q1 = multiprocessing.Queue()
    q2 = multiprocessing.Queue()

    

    # p1 = multiprocessing.Process(target=record,args= (q1,q2))
    # p2 = multiprocessing.Process(target=detect,args = (q1,q2,))

    video_capture = WebcamVideoStream(src= 0, width=500, height=500).start()




    p2 = multiprocessing.Pool(1,detect,(q1,q2))

    # p1.start()
    # p2.start()



   
    t1 = time.time()

    while True:


        frame = video_capture.read()

        frame = cv2.flip(frame,1)
        
        blob_img = cv2.dnn.blobFromImage(frame,1.0,size = (300,300),mean = (123.68,116.78,103.94),swapRB = False,crop = False)

        q1.put(frame)
        
        frame = q2.get()

        
        cv2.imshow('img',frame)
        

        #calculating fps

        fps += 1
        
        if int(time.time() - t1) >= 2:
            # print('fps ::',fps/(time.time() - t1))
            t1 = time.time()
            fps = 0
        

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    cam.release()    
    cv2.destroyAllWindows()   
            
        
    
    p2.terminate()
    video_capture.stop()

    