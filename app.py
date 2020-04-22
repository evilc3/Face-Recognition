#all imports 
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import time 
import multiprocessing
import tensorflow as tf
import numpy as np
from utils import *



t1 = time.time()

##loading all the models
print("loading models")

#paths 

folder = './models'

face_extractor_model = "ssd.caffemodel"

face_encoder_model  = "deploy.prototxt"

# face_identifier_model = "classifier.pickle"

face_labels = "encoded_label.pickle"

model_path = os.path.join(folder,face_extractor_model)

proto_path = os.path.join(folder,face_encoder_model)




clf = tf.keras.models.load_model("./tf_models/")

label = pk.load(open(face_labels,'rb'))


dnet = cv2.dnn.readNetFromCaffe(proto_path,model_path)

#face net model
enet = cv2.dnn.readNetFromTorch('./models/openface_nn4.small2.v1.t7')

# print(dnet,enet)
print(f"loading completed in {int(time.time() - t1)}")






# reading images form the cam

#gettiing training images
cam = cv2.VideoCapture(0)

while 1 :


    _,frame = cam.read()
    
  
    detect(frame,dnet,enet,clf,label)

    
    cv2.imshow('img',frame)
    

    #calculating fps

    show_fps()

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
cam.release()    
cv2.destroyAllWindows()   
        
    
    

