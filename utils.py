import numpy as np
import cv2
import time

fps = 0
t1 = time.time()


colors  = [(0,0,255),(255,0,0),(0,255,0),(0,0,0),(255,255,255)]


def iou2(bbox1,bbox2):
    #content of box : x,y,width,height
    #formula  intersection/union
    
    
    x1,y1,w1,h1 = bbox1
    x2,y2,w2,h2 = bbox2
    

  
    
    x3 = np.max((x1,x2))
    y3 = np.max((y1,y2))
    w3 = np.min((x1+w1,x2+w2)) - x3
    h3 = np.min((y1+h1,y2+h2)) - y3
    
   
    
    if w3<=0 or h3 <=0:
        return 0
    
    iint = w3*h3
    
    u = (w1*h1 + w2*h2) - iint
    
    
    return iint/u

def filter_op(r,w,h,conf = 0.9):
    
    
    box = []
    
    #filtering out using confidence levels
    for i in r[0,0,:]:

        if i[2] > conf:
            b = i[3:] * np.array([w,h,w,h])
            box.append(b.astype('int32'))
        
    return box 

def nms(box,th = 0.1):   
    
    for p in range(len(box)):
        
        box1 = box[p]
    
        if p == len(box) -1:

            break
    
        for j in range(1,len(box)):
        
            box2 = box[j]
            result = iou2(box1,box2)

            if result > th:
                box[j] = np.array([0,0,0,0])

    return box

def process(res,w,h,conf = 0.9,th = 0.1,):
    
    
    rect = filter_op(res,w,h,conf)

    return rect        

def get_embedding(img,r,face_net):

    
    face = img[r[1]:r[3],r[0]:r[2]]
    
    if face.shape[0] == 0 or face.shape[1] == 0:
        print('no extraction')
        return np.zeros(128).reshape(1,128)

    face_blob = cv2.dnn.blobFromImage(face,1.0/255,(96,96),(0, 0, 0),swapRB = True,crop = False)

    face_net.setInput(face_blob)
    return face_net.forward()    

def detect(img,dnet,enet,clf,label):


    #detect the face

    h,w,_= img.shape

    blob_img = cv2.dnn.blobFromImage(img,1.0,size = (300,300),mean = (123.68,116.78,103.94),swapRB = False,crop = False)


    dnet.setInput(blob_img)

    result = dnet.forward()

    res = process(result,w,h,conf = 0.9)

    if len(res) > 0:

            for i in range(len(res)):
                
                vec = get_embedding(img,res[i],enet).reshape(1,1,128)

                preds = clf.predict(vec)[0][0]
                
                j = np.argmax(preds)
                proba = preds[j]
                name = label.classes_[j]

                color = colors[i]

                       
                cv2.rectangle(img,(res[i][0],res[i][1]),(res[i][2],res[i][3]),color,5)

                text = name +":%.2f"%(proba*100)
       
                cv2.putText(img,text,(res[i][0],res[i][1]),cv2.FONT_HERSHEY_SIMPLEX,2,color,2,cv2.LINE_AA)


                
                
    else:
        print('no detction')
    
def show_fps():
    global fps,t1
    
    fps += 1
    delta = int(time.time() - t1)
    if( delta >= 1):
        print(f"fps::{fps//delta}")
        fps = 0
        t1 = time.time()




# class WebcamVideoStream:

#     def __init__(self, src, width, height):
#         # initialize the video camera stream and read the first frame
#         # from the stream
#         self.stream = cv2.VideoCapture(src)
#         self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#         self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#         (self.grabbed, self.frame) = self.stream.read()

#         # initialize the variable used to indicate if the thread should
#         # be stopped
#         self.stopped = False

#     def start(self):
#         # start the thread to read frames from the video stream
#         Thread(target=self.update, args=()).start()
#         return self

#     def update(self):
#         # keep looping infinitely until the thread is stopped
#         while True:
#             # if the thread indicator variable is set, stop the thread
#             if self.stopped:
#                 return

#             # otherwise, read the next frame from the stream
#             (self.grabbed, self.frame) = self.stream.read()

#     def read(self):
#         # return the frame most recently read
#         return self.frame

#     def size(self):
#         # return size of the capture device
#         return self.stream.get(3), self.stream.get(4)

#     def stop(self):
#         # indicate that the thread should be stopped
#         self.stopped = True
