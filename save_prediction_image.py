import cv2
import os
import numpy as np
import time
import testing as test
import pandas as pd
import curses



def prediction(cust_dir,view=False):
    os.chdir(cust_dir)   
    img_dir='darknet/custom_data/images/piling_machine/'

    df = pd.DataFrame(columns=['name','xmin1','ymin1','xmax1','ymax1','xmin2','ymin2','xmax2','ymax2','xmin3','ymin3','xmax3','ymax3','xmin4','ymin4','xmax4','ymax4'])
    net = cv2.dnn.readNet("darknet/backup/yolov3-custom_5000.weights", "darknet/custom_data/cfg/yolov3-custom.cfg")
    classes = []
    with open("darknet/custom_data/images/classes.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
     
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3)) 
    
    file1 = open("darknet/custom_data/test.txt","r")
    file=file1.read().split('\n')
    file1.close()
    file=file[0:len(file)-1]
  
    
    k=0
    
    
    for ff in file:
        df=df.append(pd.Series(), ignore_index=True)
        img = cv2.imread(img_dir+ff.split("/")[4].split(".")[0]+'.jpg')
        df.iloc[k,df.columns.tolist().index('name')]=ff.split("/")[4].split(".")[0]+'.jpg'
    
        img = cv2.resize(img, None, fx=0.9,fy=0.9)  
        img = cv2.resize(img, (416, 416)) 
        height, width, channels = img.shape  
        

        
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        
        # Out has all the info we need all the bounding box axis , now we only have to show the information
        
        #Showing the information
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                m=1
                if confidence > 0.8:
                    # Object detected
                    
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    #cv2.rectangle(img,(x,y),(x+w,y+w),(0,255,0),2)
                   
                    #2 is thickness
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    
                    df.iloc[k,df.columns.tolist().index('xmin'+str(m))]=x 
                    df.iloc[k,df.columns.tolist().index('xmax'+str(m))]=x+w 
                    
                    df.iloc[k,df.columns.tolist().index('ymin'+str(m))]=y
                    df.iloc[k,df.columns.tolist().index('ymax'+str(m))]=y+h
                    m+=1
                   
                    #These fields are basically normalized https://hackernoon.com/understanding-yolo-f5a74bbc7967
        
        number_of_objects_detected = len(boxes)
        print("for image {}, no of boxes detected {}".format(ff,number_of_objects_detected))
        
    
            
         #Now we want to remove the duplicate bounding boxes from the images so we are applying
        #Non Mac Suppression
        indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
        #print(indexes)
        # it gives out of those boxes only 1,0,6,2,5,4
        # 0.5 is score theroshold
        #0.4 is nms_threshold
        
        
        font = cv2.FONT_HERSHEY_COMPLEX
        
        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label=''
                label = str(classes[class_ids[i]])
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(img,label,(x+10,y+40),font,0.5,(0,0,0),1)  
                cv2.putText(img,str(round(confidences[i],3)),(x+10,y+70),font,0.5,(0,0,0),1)  
        os.chdir('output/')           
        cv2.imwrite(ff.split("/")[4].split(".")[0]+'.jpg', img) 
        os.chdir(cust_dir) 
        if view==True:
            cv2.imshow("hey there",img)
            #time.sleep(3)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        k+=1

      

if __name__=="__main__":
    cust_dir="/home/sainijagjit/Desktop/Darknet/customindz"  
    prediction(cust_dir,view=False)

    
    


