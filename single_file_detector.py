import cv2
import os
import numpy as np
import time

def detector(cust_dir,img):
    os.chdir(cust_dir) 
    net = cv2.dnn.readNet("darknet/backup/yolov3-custom_5000.weights", "darknet/custom_data/cfg/yolov3-custom.cfg")
    classes = []
    with open("darknet/custom_data/images/classes.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
     
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3)) 
    
    img_dir=img
    
    
    
    img = cv2.imread(img_dir)
    
    
    
    
    img = cv2.resize(img, None, fx=0.9,fy=0.9)   
    img = cv2.resize(img, (416, 416)) 
    height, width,channels= img.shape  
    
    
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
       
            if confidence > 0.7:
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
               
                #These fields are basically normalized https://hackernoon.com/understanding-yolo-f5a74bbc7967
    
    number_of_objects_detected = len(boxes)
    print("for image {}, no of boxes detected {}".format(img_dir,number_of_objects_detected))
    
    
        
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
    
    cv2.imshow("hey there",img)
    #time.sleep(3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
if __name__=="__main__":
    cust_dir="/home/sainijagjit/Desktop/Darknet/customindz" 
    detector(cust_dir,'/home/sainijagjit/Desktop/customindz/darknet/custom_data/images/james.jpg')
    
    
    
    

        
    









