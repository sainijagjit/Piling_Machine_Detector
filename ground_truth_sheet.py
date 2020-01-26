import pandas as pd
import numpy as np

df = pd.DataFrame(columns=['name','bounding_box_no','xmin','ymin','xmax','ymax'])

label_dir="darknet/custom_data/labels/piling_machine/"

file1 = open("darknet/custom_data/test.txt","r")
file=file1.read().split('\n')
file1.close()
file=file[0:len(file)-1]



k=0
for i in file:
  
    img = cv2.imread(i)
    img = cv2.resize(img, None, fx=0.9,fy=0.9)   
    img = cv2.resize(img, (416, 416)) 
    height, width, channels = img.shape  
    
    file2 = open(label_dir+i.split("/")[4].split(".")[0]+".txt","r")
    file1=file2.read().split('/n')
    file2.close()
    df=df.append(pd.Series(), ignore_index=True)
    
    m=1
    for line in file1:
        df.iloc[k,df.columns.tolist().index('name')]=i.split("/")[4]
        detection=[float(line2) for line2 in line.split()][1:]
        df.iloc[k,df.columns.tolist().index('bounding_box_no')]=m
        
        center_x = int(detection[0] * width)
        center_y = int(detection[1] * height)
        w = int(detection[2] * width)
        h = int(detection[3] * height)
        # Rectangle coordinates
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)
        df.iloc[k,df.columns.tolist().index('xmin')]=x
        df.iloc[k,df.columns.tolist().index('ymin')]=y
        df.iloc[k,df.columns.tolist().index('xmax')]=x+w
        df.iloc[k,df.columns.tolist().index('ymax')]=y+h
        m+=1
    
    
 
    print('{}. '.format(k))
    k+=1
        
    
df.to_csv('ground_truth_sheet.csv',index=False)
    
    






