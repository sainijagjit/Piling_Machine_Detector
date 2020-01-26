import os

txt_dir="/home/sainijagjit/Desktop/Darknet/customindz/darknet/custom_data/images/labels/"
img_dir="/home/sainijagjit/Desktop/Darknet/customindz/darknet/custom_data/images/piling_machine/"

# Remove empty annotation files
for i in os.listdir(txt_dir):
    f = open(txt_dir+i,'r')
    x = f.read()
    if(x==''):
        os.remove(txt_dir+i)
        print(txt_dir+i+"removed")

# Remove images which are not having drill (object to detect)
for i in os.listdir(img_dir):
    k = i.split('.')
    l = k[0]+".txt"
    if l not in os.listdir(txt_dir):
        os.remove(img_dir+i)
        print(img_dir+i)
        
#to split data between train and test set


arr=[]
for i in os.listdir(img_dir):
    arr.append([img_dir+i])
    

train=arr[0:int(round(0.8*len(arr)))]
test=arr[int(round(0.8*len(arr)))+1:]

for i in train:
    print(i[0])
    file1 = open("train.txt","a")       
    file1.write(i[0]+'\n')
file1.close()

for i in test:
    print(i[0])
    file1 = open("test.txt","a")       
    file1.write(i[0]+'\n')
file1.close()



#replacing the 15 with 0
for i in os.listdir(img_dir):
    if filename.endswith(".jpg"):
        continue