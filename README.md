# Piling Machine Detector

![209.jpg](https://raw.githubusercontent.com/sainijagjit/Piling_Machine_Detector/master/output/209.jpg)

![218.jpg](https://raw.githubusercontent.com/sainijagjit/Piling_Machine_Detector/master/output/218.jpg)



# Data Scrapping

1. Image Scrapping :
	File Location customidz/image_scrapping.py
 Replace variable `search_term` value with the search string. Replace number of images in                   search_and_download() function argument at line 142.

It will download images from google and save in images/<search_term> directory.
Selenium library is used

2. Video Scrapping:
	File Location customidz/video_scrapping.py
 Replace variable `search_term` value with the search string. Replace number of videos in                   search_and_download() function argument at line 29.
It will download videos from youtube and save in images/<search_term> directory.
Youtube API is used


# Data

Images dir : darknet/custom_data/images/piling_machine
Labels dir : darknet/custom_data/labels/piling_machine

Training images (0.8) : darknet/custom_data/train.txt
Test images (0.2) : darknet/custom_data/test.txt

Config File : darknet/custom_data/cfg/yolov3-custom.cfg

Weights  dir: darknet/custom_data/backup

Predicted Output : /output



# Steps to Test (on Test images)

1. Make the ground truth sheet from labels using ground_truth_sheet.py, it will create a csv file
    ground_truth_sheet.csv [already created, no need to re-run the program]
2. open detector.py file and do the following changes:
    • Change cust_dir path variable at line 132 to point to customindz folder.
    • For getting Confusion Matrix,Accuracy, Recall, Precision, F1_score change
change view argument to `False` in prediction() function at line 133.

    • It will create prediction_sheet.csv file and will calculate the the metrics by calling
		testing.py file.
		It will create a file report.csv, where you can see accuracy at different iou 		            thresholds.

    • For viewng test images along with predicted bounding box,
 		change view argument to `True` in prediction() function at line 133.
		prediction(cust_dir,view=True)



# Detect Your own Image :

1. Open single_file_detector.py and Update Cust_dir variable path to customindz folder at line 102.
2. Give image path in detector() function argument at line 103.

# Video Feed:

To see video output, change variable cap at line 28 of existing_video_feed.py file to your video
link.	  





