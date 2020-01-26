import pandas as pd , argparse
import os
from shapely.geometry import Polygon
import math 

def cal_score(ground_truth,predict,thd):
    ###############################################################################
    def calculate_iou(xmin,ymin,xmax,ymax,xmin_p,ymin_p,xmax_p,ymax_p):
        poly_1 = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
        poly_2 = Polygon([(xmin_p, ymin_p), (xmax_p, ymin_p), (xmax_p, ymax_p), (xmin_p, ymax_p)])
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        return iou
    ###############################################################################
    # Tester
    df_gt=pd.read_csv(ground_truth)
    df_pred=pd.read_csv(predict)
    df_gt=df_gt.rename(columns = {'bounding_box_no':'b_no'})
    
    #df_test=pd.merge(args['root_test'][0],df_pred,on='name',how='right') 
    df_test=pd.merge(df_gt,df_pred,on='name',how='right') 
    
    list=df_pred.columns
    
    n=int(len(list)/4)
    ###############################################################################
    # For True Neg and False Positive addition (Fp) is perfect here
    
    true_pred_box = []
    
    dk = df_test.fillna(20000000)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    emp_frame = 0
    for i, row in dk.iterrows():
        if (row['b_no'] == 20000000 and row['xmin1'] == 20000000):
            emp_frame = emp_frame + 1
        elif (row['b_no'] == 20000000 and row['xmin1'] != 20000000):
            fp = fp + 1
        elif (row['b_no'] != 20000000 and row['xmin1'] == 20000000):
            fn += 1
    ###############################################################################
    df_test = df_test.dropna(axis=0, subset=['b_no'])
    ###############################################################################
    
    ###############################################################################
    front_face = 0
    side_face = 0
    head_face = 0
    down_face=0
    all_gt_boxes = []
    all_pred_boxes = []
    true_gt = []
    
    for index, row in df_test.iterrows():
    
        #############################
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']
        gt_box = [xmin, ymin, xmax, ymax]
        all_gt_boxes.append(gt_box)
        #############################
        x_centroid = (row['xmin'] + row['xmax']) / 2
        y_centroid = (row['ymin'] + row['ymax']) / 2
        ############################################################
    ###########################################
        for j in range(1, n + 1):
    
            ########################
            xmin_p = 'xmin' + str(j)
            ymin_p = 'ymin' + str(j)
            xmax_p = 'xmax' + str(j)
            ymax_p = 'ymax' + str(j)
            ########################
            x_pred_min = row[xmin_p]
            y_pred_min = row[ymin_p]
            x_pred_max = row[xmax_p]
            y_pred_max = row[ymax_p]
            ############################################
            x_centroid_p = (x_pred_min + x_pred_max) / 2
            y_centroid_p = (y_pred_min + y_pred_max) / 2
            ############################################
            if not pd.isna(x_pred_min):
                pred_box = [x_pred_min, y_pred_min, x_pred_max, y_pred_max]
                all_pred_boxes.append(pred_box)
                ###########################################################################################
                iou = calculate_iou(xmin, ymin, xmax, ymax, x_pred_min, y_pred_min, x_pred_max, y_pred_max)
                ###########################################################################################
    
                if (iou > thd):
                    tp += 1
                    true_gt.append(gt_box)
                    true_pred_box.append(pred_box)
    
    #################################################
    unique_all_pred_boxes = []
    for item in all_pred_boxes:
        if item not in unique_all_pred_boxes:
            unique_all_pred_boxes.append(item)
    
    unique_true_pred_box=[]
    for item in true_pred_box:
        if item not in unique_true_pred_box:
            unique_true_pred_box.append(item)
    
    
    
    #################################################
    fn = len(all_gt_boxes) - len(true_gt)
    fp = len(unique_all_pred_boxes) - len(unique_true_pred_box)
    total_faces = front_face + side_face + head_face +down_face
    ###############################################################################
    #print('confusion_metrics=')
    #print(tp,fn)
    #print(fp,tn)
           
    
    acc=((tp+tn)/(tp+tn+fp+fn))*100 
    # print('accuracy=',acc)
    if(tp==0):
        recall=0
    else:
        recall=tp/(tp+fn)
    # print('recall=',recall)
    if(tp==0):
        precision=0
    else:
        precision=tp/(tp+fp)
    # print('precision=',precision)
    if(precision==0 and recall==0):
        f_measure=0
    else:
        f_measure=(2*recall*precision)/(recall+precision)
    # print('f_measure=',f_measure)
    
    true_p=tp
    false_p=fp
    false_n=fn
    ###############################################################################
    
    # print("returning in this order --- accuracy,recall,precision,f_measure")
    return (acc,recall,precision,f_measure,tp,fn,fp,tn)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


