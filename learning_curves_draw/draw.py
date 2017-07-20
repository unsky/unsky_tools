#!/usr/bin/env python
import datetime
import os
import sys
import numpy as np

import matplotlib.pyplot as plt
from scipy.interpolate import spline

def extract_loss(input_file,flag,num):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    loss = []
    for line in lines:
        line = line.strip()
        if line.find(flag) != -1:
            line = line[num:]
            print (line)
            index = line.find('(')
            loss.append(float(line[:index-1]))
    return loss



if __name__ == '__main__':
   filename='p2.txt'
#rcnn 
   loss_bbox_rcnn =np.array( extract_loss(filename,'#0',80))
   loss_cls_rcnn = np.array(extract_loss(filename,'#1',79))

#p2
   loss_bbox_p2 =np.array( extract_loss(filename,'#2',86))
   loss_cls_p2 = np.array(extract_loss(filename,'#6',87))

#p3
   loss_bbox_p3 = np.array( extract_loss(filename,'#3',86))
   loss_cls_p3 = np.array(extract_loss(filename,'#7',87))
   
#p4
   loss_bbox_p4 =np.array( extract_loss(filename,'#4',86))
   loss_cls_p4 = np.array(extract_loss(filename,'#8',87))
   
#p5
   loss_bbox_p5 =np.array( extract_loss(filename,'#5',86))
   loss_cls_p5 = np.array(extract_loss(filename,'#9',87))
   
   loss_bbox = loss_bbox_p2 + loss_bbox_p3 + loss_bbox_p4 + loss_bbox_p5
   loss_cls = loss_cls_p2 + loss_cls_p3 + loss_cls_p4 + loss_cls_p5
   

   x=np.array(range(1,len(loss_cls)+1,1))
   
   z1 = np.polyfit(x, loss_cls, 50)
   z2 = np.polyfit(x, loss_bbox, 50)
   p1 = np.poly1d(z1)
   p2 = np.poly1d(z2)
   
   plt.figure(1)
   plt.subplot(211)
   plt.plot(x, p1(x),label="loss_cls")
   plt.plot(x, p2(x),label="loss_bbox")
   plt.grid(True)
   plt.legend()
   
   z1_rcnn = np.polyfit(x, loss_cls_rcnn, 50)
   z2_rcnn = np.polyfit(x, loss_bbox_rcnn, 50)
   p1_rcnn = np.poly1d(z1_rcnn)
   p2_rcnn = np.poly1d(z2_rcnn)
   
   plt.subplot(212)
   plt.plot(x, p1_rcnn(x),label="rcnn_cls_loss")
   plt.plot(x, p2_rcnn(x),label="rcnn_bbox_loss")

   plt.grid(True)
   plt.legend()
   plt.show()


