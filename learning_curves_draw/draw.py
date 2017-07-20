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
   filename='train_log.txt'
#rcnn 
#    loss_bbox =np.array( extract_loss(filename,'#0',80))
#    loss_cls = np.array(extract_loss(filename,'#1',79))


#    loss_bbox =np.array( extract_loss(filename,'#2',85))
#    loss_cls = np.array(extract_loss(filename,'#3',86))
#p3
   loss_bbox =np.array( extract_loss(filename,'#2',86))
   loss_cls = np.array(extract_loss(filename,'#6',87))
   

   x=np.array(range(1,len(loss_cls)+1,1))
   
   z1 = np.polyfit(x, loss_cls, 50)
   z2 = np.polyfit(x, loss_bbox, 50)
   p1 = np.poly1d(z1)
   p2 = np.poly1d(z2)
   plt.figure()
   plt.plot(x, p1(x),label="loss_cls")
   plt.plot(x, p2(x),label="loss_bbox")
#    xnew = np.linspace(x.min(),x.max(), 100) 
#    loss_cls= spline(x,loss_cls, xnew)
#    loss_bbox = spline(x, loss_bbox, xnew)
#  
#    plt.figure()
#   
#    plt.plot(xnew,loss_cls ,label="loss_cls/p5")
#    plt.plot(xnew, loss_bbox, label="loss_bbox/p5")
# 
   plt.grid(True)
   plt.legend()
   plt.show()


