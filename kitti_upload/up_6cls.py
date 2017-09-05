
#!/usr/bin/env python
import datetime
import os
import sys
import numpy as np

import matplotlib.pyplot as plt
from scipy.interpolate import spline

threshold_score = 0.043
h_threshold = 23

image_index = ['{:0>6}'.format(x) for x in range(0, 7518)]

for image in image_index:
    out_file = open('uptxt/%s.txt'%(image), 'a')
# for list in lists:
#      out_file = open('restxt/%s.txt'%(str(list)), 'a')
  #   out_file.write("")
      
           
with open('res/comp4_det_test_Car.txt', 'r') as f:
     lines = f.readlines()   
for index,line in enumerate(lines):
     line = line.split()
     st =[0,0,0,0,0]
     st[0] = float(line[2])
     st[1] = float(line[3])
     st[2] = float(line[4])
     st[3] = float(line[5])
     st[4] = float(line[1])
     h = np.sqrt((st[2]-st[0])*(st[3]-st[1]))
     if h>=h_threshold:
        if st[4]>threshold_score:
             out_file = open('uptxt/%s.txt'%(str(line[0])), 'a')
             out_file.write("car -1.00 -1 -1 " + str(st[0]) + " " + str(st[1]) + " " + str(st[2])+ " "+ str(st[3])+ " -1.00 -1.00 -1.00 -1000.00 -1000.00 -1000.00 -1 "+ str(st[4]) + "\n")
     
     
with open('res/comp4_det_test_Cyclist.txt', 'r') as f:
     lines = f.readlines()   
for index,line in enumerate(lines):
     line = line.split()
     st =[0,0,0,0,0]
     st[0] = float(line[2])
     st[1] = float(line[3])
     st[2] = float(line[4])
     st[3] = float(line[5])
     st[4] = float(line[1])
     h = np.sqrt((st[2]-st[0])*(st[3]-st[1]))
     if h>=h_threshold:
         if st[4]>threshold_score :
             out_file = open('uptxt/%s.txt'%(str(line[0])), 'a')
             out_file.write("cyclist -1.00 -1 -1 " + str(st[0]) + " " + str(st[1]) + " " + str(st[2])+ " "+ str(st[3])+ " -1.00 -1.00 -1.00 -1000.00 -1000.00 -1000.00 -1 "+ str(st[4]) + "\n")
     
with open('res/comp4_det_test_Pedestrian.txt', 'r') as f:
     lines = f.readlines()   
for index,line in enumerate(lines):
     line = line.split()
     st =[0,0,0,0,0]
     st[0] = float(line[2])
     st[1] = float(line[3])
     st[2] = float(line[4])
     st[3] = float(line[5])
     st[4] = float(line[1])
     h = np.sqrt((st[2]-st[0])*(st[3]-st[1]))
     if h>=h_threshold:
         if st[4]>threshold_score:
             out_file = open('uptxt/%s.txt'%(str(line[0])), 'a')
             out_file.write("pedestrian -1.00 -1 -1 " + str(st[0]) + " " + str(st[1]) + " " + str(st[2])+ " "+ str(st[3])+ " -1.00 -1.00 -1.00 -1000.00 -1000.00 -1000.00 -1 "+ str(st[4]) + "\n")
     


with open('res/comp4_det_test_Tram.txt', 'r') as f:
     lines = f.readlines()   
for index,line in enumerate(lines):
     line = line.split()
     st =[0,0,0,0,0]
     st[0] = float(line[2])
     st[1] = float(line[3])
     st[2] = float(line[4])
     st[3] = float(line[5])
     st[4] = float(line[1])
     
     h = np.sqrt((st[2]-st[0])*(st[3]-st[1]))
     if h>=h_threshold:
         if st[4]>threshold_score:
             out_file = open('uptxt/%s.txt'%(str(line[0])), 'a')
             out_file.write("tram -1.00 -1 -1 " + str(st[0]) + " " + str(st[1]) + " " + str(st[2])+ " "+ str(st[3])+ " -1.00 -1.00 -1.00 -1000.00 -1000.00 -1000.00 -1 "+ str(st[4]) + "\n")
     
with open('res/comp4_det_test_Truck.txt', 'r') as f:
     lines = f.readlines()   
for index,line in enumerate(lines):
     line = line.split()
     st =[0,0,0,0,0]
     st[0] = float(line[2])
     st[1] = float(line[3])
     st[2] = float(line[4])
     st[3] = float(line[5])
     st[4] = float(line[1])
     h = np.sqrt((st[2]-st[0])*(st[3]-st[1]))
     if h>=h_threshold:
         if st[4]>threshold_score :     
             out_file = open('uptxt/%s.txt'%(str(line[0])), 'a')
             out_file.write("truck -1.00 -1 -1 " + str(st[0]) + " " + str(st[1]) + " " + str(st[2])+ " "+ str(st[3])+ " -1.00 -1.00 -1.00 -1000.00 -1000.00 -1000.00 -1 "+ str(st[4]) + "\n")
