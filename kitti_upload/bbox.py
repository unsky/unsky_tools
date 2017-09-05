import cv2
import datetime
import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
import pickle
test_image_path = "testimages/"
res_path = "uptxt/"

for root, subdirs, files in os.walk('uptxt/'):
    for file in files:
        imagename='image_2/'+str(file)[:-3]+'png'
        resname ='uptxt/'+str(file)
        savename='image_res/'+str(file)[:-3]+'png'
        im = cv2.imread(imagename)
        with open(resname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                
                line = line.split()
                if float(line[15])>0.15:       
                   x1 = float(line[4])
                   y1 = float(line[5])
                   x2 = float(line[6])
                   y2 = float(line[7])
                   cv2.rectangle(im, (int(x1), int(y1)), (int(x2),int(y2)),(255, 0, 0), 2)
                   font=cv2.FONT_HERSHEY_SIMPLEX
                   text = str(line[0]) + str(line[15])
                   cv2.putText(im, text, (int(x1), int(y1)),font, 0.5, (0,0,255), 1)
        cv2.imwrite(savename, im)
        
        
        
#                    out_file = open('uptxt/%s'%(str(file)), 'a')
#                    out_file.write("car -1.00 -1 -1 " + str(x1) + " " + str(y1) + " " + str(x2)+ " "+ str(y2)+ " -1.00 -1.00 -1.00 -1000.00 -1000.00 -1000.00 -1 "+ str(float(line[15])) + "\n")

