import cv2
import datetime
import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import spline
dir = 'An2007/'
k = []
k2 = 0
k3 = 0
k4 = 0
k5 = 0
for root, subdirs, files in os.walk(dir):
    for file in files:
            filename = dir + file 
            print (filename)
            tree=ET.parse(filename)
            root = tree.getroot()
            for obj in root.iter('object'):
                xmlbox = obj.find('bndbox')
                x1 = int(xmlbox.find('xmin').text)
                y1 = int(xmlbox.find('ymin').text)
                x2 = int(xmlbox.find('xmax').text)
                y2 = int(xmlbox.find('ymax').text)
                area = (x2 - x1) * (y2 - y1)
                key = np.floor(4 + np.log2(np.sqrt(area)*1.0 / 128))
                if key > 5:
                    key = 5
                if key < 2:
                    key = 2
                if key == 2:
                    k2 = k2 + 1
                if key == 3:
                    k3 = k3 + 1
                if key == 4:
                    k4 = k4 +1
                if key == 5:
                    k5 = k5 +1
dir = 'An2012/'
for root, subdirs, files in os.walk(dir):
    for file in files:
            filename = dir + file 
            print (filename)
            tree=ET.parse(filename)
            root = tree.getroot()
            for obj in root.iter('object'):
                xmlbox = obj.find('bndbox')
                x1 = int(xmlbox.find('xmin').text)
                
                y1 = int(float(xmlbox.find('ymin').text))
             
            
                x2 = int(xmlbox.find('xmax').text)
                y2 = int(xmlbox.find('ymax').text)
                area = (x2 - x1) * (y2 - y1)
                key = np.floor(4 + np.log2(np.sqrt(area)*1.0 / 128))
                if key > 5:
                    key = 5
                if key < 2:
                    key = 2
                if key == 2:
                    k2 = k2 + 1
                if key == 3:
                    k3 = k3 + 1
                if key == 4:
                    k4 = k4 +1
                if key == 5:
                    k5 = k5 +1
                    
                    
k = [k2,k3,k4,k5]
x=np.array(range(1,4+1,1))
plt.bar(x,k,0.4,color="green")                
plt.show()
   

 