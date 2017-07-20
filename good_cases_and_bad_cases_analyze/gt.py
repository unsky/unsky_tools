import cv2
import datetime
import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
import pickle
input_file = 'test.txt'
with open(input_file, 'r') as f:
    lines = f.readlines()
    loss = []

i = 0
for line in lines:
    line = line.strip()
    filename = 'Annotations_groundtruth/'+ line +'.xml'
    txtname = 'annotations_generated/'+str(i)+'.txt'
    print (filename)
    in_file = open(filename)
    imgname = 'JPEGImages/' + line + '.png'
    im = cv2.imread(imgname)
    tree=ET.parse(in_file)
    root = tree.getroot()
    print ('--------------')
    obj_key = 0
    d = 0
    for obj in root.iter('object'):
        xmlbox = obj.find('bndbox')
        x1 = int(xmlbox.find('xmin').text)
        y1 = int(xmlbox.find('ymin').text)
        x2 = int(xmlbox.find('xmax').text)
        y2 = int(xmlbox.find('ymax').text)
        with open(txtname, 'r') as f:
             txts = f.readlines()
        obj_key = obj_key + 1
        for txt in txts:       
            txt = txt[1:-2]
            txt = txt.split()
            
            score = float(txt[4][0:-1])
            print (score)
            if score >0.0:
                bb0 = int(float(txt[0][0:-1]))
                bb1 = int(float(txt[3][0:-1]))
                bb2 = int(float(txt[2][0:-1]))
                bb3 = int(float(txt[1][0:-1]))
                ixmin = np.maximum(x1, bb0)
                iymin = np.maximum(y1, bb1)
                ixmax = np.minimum(x2, bb2)
                iymax = np.minimum(y2, bb3)
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih
            # union
                uni = ((bb2 - bb0 + 1.) * (bb3 - bb1 + 1.) +
                          (x2 - x1 + 1.) *
                          (y2 - y1 + 1.) - inters)
                overlaps = inters / uni
                print (overlaps)
                if overlaps > 0:
                     cv2.rectangle(im, (int(float(txt[0][0:-1])), int(float(txt[1][0:-1]))), (int(float(txt[2][0:-1])),int(float(txt[3][0:-1]))),(0, 255, 0), 2)
                     d = d+1
        cv2.rectangle(im, (x1, y1), (x2,y2),(255, 0, 0), 2)
    if d == obj_key:
        name = 'goods/' + str(i) +'.png'
        cv2.imwrite(name, im)
    else:
         name = 'bads/' + str(i) +'.png'
         cv2.imwrite(name, im)
    i= i+1