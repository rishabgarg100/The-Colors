import cv2
import os
import numpy as np
import sys
import re
import time
import pipes
import shutil
#color frame break
path=sys.argv[1]
direc="./data/train/gray/"
os.mkdir('./data/')
os.mkdir('./data/train/')
os.mkdir(direc)
vidcap=cv2.VideoCapture(path)
success,image = vidcap.read()
count = 0
success = True
while success:
  cv2.imwrite(direc+"frame%05d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  count += 1   


#colouring
l=os.listdir(direc)
l=sorted(l)
final_link="./data/test/gray/"
os.mkdir('./data/test/')
os.mkdir(final_link)
os.mkdir("./data/test/color/")
i=0
for t in l:
    i+=1
    img=cv2.imread(direc+'/'+t)
    im=cv2.resize(img,(224,224))
    cv2.imwrite(final_link+t,im)

c="python colorize.py "+final_link
os.system(c)
    
#frames to video
cap = cv2.VideoCapture(0)
images="./data/test/color/"
kk='output3.avi'
t=os.listdir(images)
t=sorted(t)
img = cv2.imread(images+t[0])
height, width, channels = img.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("./results/"+kk,fourcc, 23.97, (width,height))
for g in t:
    img = cv2.imread(images+g)
    out.write(img)
out.release()
shutil.rmtree('data')
print("Hurray! Video coloured")



