# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os
import sys
from skimage.measure import compare_ssim
import argparse
import sys
import cv2
import numpy as np
import imutils
# 
import argparse

THRESHOLD = 0.7

argparser = argparse.ArgumentParser(description='Weight-DAgger')
argparser.add_argument('--iter', default=0, type=int, help='weight update iteration variable')
args = argparser.parse_args()
ITER = args.iter

if ITER == 0:
    print('arg must larger then 0 !')
    sys.exit()
elif ITER == 1:
    x_dataA = np.load('./DATA/EnsembleDAgger'+str(ITER)+'/seg.npy')
    y_dataA = np.load('./DATA/EnsembleDAgger'+str(ITER)+'/label.npy')[:,:3]

    x_dataB = np.load('./DATA/BC/seg.npy')
    y_dataB = np.load('./DATA/BC/label.npy')[:,:3]
else:
    x_dataA = np.load('./DATA/WeightDAgger'+str(ITER)+'/seg.npy')
    y_dataA = np.load('./DATA/WeightDAgger'+str(ITER)+'/label.npy')[:,:3]

    x_dataB = np.load('./DATA/WeightDAgger'+str(ITER - 1)+'/seg.npy')
    y_dataB = np.load('./DATA/WeightDAgger'+str(ITER - 1)+'/label.npy')[:,:3]

print("x_DataA.shape", x_dataA.shape)
print("x_DataB.shape", x_dataB.shape)

dir_ = str("./DATA/WeightDAgger"+str(ITER))
dir_image = str("./DATA/WeightDAgger"+str(ITER)+'/image/')
if not(os.path.isdir(dir_)):
    os.makedirs(os.path.join(dir_))
if not(os.path.isdir(dir_image)):
    os.makedirs(os.path.join(dir_image))

for i in range(len(x_dataA)):  # DAgger #
    grayA = x_dataA[i].astype('float32')*255

    for ii in range(len(x_dataB)):  # DAgger #-1 or BC
        grayB = x_dataB[ii].astype('float32')*255
        similarity_score = compare_ssim(grayA, grayB)

        if THRESHOLD <= similarity_score:
            if y_dataB[ii, 2] < y_dataA[i, 2]:
                y_dataB[ii, 2] = y_dataA[i, 2]  # change BC weight <<-- DAgger1 weight
                name = 'i_'+str(i)+'_'+str(ii)+'_w:'+str(np.round(y_dataB[ii, 2], 2))+'_s:'+str(np.round(similarity_score, 2))
            else:
                y_dataA[i, 2] = y_dataB[ii, 2]  # change DAgger2 weight <<-- DAgger1 weight
                name = 'ii_'+str(i)+'_'+str(ii)+'_w:'+str(np.round(y_dataA[i, 2], 2))+'_s:'+str(np.round(similarity_score, 2))

            ### save img
            print(name)
            img = cv2.hconcat([x_dataA[i].astype('float32')*255, x_dataB[ii].astype('float32')*255])
            img = cv2.resize(img, (800, 400), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dir_image+name+'.jpg', img)
  
x_data = np.append(x_dataB, x_dataA, axis=0)  # Aggregate seg_image
y_data = np.append(y_dataB, y_dataA, axis=0)  # Aggregate weight

print("x_data.shape", x_data.shape)
print("y_data.shape", y_data.shape)

np.save(dir_+"/seg.npy", np.array(x_data))
np.save(dir_+"/label.npy", np.array(y_data))

print("-- weight_update finish !")

sys.exit()