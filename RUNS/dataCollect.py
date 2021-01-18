# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os
import sys
import math
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Header
import numpy as np
import scipy.misc
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
##
import keras
from keras.models import load_model, Model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
##
import argparse
import random
import time
import pygame
import numpy as np
from pygame.locals import K_f
from pygame.locals import K_s

import utils_jw.warping as warping
import utils_jw.vehicle_function as vehicle_function

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

rospy.init_node("Data_collection_for_imitation_learning")

segCall = stateImgCall = frontCall = None
def seg_callback(msg):
    global segCall
    segCall = np.fromstring(msg.data, dtype = np.uint8).reshape(200, 200, 1)

def green_callback(msg):
    global stateImgCall
    stateImgCall = np.fromstring(msg.data, dtype = np.uint8).reshape(200, 200, 3)

def img_callback(msg):
    global frontCall
    frontCall = np.fromstring(msg.data, dtype = np.uint8).reshape(125, 400, 3)

sub_seg = rospy.Subscriber("/front_usb_cam/image_seg", Image, seg_callback)    
sub_stateImg = rospy.Subscriber("/front_usb_cam/image_green", Image, green_callback)    
sub_front = rospy.Subscriber("/front_usb_cam/image_raw", Image, img_callback)    

pub_control = rospy.Publisher('learningData', Float32MultiArray, queue_size = 1)

START_COLLECT = False
_tau, _chi = 0.05, 0.05
_chi_dotGain = 3.0

def main():
    global START_COLLECT
    if segCall is not None and stateImgCall is not None:
        seg_image, state_image = segCall, stateImgCall

        if pygame.key.get_pressed()[K_s]:
            START_COLLECT = True

        if ITER != 'BC':   # DAgger
            steerNet, state_image, lookAheadPtNetX, lookAheadPtNetY, sigma, chi_hat = networkPolicy(seg_image, state_image)
        steerExp, state_image, lookAheadPtExpX, lookAheadPtExpY = expertPolicy(seg_image, state_image)
        controlData = Float32MultiArray()

        if ITER != 'BC':   # DAgger
            tau_hat = math.sqrt((lookAheadPtNetX - lookAheadPtExpX)**2 + (lookAheadPtNetY - lookAheadPtExpY)**2)
            deltaLookAheadPtX, deltaLookAheadPtY = abs(lookAheadPtExpX - lookAheadPtNetX), abs(lookAheadPtExpY - lookAheadPtNetY)
            deltaDir = math.atan2(deltaLookAheadPtY, deltaLookAheadPtX)*180.0/3.14
        
            ### original Ensemble Dagger
            if _tau < tau_hat or _chi < chi_hat/_chi_dotGain:   # Expert Action
                collectData(seg_image, steerExp, lookAheadPtExpX, lookAheadPtExpY, error=tau_hat)
                controlData.data = [steerExp, 1.0-lookAheadPtExpY, lookAheadPtExpX, lookAheadPtExpY, lookAheadPtNetX, lookAheadPtNetY, sigma[0], sigma[1], tau_hat, chi_hat, 1.0]
                cv2.putText(state_image, str(len(label))+', a_EXP', (0, 9), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0,255,255), 1)
            else:   # Network Action
                controlData.data = [steerNet, 1.0-lookAheadPtNetY, lookAheadPtExpX, lookAheadPtExpY, lookAheadPtNetX, lookAheadPtNetY, sigma[0], sigma[1], tau_hat, chi_hat, 0.0]
                cv2.putText(state_image, str(len(label))+', a_Net', (0, 9), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0,0,255), 1)
        else:   # Behavior Cloning
            collectData(seg_image, steerExp, lookAheadPtExpX, lookAheadPtExpY, error=0.0)
            controlData.data = [steerExp, 1.0-lookAheadPtExpY, lookAheadPtExpX, lookAheadPtExpY]
            cv2.putText(state_image, str(len(label)), (0, 9), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255,255,255), 1)

        if START_COLLECT:
            pub_control.publish(controlData)

        cv2.imshow('state_image', state_image)
        
        if frontCall is not None:
            image_front = frontCall
            cv2.imshow('front', image_front)

        cv2.waitKey(1)

def networkPolicy(seg_image, state_image):
    seg_image = np.array([cv2.resize(seg_image, (25, 25), interpolation=cv2.INTER_CUBIC)])
    seg_image = seg_image.reshape(seg_image.shape[0], seg_image.shape[1], seg_image.shape[2], 1).astype('float32')/255.0
    network_output = TRAINED_POLICY.predict(np.array(seg_image))

    mean = network_output[:, :2]
    sigma = np.exp(network_output[:, 2:])*_chi_dotGain
    # print('mean: ', mean[0], ', sigma: ', sigma[0])

    steer, state_image, lookAheadPtXRaw, lookAheadPtYRaw = vehicleControl(mean[0][0], mean[0][1], sigma[0][0], sigma[0][1], state_image, True)
    chi = math.sqrt((sigma[0][0] * sigma[0][0]) + (sigma[0][1] * sigma[0][1]))

    return steer, state_image, lookAheadPtXRaw, lookAheadPtYRaw, sigma[0], chi

def expertPolicy(seg_image, state_image):
    cv2.namedWindow('state_image')
    cv2.setMouseCallback('state_image', draw_circle)
    cv2.line(state_image, (99, 0), (99, 199), (0, 0, 255), 1)
    lookAheadPtXRaw, lookAheadPtYRaw = lookAheadPtXImg/state_image.shape[1], lookAheadPtYImg/state_image.shape[0]
    steer, state_image, lookAheadPtXRaw, lookAheadPtYRaw = vehicleControl(lookAheadPtXRaw, lookAheadPtYRaw, 0, 0,  state_image, False)

    return steer, state_image, lookAheadPtXRaw, lookAheadPtYRaw

_step = 0
segImage = []
label = []
lookAheadPtXImg = lookAheadPtYImg = 100
def draw_circle(event, x, y, flags, param):
    global lookAheadPtXImg, lookAheadPtYImg
    lookAheadPtXImg, lookAheadPtYImg = x, y

def collectData(seg_image, steer, lookAheadPtXRaw, lookAheadPtYRaw, error):
    global _step
    lookAheadPtXRaw = np.clip(lookAheadPtXRaw, 0.0, 1.0)
    lookAheadPtYRaw = np.clip(lookAheadPtYRaw, 0.0, 1.0)
    error = np.clip(error, 0.0, 1.0)

    if pygame.key.get_pressed()[K_f]: # or cvkey == ord('f'):
        print("label", np.info(np.array(label)))
        print("JW State", np.info(np.array(segImage)))

        dir_name = 'BC' if ITER == 'BC' else ITER

        dir_ = str("./DATA/"+dir_name)
        if not(os.path.isdir(dir_)):
            os.makedirs(os.path.join(dir_))

        np.save(dir_+"/label", np.array(label))
        np.save(dir_+"/seg", np.array(segImage))

        print('--- DATA COLLECT & SAVE FINISH !!!')
        sys.exit()

    seg_image = cv2.resize(seg_image, (25, 25), interpolation=cv2.INTER_CUBIC)/255.0
    ### collect data
    _step += 1

    step_dv = 7 if ITER == 'BC' else 2

    if START_COLLECT:
        if _step % step_dv == 0:
            _step = 0
            segImage.append(seg_image)
            label.append([lookAheadPtXRaw, lookAheadPtYRaw, error])
    
    seg_image = cv2.resize(seg_image*255.0, (200, 200), interpolation=cv2.INTER_AREA)
    _, seg_image = cv2.threshold(seg_image,199,255,cv2.THRESH_BINARY)
    cv2.imshow('seg_image', seg_image)

def vehicleControl(lookAheadPtXRaw, lookAheadPtYRaw, sigmaX, sigmaY, state_image, NetworkPolicy):
    lookAheadPtXRaw = np.clip(lookAheadPtXRaw, 0.0, 1.0)
    lookAheadPtYRaw = np.clip(lookAheadPtYRaw, 0.0, 1.0)
    lookAheadPtXImg = (int)(lookAheadPtXRaw*state_image.shape[1])
    lookAheadPtYImg = (int)(lookAheadPtYRaw*state_image.shape[0])

    lookAheadPtX = abs(lookAheadPtYRaw - 1.0) * REAL_RANGE + IMG_X2VEHICLE_X
    lookAheadPtY = (lookAheadPtXRaw - 0.5) * REAL_RANGE*2
    steer = vehicle_function.purePursuit(lookAheadPtX, lookAheadPtY)
    vel = lookAheadPtX/2.24
    if vel < 3:
        vel = 3
    time = 1.5
    
    _, listL, listR = vehicle_function.KinematicPrediction(steer*45.0/540.0, time, vel, state_image.shape[0], REAL_RANGE, IMG_X2VEHICLE_X*2)
    if NetworkPolicy:  # from Network
        cv2.circle(state_image, (lookAheadPtXImg, lookAheadPtYImg), 7, (0,0,255), 2)
        cv2.circle(state_image, (lookAheadPtXImg, lookAheadPtYImg), 7+(int)(_tau*state_image.shape[0]), (0,0,255), 1)
        
        cv2.line(state_image, (lookAheadPtXImg-(int)((_chi*_chi_dotGain*0.5)*state_image.shape[1]), lookAheadPtYImg), (lookAheadPtXImg+(int)((_chi*_chi_dotGain*0.5)*state_image.shape[1]), lookAheadPtYImg), (0, 0, 0), 1)
        cv2.line(state_image, (lookAheadPtXImg, lookAheadPtYImg-(int)((_chi*_chi_dotGain*0.5)*state_image.shape[0])), (lookAheadPtXImg, lookAheadPtYImg+(int)((_chi*_chi_dotGain*0.5)*state_image.shape[0])), (0, 0, 0), 1)
        cv2.line(state_image, (lookAheadPtXImg-(int)(sigmaX*state_image.shape[1]), lookAheadPtYImg), (lookAheadPtXImg+(int)(sigmaX*state_image.shape[1]), lookAheadPtYImg), (0, 0, 255), 2)
        cv2.line(state_image, (lookAheadPtXImg, lookAheadPtYImg-(int)(sigmaY*state_image.shape[0])), (lookAheadPtXImg, lookAheadPtYImg+(int)(sigmaY*state_image.shape[0])), (0, 0, 255), 2)

        cv2.polylines(state_image, [np.array(listL, np.int32).reshape(-1, 1, 2)], False, (0, 0, 255), 2)
        cv2.polylines(state_image, [np.array(listR, np.int32).reshape(-1, 1, 2)], False, (0, 0, 255), 2)
    else:               # from Expert
        cv2.circle(state_image, (lookAheadPtXImg, lookAheadPtYImg), 7, (0,255,255), -1)
        cv2.polylines(state_image, [np.array(listL, np.int32).reshape(-1, 1, 2)], False, (0, 255, 255), 2)
        cv2.polylines(state_image, [np.array(listR, np.int32).reshape(-1, 1, 2)], False, (0, 255, 255), 2)

    return steer, state_image, lookAheadPtXRaw, lookAheadPtYRaw

def execute():
    pygame.init()
    pygame.display.set_mode((100, 100), pygame.HWSURFACE | pygame.DOUBLEBUF)
    
    try:
        while not rospy.is_shutdown():
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                main()
            except KeyboardInterrupt:
                print("Done.")
                break            
    finally:
        pygame.quit()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Weight-DAgger')
    argparser.add_argument('--iter', default='---', help='data collect iteration variable')
    args = argparser.parse_args()
    ITER = args.iter

    from keras.models import load_model
    if str(ITER) != 'BC':
        if int(ITER[-1]) == 0:
            print('--iter must larger then 0 !')
            sys.exit()
        else:
            if int(ITER[-1]) == 1:
                if str(ITER[:-1]) == 'WeightDAgger':
                    print('-- when --iter is 1, Data collection process is same as EnsembleDAgger !')
                    print('-- So, Skip this process, and try the next process, python weightUpdate.py --iter 1')
                    sys.exit()
                else:
                    TRAINED_POLICY = load_model('./RUNS/BC/trained_policy.hdf5', compile=False)
            else:
                TRAINED_POLICY = load_model('./RUNS/'+str(ITER[:-1])+str(int(ITER[-1])-1)+'/trained_policy.hdf5', compile=False)

            print(TRAINED_POLICY.summary())

    REAL_RANGE = 12.0       
    IMG_X2VEHICLE_X = 0.5   

    print("- This project is to collect the expert action with the state as dataset.")
    print("- You are the expert, and select the look-ahead point at the 'state_image' window.")
    print("- Where, the look-ahead point is the point where the vehicle reaches.")

    print("- When U ready, press 's' on the pygame window !, and the vehicle move toward the look-ahead point.")
    print("- Then the data collection is finished, press 'f' on the pygame window. The collected dataset is stored at the './DATA/--iter(arg)/seg.npy and label.npy")

    execute()