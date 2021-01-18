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
from pygame.locals import K_s

import utils_jw.warping as warping
import utils_jw.vehicle_function as vehicle_function

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

rospy.init_node("play_policy_for_imitation_learning")

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

curr_v = 0.0
localizationData = []
def localizationData_callback(msg):
    global localizationData, curr_v
    meter2pixel = 4.0*1.221
    localizationData.append([397-meter2pixel*(msg.data[1]-190), meter2pixel*(msg.data[0]-60)+75])
    curr_v = msg.data[3]

sub_seg = rospy.Subscriber("/front_usb_cam/image_seg", Image, seg_callback)    
sub_stateImg = rospy.Subscriber("/front_usb_cam/image_green", Image, green_callback)    
sub_front = rospy.Subscriber("/front_usb_cam/image_raw", Image, img_callback)    
sub_localizationData = rospy.Subscriber("/LocalizationData", Float32MultiArray, localizationData_callback) 

pub_control = rospy.Publisher('learningData', Float32MultiArray, queue_size = 1)    # JW

START_POLICY = False
_tau, _chi = 0.05, 0.05
_chi_dotGain = 3.0

def main():
    global START_POLICY
    if segCall is not None and stateImgCall is not None:
        seg_image, green_image = segCall, stateImgCall

        if pygame.key.get_pressed()[K_s]:
            START_POLICY = True

        steerNet, green_image, lookAheadPtNetX, lookAheadPtNetY, sigma, chi_hat = networkPolicy(seg_image, green_image)

        controlData = Float32MultiArray()
        controlData.data = [steerNet, 1.0-lookAheadPtNetY, lookAheadPtNetX, lookAheadPtNetY]

        if START_POLICY:
            pub_control.publish(controlData)

        seg_image = cv2.resize(seg_image, (25, 25), interpolation=cv2.INTER_CUBIC)/255.0
        seg_image = cv2.resize(seg_image*255.0, (200, 200), interpolation=cv2.INTER_AREA)
        _, seg_image = cv2.threshold(seg_image, 199, 255, cv2.THRESH_BINARY)
        seg_image = cv2.cvtColor(seg_image.astype('uint8'), cv2.COLOR_GRAY2RGB)
        
        if frontCall is not None:
            front_image = frontCall
            cv2.putText(front_image, str(round(curr_v, 1)) + ' km/h', (171, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            total_image = cv2.vconcat([front_image, cv2.hconcat([seg_image, green_image])])
            if map is not None:
                _map = cv2.resize(map, (400, (int)(460*400/955)+1), interpolation=cv2.INTER_CUBIC)
                cv2.polylines(_map, [np.array(localizationData, np.int32).reshape(-1, 1, 2)], False, (0, 255, 0), 2)
                total_image = cv2.vconcat([total_image, _map])
            cv2.imshow('total_image', total_image)

        cv2.waitKey(1)


def networkPolicy(seg_image, green_image):
    seg_image = np.array([cv2.resize(seg_image, (25, 25), interpolation=cv2.INTER_CUBIC)])
    seg_image = seg_image.reshape(seg_image.shape[0], seg_image.shape[1], seg_image.shape[2], 1).astype('float32')/255.0
    network_output = TRAINED_POLICY.predict(np.array(seg_image))

    mean = network_output[:, :2]
    sigma = np.exp(network_output[:, 2:])*3.0
    # print('mean: ', mean[0], ', sigma: ', sigma[0])

    steer, green_image, lookAheadPtXRaw, lookAheadPtYRaw = vehicleControl(mean[0][0], mean[0][1], sigma[0][0]*3, sigma[0][1]*1.5, green_image, True)
    chi = math.sqrt((sigma[0][0] * sigma[0][0]) + (sigma[0][1] * sigma[0][1]))

    return steer, green_image, lookAheadPtXRaw, lookAheadPtYRaw, sigma[0], chi


def vehicleControl(lookAheadPtXRaw, lookAheadPtYRaw, sigmaX, sigmaY, green_image, NetworkPolicy):
    lookAheadPtXRaw = np.clip(lookAheadPtXRaw, 0.0, 1.0)
    lookAheadPtYRaw = np.clip(lookAheadPtYRaw, 0.0, 1.0)
    lookAheadPtXImg = (int)(lookAheadPtXRaw*green_image.shape[1])
    lookAheadPtYImg = (int)(lookAheadPtYRaw*green_image.shape[0])

    lookAheadPtX = abs(lookAheadPtYRaw - 1.0) * REAL_RANGE + IMG_X2VEHICLE_X
    lookAheadPtY = (lookAheadPtXRaw - 0.5) * REAL_RANGE*2
    steer = vehicle_function.purePursuit(lookAheadPtX, lookAheadPtY)
    vel = lookAheadPtX/2.24
    if vel < 3:
        vel = 3
    time = 1.5
    
    _, listL, listR = vehicle_function.KinematicPrediction(steer*45.0/540.0, time, vel, green_image.shape[0], REAL_RANGE, IMG_X2VEHICLE_X*2)
 
    if NetworkPolicy:  # from Network
        cv2.circle(green_image, (lookAheadPtXImg, lookAheadPtYImg), 7, (0,0,255), 2)
        cv2.line(green_image, (lookAheadPtXImg-(int)(sigmaX*green_image.shape[1]), lookAheadPtYImg), (lookAheadPtXImg+(int)(sigmaX*green_image.shape[1]), lookAheadPtYImg), (0, 0, 255), 2)
        cv2.line(green_image, (lookAheadPtXImg, lookAheadPtYImg-(int)(sigmaY*green_image.shape[0])), (lookAheadPtXImg, lookAheadPtYImg+(int)(sigmaY*green_image.shape[0])), (0, 0, 255), 2)

        cv2.polylines(green_image, [np.array(listL, np.int32).reshape(-1, 1, 2)], False, (0, 0, 255), 2)
        cv2.polylines(green_image, [np.array(listR, np.int32).reshape(-1, 1, 2)], False, (0, 0, 255), 2)
    
    return steer, green_image, lookAheadPtXRaw, lookAheadPtYRaw

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
    argparser.add_argument('--policy', default='---', help='select the policy to play')
    args = argparser.parse_args()
    ITER_NUM = args.policy

    map = cv2.imread('./DATA/map.png', cv2.IMREAD_COLOR)

    TRAINED_POLICY = keras.models.load_model('./RUNS/' + str(ITER_NUM) + '/trained_policy.hdf5', compile=False)
    print(TRAINED_POLICY.summary())

    REAL_RANGE = 12.0       
    IMG_X2VEHICLE_X = 0.5   

    print('- Press "s" on the pygame window, then the vehcile move toward the look-ahead point (red point).')

    execute()
