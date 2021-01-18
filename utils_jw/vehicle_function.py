#-*- coding: utf-8 -*-
##
import os
import sys
import math
import argparse
import numpy as np
import cv2

###### Pure Pursuit #######
def purePursuit(lookAheadPtX, lookAheadPtY):
    ratio_s2w = 18.6
    L = 2.845       # front wheel base -> rear wheel base
    Lfw = 0.3

    eta = math.atan2(lookAheadPtY, lookAheadPtX)
    Lw = math.sqrt(lookAheadPtX**2 + lookAheadPtY**2)

    steerAngle = ratio_s2w*math.atan((L * math.sin(eta)) / (Lw*1.5 / 2 + Lfw*math.cos(eta)))

    steerAngle = math.degrees(steerAngle)
    max_steering_angle = 540.0
    if abs(steerAngle) > max_steering_angle:
        steerAngle = np.sign(steerAngle) * max_steering_angle

    return steerAngle

def KinematicPrediction(steer, t, v, imageSize, imgRange, IMG_X2VEHICLE_X):
    real2pix = imageSize/imgRange
    dt = 0.2
    dist_w2w = 2.0*1.27
    # gain = 0.33
    
    trajC = []
    trajL = []
    trajR = []
    cX = (int)(imageSize/2.0)
    cY = (int)(imageSize - 1) + IMG_X2VEHICLE_X*real2pix
    
    x = y = yaw = 0.0
    for i in range(0, (int)(t/dt)):
        x += dt*v*math.cos(yaw)
        y += dt*v*math.sin(yaw)
        yaw += dt*v*math.atan2(math.radians(steer), 1.0)/2.845

        tX = (int)(cX + real2pix * y)
        tY = (int)(cY - real2pix * x)

        if tX >= imageSize or tX < 0 or tY >= imageSize or tY < 0:
            continue

        trajC.append((tX, tY))
        trajL.append((tX - real2pix*dist_w2w/2, tY))
        trajR.append((tX + real2pix*dist_w2w/2, tY))
    return trajC, trajL, trajR


def CTRAprediction(steer, v, imageSize, imgRange, IMG_X2VEHICLE_X):
    meter2pixel = imageSize/imgRange

    steer = math.radians(steer)
    R = h = h_pre = 0
    acc_time_on = acc_time_off = 1.0
    acc_time_zero = 1.0

    acc_data_on = acc_data_off = 0.1
    acc_data_zero = 0.01

    xOffset = 0.1
    x = xOffset
    y = 0.0
    dt = 12
    gain =(imageSize/2)/imageSize

    dist_f2r = 2.6
    dist_c2r = 1.3
    dist_w2w = 1.95*meter2pixel
    wheel_angle = steer/18.6

    try:
        RR = math.sqrt(dist_c2r*dist_c2r + (dist_f2r / (float)(math.tan(wheel_angle)))*(dist_f2r / (float)(math.tan(wheel_angle))))
    except ZeroDivisionError:
        RR = 0

    if steer < 0:
        RR *= -1

    CTRAlistL = []
    CTRAlistR = []

    for time in range(0, v):
        R += 1
        if abs(steer) > 0:
            if  time >= 0 and time <= acc_time_on:
                a_long = acc_data_on
                pre_v = v_long = acc_data_on * time
            elif time > acc_time_on and time <= acc_time_on + acc_time_zero:
                a_long = acc_data_zero
                v_pre = v_long = acc_data_zero * (time-acc_time_on) + pre_v
            elif time > acc_time_on + acc_time_zero and time <= acc_time_on + acc_time_zero + acc_time_off:
                a_long = acc_data_off
                v_long = -acc_data_off * (time - acc_time_on - acc_time_zero) + v_pre + pre_v

            if v_long > 0:
                w = v_long / RR
                h = h_pre + w * dt
                y = y + (v_long * (math.sin(h) - math.sin(h_pre)) * 1/w) + (a_long*(math.cos(h) - math.cos(h_pre)) + a_long * math.sin(h)*w*dt)*(1/w*w)
                x = x - (v_long * (math.cos(h) - math.cos(h_pre)) * 1/w) + (a_long*(math.sin(h) - math.sin(h_pre)) - a_long * math.cos(h)*w*dt)*(1/w*w)
            h_pre = h
        else:
            if time >= 0 and time <= acc_time_on:
                v = acc_data_on * time
                pre_v = v
            elif time > acc_time_on and time <= acc_time_on+acc_time_zero:
                v = acc_data_zero * (time-acc_time_on) + pre_v
                v_pre = v
            elif time > acc_time_on+acc_time_zero and time <= acc_time_on + acc_time_zero + acc_time_off:
                v = -acc_data_off * (time - acc_time_on - acc_time_zero) + v_pre + pre_v

            y = y + v*dt
            x = xOffset

        
        CTRAlistL.append( ((imageSize/2 + (x*meter2pixel - dist_w2w/2)), (imageSize - gain*(y-IMG_X2VEHICLE_X)*meter2pixel)) )
        CTRAlistR.append( ((imageSize/2 + (x*meter2pixel + dist_w2w/2)), (imageSize - gain*(y-IMG_X2VEHICLE_X)*meter2pixel)) )
    return CTRAlistL, CTRAlistR

if __name__ == '__main__':
    print('main')

