#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image warping and distortation processcing
"""
# import rospy
# from sensor_msgs.msg import Image
import os
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

# rospy.init_node('warping')

imageCall = None

# def image_front_callback(msg):
#     global imageCall
#     tmp = np.fromstring(msg.data, dtype = np.uint8).reshape(320, 480, 3)
#     imageCall = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    
# avm_img_sub = rospy.Subscriber("front_usb_cam/image_raw", Image, image_front_callback)


def data_load(dir):
    im_path = './' + dir.get('x') + '/'
    gt_path = './' + dir.get('y') + '/'

    val_image = []
    gt_image = []
    for files in os.listdir(im_path):
        test_img = cv2.imread(im_path + files)
        _, test_img = warpingLR(test_img)
        cv2.imwrite('./aft/image_2/'+files, test_img)
        # test_img = ut.normalize(test_img, normalize_mean)
        # val_image.append(test_img)

    for files in os.listdir(gt_path):
        label_img = cv2.imread(gt_path + files)
        _, label_img = warpingLR(label_img)
        cv2.imwrite('./aft/gt_image_2/'+files, label_img)
        # label_img = ut.label_onehot(label_img, ([255, 0, 0], [255, 0, 255]))
        # gt_image.append(label_img)
        
def mat(w, h, gainX, side):
    h1 = 35
    hL = 71
    hR = hL
    gain1 = 175#191
    gain2 = 0#250
    if side is 'L':
        pts = np.array([(gainX, hL), (w-gainX-3, 0), (w, h), (0, h)], dtype = "float32")
        dst = np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype = "float32")
    elif side is 'R':
        pts = np.array([(gainX+1, 0), (w-gainX, hR), (w, h), (0, h)], dtype = "float32")
        dst = np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype = "float32")
    else:
        pts = np.array([(gain1+3, h1), (w-gain1, h1), (w-gain2, h), (gain2, h)], dtype = "float32")
        w = w-gain1*2
        dst = np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype = "float32")
        h = h-h1
        # pts = np.array([(gain1, h1), (w-gain1, h1), (w-gain1, h), (gain1, h)], dtype = "float32")
        # dst = np.array([(0, 0), (w, 0), (w-gain2, h), (gain2, h)], dtype = "float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    # print(w, h)
    return w, h, M

w_l, h_l, M_left =  mat(240, 320, 7.9, 'L')
w_r, h_r, M_right = mat(240, 320, 5.9, 'R')
w, h, M = mat(480, 240, 0, None)

def undistort(img):
    h, w = img.shape[:2]
    mtx = np.array([[226.93, 0, w/2], [0, 507.263, h/2], [0, 0, 1]])
    dist = np.array([-0.386118, 0.148859, -0.002092, 0.001841, 1.0])
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    return cv2.undistort(img, mtx, dist, None, newcameramtx)

def warpingLR(img):
    # img = cv2.resize(img, (320*2, 240), interpolation=cv2.INTER_CUBIC)

    img_L = cv2.warpPerspective(img[:,:240,:], M_left, (w_l, h_l))
    img_R = cv2.warpPerspective(img[:,240:,:], M_right, (w_r, h_r))
    warpedLRImage = np.concatenate((img_L, img_R), 1)#[:, 130:640-130,:]
    # warpedLRImage = cv2.resize(warpedLRImage, (warpedLRImage.shape[1], 170), interpolation=cv2.INTER_CUBIC)
    # warpedLRImage = undistort(warpedLRImage)
 
    warpedImage = cv2.warpPerspective(warpedLRImage, M, (w, h))#[:, 130:640-130,:]
    warpedImage = cv2.resize(warpedImage, (200, 200), interpolation=cv2.INTER_CUBIC)
    
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    kernel = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0
    warpedImage = cv2.filter2D(warpedImage, -1, kernel)

    # print(warpedImage.shape)

    return warpedLRImage, warpedImage
    # return warpedLRImage

def warping_indoor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_L = img[:,:240,:]
    w_r_, h_r_, M_right_ = mat(240, 320, 9.9, 'R')
    img_R = cv2.warpPerspective(img[:,240:,:], M_right_, (w_r_, h_r_))

    # kernel = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0
    # warpedLRImage = cv2.filter2D(warpedLRImage, -1, kernel)
    
    return np.concatenate((img_L, img_R), 1), np.concatenate((img_L[:,20:,:], img_R[:,:50,:]), 1)


if __name__ == '__main__':
    print('warping start')    
    data_load({'x': '/bef/image_2', 'y': '/bef/gt_image_2'})
    print('warping finish')
    # while not rospy.is_shutdown():
    #     try:
    #         if imageCall is not None:
    #             warpedImage, warpedLRImage = warpingLR(imageCall)
    #             cv2.imshow('orig', imageCall)
    #             cv2.imshow('warpedImage', cv2.resize(warpedImage, (w*2, h*2), interpolation=cv2.INTER_LINEAR))
    #             cv2.imshow('L R', cv2.resize(warpedLRImage, (700, 700), interpolation=cv2.INTER_LINEAR))
    #             cv2.waitKey(1)
    #     except KeyboardInterrupt:
    #         print("Done.")
    #         break
