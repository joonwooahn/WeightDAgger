#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.
"""

from __future__ import print_function

import glob
import os
import sys

egg_file_name = './carla/dist/carla-*%d.%d-%s.egg' % (sys.version_info.major, sys.version_info.minor, 'win-amd64' if os.name == 'nt' else 'linux-x86_64')
sys.path.append(glob.glob(egg_file_name)[0])

import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
from math import *
import math
import random
import re
import weakref
import time

# ROS
import rospy
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Bool, String, Header
from sensor_msgs.msg import Image

# CV2
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import pygame
import numpy as np

DEG2RAD = pi/180.0
KMpH2MpS = 0.277778

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================
class CameraManager(object):
    def __init__(self, parent_actor, gamma_correction):
        self.sensorSeg, self.sensorRGB, self.sensorFront = None, None, None
        self.imageSeg, self.imageRGB, self.imageFront = None, None, None

        self.surface = None
        self._parent = parent_actor

        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=5.7, y=0.0, z=5.7), carla.Rotation(pitch = -90.0)), Attachment.Rigid),    # top view seg
            (carla.Transform(carla.Location(x=5.7, y=0.0, z=5.7), carla.Rotation(pitch = -90.0)), Attachment.Rigid),    # top view rgb
            (carla.Transform(carla.Location(x=-0.35, y=0.0, z=1.9), carla.Rotation(pitch = -17.0)), Attachment.Rigid),    # front view rgb
            ]
        self.sensors = [
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,'Camera Semantic Segmentation', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera Front', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)

            if item[2].startswith('Camera Front'):
                bp.set_attribute('image_size_x', str(400))
                bp.set_attribute('image_size_y', str(125))
            else:
                bp.set_attribute('image_size_x', str(200))
                bp.set_attribute('image_size_y', str(200))
            
            item.append(bp)

    def set_sensor(self, index, notify=True, force_respawn=False):
        self.sensorSeg = self._parent.get_world().spawn_actor(self.sensors[0][-1], self._camera_transforms[0][0], attach_to=self._parent, attachment_type=self._camera_transforms[0][1])
        self.sensorRGB = self._parent.get_world().spawn_actor(self.sensors[1][-1], self._camera_transforms[1][0], attach_to=self._parent, attachment_type=self._camera_transforms[1][1])
        self.sensorFront = self._parent.get_world().spawn_actor(self.sensors[2][-1], self._camera_transforms[2][0], attach_to=self._parent, attachment_type=self._camera_transforms[2][1])
        
        weak_self = weakref.ref(self)
        self.sensorSeg.listen(lambda image: CameraManager._parse_image(weak_self, image, 0))  # Get true seg
        self.sensorRGB.listen(lambda image: CameraManager._parse_image(weak_self, image, 1))  # Get true RGB
        self.sensorFront.listen(lambda image: CameraManager._parse_image(weak_self, image, 2))  # Get true RGB Front

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image, ind):
        self = weak_self()
        if not self:
            return

        image.convert(self.sensors[ind][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        if ind == 0:
            self.imageSeg = np.copy(array[:, :, 0])
        elif ind == 1:
            self.imageRGB = np.copy(array[:, :, :3])
        elif ind == 2:
            self.imageFront = np.copy(array[:, :, :3])
    
# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================
class World(object):
    def __init__(self, carla_world, args):
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            sys.exit(1)

        self.actor_role_name = args.rolename
        self.player = None
        self.camera_manager = None
        self._actor_filter = args.filter
        self._gamma = args.gamma

        blueprint = self.world.get_blueprint_library().filter(self._actor_filter)[18] #ms: get Tesla//18//19///  small car: 18
        blueprint.set_attribute('role_name', self.actor_role_name)

        if args.start == 1:
            print("--- start at Start1 pose")
            self.player = self.world.try_spawn_actor(blueprint, ScenarioSpawnPose(self.world, 74.0, 201.0, 90.0)) # map1 for
        elif args.start == 2:
            print("--- start at Start2 pose")
            self.player = self.world.try_spawn_actor(blueprint, ScenarioSpawnPose(self.world, 78.0, 193.5, 180.0)) # map1 rev
        else:
            print("Please set argparser as --start 1 or 2 to set the start pose")
            sys.exit(1)
        
        self.camera_manager = CameraManager(self.player, self._gamma)
        self.camera_manager.set_sensor(0, notify=False)

    def render(self, display):
        self.camera_manager.render(display)

def ScenarioSpawnPose(world, Initial_X, Initial_Y, Initial_Angle):
    spawn_point = carla.Transform()
    spawn_point.location.x, spawn_point.location.y, spawn_point.location.z = Initial_X, Initial_Y, 0.1
    spawn_point.rotation.roll, spawn_point.rotation.pitch, spawn_point.rotation.yaw = 0.0, 0.0, Initial_Angle
    print('spawn_point', spawn_point)

    return spawn_point 

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================
def game_loop(args):
    global m_steer, m_target_velocity

    client = carla.Client(args.host, args.port)
    print(client)
    client.load_world('Town01')
    client.reload_world()
    world = World(client.get_world(), args)
    
    settings = world.world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_deleta_seconds = 0.05
    world.world.apply_settings(settings)

    clock = pygame.time.Clock()
    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)

    m_target_velocity = 0.0
    while not rospy.is_shutdown(): # main thread
        seg_img = np.copy(world.camera_manager.imageSeg)
        rgb_img = np.copy(world.camera_manager.imageRGB)
        if seg_img is not None and rgb_img is not None:
            pubImage(seg_img, 'mono8', 0)
            pubImage(rgb_img, 'bgr8', 1)

        front_img = world.camera_manager.imageFront
        if front_img is not None:
            pubImage(front_img, 'bgr8', 2)
                
        if isinstance(world.player, carla.Vehicle):
            controller = carla.VehicleControl()

        controller.gear = 1 if m_target_velocity >= 0.0 else -1
        controller.steer = m_steer/540.0
        world.player.apply_control(controller) # apply steering command
        
        curr_v = v = world.player.get_velocity()
        curr_v = round(math.sqrt(curr_v.x**2+curr_v.y**2+curr_v.z**2)*3.6, 1) # [km/h]
        th = world.player.get_transform().rotation.yaw*DEG2RAD
        v.x = m_target_velocity*math.cos(th)
        v.y = m_target_velocity*math.sin(th)
        v.z = 0
        world.player.set_velocity(v)

        localizationData = Float32MultiArray()  # [x, y, heading, velocity]
        x = world.player.get_transform().location.x
        y = world.player.get_transform().location.y
        localizationData.data = [x, y, th, curr_v]
        pub_localizationData.publish(localizationData)

        world.render(display)
        pygame.display.flip()
    # finally:
    #     pygame.quit()

def pubImage(img, encode_str, idx):
    img_msg = Image()
    img_msg.height, img_msg.width = img.shape[0], img.shape[1]
    img_msg.step = img.strides[0]
    img_msg.encoding = encode_str
    img_msg.header.frame_id = 'map'
    img_msg.header.stamp = rospy.Time.now()
    img_msg.data = img.flatten().tolist()

    if idx == 0:
        pub_seg.publish(img_msg)
    elif idx == 1:
        pub_green.publish(img_msg)	
    elif idx == 2:
        pub_front.publish(img_msg)	

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================
def main():
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')
    # netstat -nlp|grep -i carla
    argparser.add_argument('--host', metavar='H', default='0.0.0.0', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-a', '--autopilot', action='store_true', help='enable autopilot')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='200x200', help='window resolution (default: 1280x720)')
    argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.*', help='actor filter (default: "vehicle.*")')
    argparser.add_argument('--rolename', metavar='NAME', default='hero', help='actor role name (default: "hero")')
    argparser.add_argument('--gamma', default=2.2, type=float, help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument('--start', default=1, type=int, help='to set the start pose')
    
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)
    # try:
    game_loop(args)
    # except KeyboardInterrupt:
    #     print('\nCancelled by user. Bye!')

def control_callback(msg):
    global m_steer, m_target_velocity
    m_steer = msg.data[0]*0.59
    m_target_velocity = msg.data[1]*KMpH2MpS*11.0

if __name__ == '__main__':
    rospy.init_node('dyros_phantom_vehicle')
    
    rospy.Subscriber('/learningData', Float32MultiArray, control_callback)
    pub_seg = rospy.Publisher("/front_usb_cam/image_seg", Image, queue_size = 1) # Publishing
    pub_green = rospy.Publisher("/front_usb_cam/image_green", Image, queue_size = 1)    
    pub_front = rospy.Publisher("/front_usb_cam/image_raw", Image, queue_size = 1)
    pub_localizationData = rospy.Publisher('/LocalizationData', Float32MultiArray, queue_size = 1) 

    m_steer, m_target_velocity = 0.0, 0.0
    main()