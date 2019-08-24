# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:06:02 2019

@author: Evangelos Tzardis
"""

import mylibrary as mlb
import rw_config as cfg
import imageio
import numpy as np
import os
from time import sleep, time

def init_camera():
    # read config file
    section= 'image sequence config'
    keys = ['height','width']
    settings = cfg.read_config(section, keys)
    num = 0
    
    # Init camera, set configurations
    nodes = mlb.run_camera(num)
    
    return [nodes, settings]

cam_nodes, cam_settings = init_camera()

subfolder = '\\videoframes\\m'

folder = os.path.dirname(os.path.realpath(__file__)) + subfolder

if not os.path.exists(folder):
    os.makedirs(folder)
    
rows = cam_settings[0] # Look at init_camera() --> keys --> order of elements
cols = cam_settings[1]
height = 500
#
image_stack = np.zeros([height, rows, cols])
    
cam = cam_nodes[0]
mlb.begin_acquisition(cam)

for j in range(height):
    image_stack[j] = mlb.grab_next_image_by_trigger(cam)
    sleep(1/24)

del cam
mlb.release_camera(cam_nodes)

for j in range(height):
    fn = '\\' +  str(j) + '.tiff'
    imageio.imsave(folder + fn, image_stack[j])
    
    
    
    
    
    
    
    
    
    
    
   
    
