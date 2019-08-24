# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:42:49 2019

@author: Evangelos Tzardis
"""

import mylibrary as mlb
import rw_config as cfg
import imageio
import os
import matplotlib.pyplot as plt
import numpy as np

def init_camera():
    settings = ['acquisition mode','single frame',\
                'pixel format','mono16',\
                'exposure auto','off',\
                'exposure time', '552',\
                'width','556',\
                'height','350',\
                'offset x','496',\
                'offset y','390']
    cfg.write_config(settings)
    cam_num = 0
    
    # Init camera, set configurations
    cam_nodes = mlb.run_camera(cam_num)
    
    return cam_nodes

cam_nodes = init_camera()

subfolder = '\\image_calibrationXY'

folder = os.path.dirname(os.path.realpath(__file__)) + subfolder

if not os.path.exists(folder):
    os.makedirs(folder)
    
image = mlb.trigger_image_acquisition(cam_nodes)
    
mlb.release_camera(cam_nodes)
imageio.imsave(folder + '\\knownsample.tiff', image)

plt.imshow(image)
x = plt.ginput(2)

pixels = x[1][1] - x[0][1]

"""
known distance
"""
d = 400 # um
pixel = d/pixels # imaged distance on a pixel

np.savetxt('pixel_calibrated.txt', (pixel,), fmt='%.3f')






