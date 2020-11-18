# -*- coding: utf-8 -*-
"""
Created on Fri May 31 20:39:13 2019

@author: Evangelos Tzardis
"""
from time import sleep, time
import serial
import mylibrary as mlb
import rw_config as cfg
import numpy as np
import imageio
import os


def init_camera():
    # read config file
    config_with = 'IMAGE SEQUENCE CONFIGURATION'
    config_with_enum = 1
    
    section = cfg.load_config(config_with)
    section = section[0]
    keys = ['height','width']
    settings = [ section[key] for key in keys ]
    
    # Init camera, set configurations
    cam_num = 0
    nodes = mlb.run_camera(cam_num, config_with_enum)
    
    return [nodes, settings]

def encode_int2fixednumbytes(number):
    max_num = 5
    enc = str(number)+'\n'
    
    while len(enc) < max_num:
        enc = '0' + enc
        
    return str.encode(enc)

def run(steps, mode, stackfolder_name):
    cam_nodes, cam_settings = init_camera()
    cam = cam_nodes[0]
    
    #input('Capture sample illuminance: Enter when ready...')
    #Is = mlb.trigger_image_acquisition(cam_nodes)
    #Is = gaussian_filter(Is, sigma=5)
    #
    #input('Capture reference illuminance: Enter when ready...')
    #Ir = mlb.trigger_image_acquisition(cam_nodes)
    #Ir = gaussian_filter(Ir, sigma=5)
    #
    #input('Capture background illuminance: Enter when ready...')
    #Ib = mlb.trigger_image_acquisition(cam_nodes)
    #Ib = gaussian_filter(Ib, sigma=5)
    
    if mode == 'c':
        subfolder = '\\images_calibration'
    elif mode == 'm':
        subfolder = '\\images_measurement'
    
    folder = os.path.dirname(os.path.realpath(__file__)) + subfolder + stackfolder_name
    folder_srb = folder + '\\IsIrIb'
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(folder_srb):
        os.makedirs(folder_srb)
    #imageio.imsave(folder_srb + '\\Is.tiff', Is)
    #imageio.imsave(folder_srb + '\\Ir.tiff', Ir)
    #imageio.imsave(folder_srb + '\\Ib.tiff', Ib)
    
#    input('Enter to continue...')
    
    ser = serial.Serial('COM4', 9600) # Establish the connection on a specific port
    sleep(4)
    
    max_steps = 4096 # minimum step: 1 mV, maximum step: 4096 mV
    increment = round(max_steps/steps)
    
    ser.write(encode_int2fixednumbytes(steps)) # Send number of steps to arduino
    ser.flush()
    ser.write(encode_int2fixednumbytes(increment)) # Send voltage increment to arduino
    ser.flush()
    
    # rounding of increment may lead to extra steps
    extra = (increment*steps - max_steps)//increment
    extra = extra if extra > 0 else 0
    # arbitrarily omit a few of the first images, since PZT is oscillating
    # at the beginning
    omit = 5
    
    rows = mlb.correct_type(cam_settings[0]) # Look at init_camera() --> keys
    cols = mlb.correct_type(cam_settings[1])
    height = steps-extra-omit-1
    
    image_stack = np.zeros([height, rows, cols])
    
    #t1 = np.zeros(steps-extra)
    #t2 = np.zeros(steps-extra)
    #t3 = np.zeros(steps-extra)
    #t4 = np.zeros(steps-extra)
    #t5 = np.zeros(steps-extra)
#    ts = time()
    mlb.begin_acquisition(cam)
    
    for i in range(steps-extra):
    #    ts1 = time()
        rvalue = ser.readline().split(b'\r')[0]
    #    t1[i] = time() - ts1
    #    ts2 = time()
        print(rvalue)
    #    t2[i] = time() - ts2
    #    ts3 = time()
        
        if i+1 > omit:
            image_stack[i-omit-1] = mlb.grab_next_image_by_trigger(cam)
    #    image_stack[i] = mlb.trigger_image_acquisition(cam_nodes)
             
    #    t3[i] = time() - ts3
    #    ts4 = time()
        
        ser.write(b'ok\n')
    #    t4[i] = time() - ts4
    #    ts5 = time()
        ser.flush()
    #    t5[i] = time() - ts5
#    t = time() - ts    
    
    del cam
    mlb.end_acquisition(cam_nodes)
    
    ser.close()
    
    # SAVE IMAGES TO DISK
    for j in range(height):
        fn = '\\' +  str(j) + '.tiff'
        imageio.imsave(folder + fn, image_stack[j])
