# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:41:31 2019

@author: Evangelos Tzardis
"""

from time import sleep
import serial
import sys
#import mylibrary as mlb
#import rw_config as cfg
#import numpy as np
#import shared_data


# This is a Queue that behaves like stdout
class StdoutQueue:
    def __init__(self, stdout_queue):
        self.queue = stdout_queue

    def write(self,msg):
        self.queue.put(msg)

    def flush(self):
        sys.__stdout__.flush()

def encode_int2fixednumbytes(number):
    max_num = 5
    enc = str(number)+'\n'
    
    while len(enc) < max_num:
        enc = '0' + enc
        
    return str.encode(enc)

def run(piezostep_event, steps, stdout_queue):
    
    sys.stdout = StdoutQueue(stdout_queue)
    sys.stderr = StdoutQueue(stdout_queue)
    
    ser = serial.Serial('COM4', 9600) # Establish the connection on a specific port
    sleep(4)
    print('Connection with Arduino established...')
    
    max_steps = 4096 # minimum step: 1 mV, maximum step: 4096 mV
    increment = round(max_steps/steps)
    
    ser.write(encode_int2fixednumbytes(steps)) # Send number of steps to arduino
    ser.flush()
    ser.write(encode_int2fixednumbytes(increment)) # Send voltage increment to arduino
    ser.flush()
    print('Arduino ready...')
    
    # rounding of increment may lead to extra steps
    extra = (increment*steps - max_steps)//increment
    extra = extra if extra > 0 else 0
    # arbitrarily omit a few of the first images, since PZT is oscillating
    # at the beginning
    omit = 5
    
    print('*** PIEZO OSCILLATION BEGINS ***\n')
    
    for i in range(steps-extra):
        rvalue = ser.readline().split(b'\r')[0]
        print(rvalue)
        
        if i+1 > omit:
            # send signal gui-plotting of POI
            piezostep_event.set()
        
        # wait for clearing by the gui-plotting process
        while (piezostep_event.is_set()):
#            sleep(0.01)
            pass
        
        ser.write(b'ok\n')
        ser.flush()
        
#    shared_data.flag_POI = 0
    
    ser.close()
