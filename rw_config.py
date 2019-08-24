# -*- coding: utf-8 -*-
"""
@author: Evangelos Tzardis
"""
import configparser
import os

def write_config(cam_settings):
    
    # default configuration values
    config = configparser.ConfigParser()
    config['DEFAULT'] = {}
    defconfig = config['DEFAULT']
    defconfig['acquisition mode'] = 'single frame'
    defconfig['pixel format'] = 'mono16'
    defconfig['width'] = '1280'
    defconfig['height'] = '1024'
    defconfig['offset x'] = '0'
    defconfig['offset y'] = '0'
    defconfig['exposure auto'] = 'continuous'
    defconfig['gain'] = '0.0'
    defconfig['gamma enable'] = 'True'
    defconfig['gamma'] = '1'
    defconfig['frame rate enable'] = 'False'
    defconfig['trigger mode'] = 'off'
    
    if cam_settings[0] == 'default':
        config['CAMERA CONFIGURATION'] = defconfig
    else:
        config.read('config.ini')
        cameraconfig = config['CAMERA CONFIGURATION']
    
#    # copy default values before any adjustment.
#    # new values will be overwritten
#    config['CAMERA CONFIGURATION'] = defconfig
#    cameraconfig = config['CAMERA CONFIGURATION']
    
    for i in range(0,len(cam_settings),2):
            
        if cam_settings[i] == 'acquisition mode':
            cameraconfig['acquisition mode'] = cam_settings[i+1]
            
        elif cam_settings[i] == 'frame count':
            cameraconfig['frame count'] = cam_settings[i+1]
            
        elif cam_settings[i] == 'frame rate enable':
            cameraconfig['frame rate enable'] = cam_settings[i+1]
            
        elif cam_settings[i] == 'frame rate':
            cameraconfig['frame rate'] = cam_settings[i+1]
        
        elif cam_settings[i] == 'pixel format':
            cameraconfig['pixel format'] = cam_settings[i+1]
                
        elif cam_settings[i] == 'width':
            cameraconfig['width'] = cam_settings[i+1]
            
        elif cam_settings[i] == 'height':
            cameraconfig['height'] = cam_settings[i+1]
            
        elif cam_settings[i] == 'offset x':
            cameraconfig['offset x'] = cam_settings[i+1]
            
        elif cam_settings[i] == 'offset y':
            cameraconfig['offset y'] = cam_settings[i+1]
        
        elif cam_settings[i] == 'exposure auto':
            cameraconfig['exposure auto'] = cam_settings[i+1]
        
        elif cam_settings[i] == 'exposure time':
            cameraconfig['exposure time'] = cam_settings[i+1]
            
        elif cam_settings[i] == 'gain':
            cameraconfig['gain'] = cam_settings[i+1]
            
        elif cam_settings[i] == 'gamma enable':
            cameraconfig['gamma enable'] = cam_settings[i+1]
            
        elif cam_settings[i] == 'gamma':
            cameraconfig['gamma'] = cam_settings[i+1]
        
        elif cam_settings[i] == 'trigger mode':
            cameraconfig['trigger mode'] = cam_settings[i+1]
        
        elif cam_settings[i] == 'trigger source':
            cameraconfig['trigger source'] = cam_settings[i+1]
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open('%s' %dir_path + '\config.ini','w') as configfile:
        config.write(configfile)
        
def read_config(section, ret_key):
    """
    section: string
    ret_key: list of strings
    return configuration settings of the camera
    rtype: list
    """
    
    config = configparser.ConfigParser()
    config.sections()
    config.read('config.ini')
    config.sections()
    
    if section == 'image sequence config':
        section_config = config['IMAGE SEQUENCE CONFIGURATION']
    elif section == 'live view config':
        section_config = config['LIVE VIEW CONFIGURATION']
    
    settings = []
    
    if ret_key[0] == 'all':
        for key in section_config:
            settings.append(key)
            # convert to original data types.
            # eval returns its string argument with its correct data type,
            # except for the case in which the arguement is an incomprehensible string
            # so a NameError appears
            try:
                t = eval(section_config[key])
            except:
                t = section_config[key]
            settings.append(t)
    else:
        for i in range(len(ret_key)):
            try:
                t = eval(section_config[ret_key[i]])
            except:
                t = section_config[ret_key[i]]
            settings.append(t)
        
    return settings

def load_config(sections):
    global config
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    section_default = config['DEFAULT']
    section_imseq = config['IMAGE SEQUENCE CONFIGURATION']
    section_live = config['LIVE VIEW CONFIGURATION']
    
    if sections == 'ALL':
        ret = [section_default, section_imseq, section_live]
    else:
        ret = [config[sections]]
    
    return ret

def load_config_fromtkText(sections, tbox_contents):
    config = configparser.ConfigParser()
    config.read_string(tbox_contents)
    
    section_default = config['DEFAULT']
    section_imseq = config['IMAGE SEQUENCE CONFIGURATION']
    section_live = config['LIVE VIEW CONFIGURATION']
    
    if sections == 'ALL':
        ret = [section_default, section_imseq, section_live]
    else:
        ret = [config[sections]]
    

    return ret

# write every section of the config object to a tkinter text box
def save_config_totkText(config_list):
    string = ''
    secnames = [config_list[i]._name for i in range(len(config_list))]
    
    ctr = 0
    for sec in secnames:
        string += '['+sec+']\n'
        for item in config_list[ctr]:
            string += ('%s = %s\n'%(item,config_list[ctr][item]))
        string += '\n'
        ctr += 1
            
    return string
    
#s = save_config_totkText(sec)
#sec = load_config_fromtkText('ALL', s)
#sec = load_config('ALL')
    
    
    
   
    
    
