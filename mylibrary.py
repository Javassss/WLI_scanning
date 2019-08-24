# -*- coding: utf-8 -*-
"""
@author: Evangelos Tzardis
"""

import PySpin
import scipy.misc as misc
import numpy as np
import rw_config as cfg
from multiprocessing import Queue

#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
#from matplotlib.figure import Figure

def PIL2array(imgptr):
    bpp = imgptr.GetBitsPerPixel()
    if bpp == 8:
        bppnumpy = np.uint8
        
    elif bpp == 16:
        bppnumpy = np.uint16
        
    return np.array(imgptr.GetData(), bppnumpy).reshape(imgptr.GetHeight(), imgptr.GetWidth())

def print_device_info(nodemap):
    """
    This function prints the device information of the camera from the transport
    layer

    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :returns: True if successful, False otherwise.
    :rtype: bool
    """

    print('*** DEVICE INFORMATION ***\n')

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

        else:
            print('Device control information not available.')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result

def reset_exposure(cam):
    """
    This function returns the camera to a normal state by re-enabling automatic exposure.

    :param cam: Camera to reset exposure on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True

        if cam.ExposureAuto.GetAccessMode() != PySpin.RW:
            print('Unable to enable automatic exposure (node retrieval). Non-fatal error...')
            return False

        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)

        print('Automatic exposure enabled...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def reset_trigger(cam):
    """
    This function returns the camera to a normal state by turning off trigger mode.

    :param cam: Camera to acquire images from.
    :type cam: CameraPtr
    :returns: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        # Ensure trigger mode off
        # The trigger must be disabled in order to configure whether the source
        # is software or hardware.
        if cam.TriggerMode.GetAccessMode() != PySpin.RW:
            print('Unable to disable trigger mode (node retrieval). Aborting...')
            return False

        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)

        print('Trigger mode disabled...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def reset_offsets(cam):
    
    if cam.OffsetX.GetAccessMode() == PySpin.RW:
        cam.OffsetX.SetValue(0)
    
    if cam.OffsetY.GetAccessMode() == PySpin.RW:
        cam.OffsetY.SetValue(0)
        
def correct_type(string):
    # convert to original data types.
    # eval returns its string argument with its correct data type,
    # except for the case in which the arguement is an incomprehensible string
    # so a NameError appears
    try:
        t = eval(string)
    except:
        t = string
    
    return t

def configure_custom_image_settings(cam, section):
    """
    This function accesses some nodes of settings via QuickSpin.
    :param cam: Camera to configure settings for.
    :type cam: CameraPtr
    :param section: section of 'config.ini' file to configure camera from
    :type section: SectionProxy
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        
        for key in section:
            
            if key == 'acquisition mode':
                config_acquisitionmode = correct_type( section[key] )
            elif key == 'frame count':
                config_framecount = correct_type( section[key] )
            elif key == 'frame rate enable':
                config_framerateenable = correct_type( section[key] )
            elif key == 'frame rate':
                config_framerate = correct_type( section[key] )
            elif key == 'pixel format':
                config_pixelformat = correct_type( section[key] )
            elif key == 'width':
                config_width = correct_type( section[key] )
            elif key == 'height':
                config_height = correct_type( section[key] )
            elif key == 'offset x':
                config_offsetx = correct_type( section[key] )
            elif key == 'offset y':
                config_offsety = correct_type( section[key] )
            elif key == 'exposure auto':
                config_exposureauto = correct_type( section[key] )
            elif key == 'exposure time':
                config_exposuretime = correct_type( section[key] )
            elif key == 'gain':
                config_gain = correct_type( section[key] )
            elif key == 'gamma enable':
                config_gammaenable = correct_type( section[key] )
            elif key == 'gamma':
                config_gamma = correct_type( section[key] )
            elif key == 'trigger mode':
                config_triggermode = correct_type( section[key] )
            elif key == 'trigger source':
                config_triggersource = correct_type( section[key] )
        

        
        ###### CONFIGURE ACQUISITION MODE ######
        if cam.AcquisitionMode.GetAccessMode() == PySpin.RW:
            
            if config_acquisitionmode == 'single frame':
                cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_SingleFrame)
            elif config_acquisitionmode == 'continuous':
                cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
            elif config_acquisitionmode == 'multiframe':
                cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_MultiFrame)
                
            print('Acquisition mode set to %s...' % cam.AcquisitionMode.GetCurrentEntry().GetSymbolic()) 
                
            if config_acquisitionmode == 'multiframe':
                ###### CONFIGURE ACQUISITION FRAME COUNT ######
                if cam.AcquisitionFrameCount.GetAccessMode() == PySpin.RW:
                    cam.AcquisitionFrameCount.SetValue(config_framecount)
                else:
                    print('Acquisition frame count not available...')
                    result = False
            
            if config_acquisitionmode == 'continuous' or config_acquisitionmode == 'multiframe':
                    
                ###### CONFIGURE ACQUISITION FRAME RATE ######
                if cam.AcquisitionFrameRateEnable.GetAccessMode() == PySpin.RW:
                    cam.AcquisitionFrameRateEnable.SetValue(config_framerateenable)
                
                if cam.AcquisitionFrameRate.GetAccessMode() == PySpin.RW:
                    cam.AcquisitionFrameRate.SetValue(config_framerate)
                else:
                    if cam.AcquisitionFrameRateEnable.GetValue() == 'on':
                        print('Acquisition frame rate not available...')
                        result = False
        
        else:
            print('Acquisition mode not available...')
            result = False
        
        ###### CONFIGURE PIXEL FORMAT ######
        if cam.PixelFormat.GetAccessMode() == PySpin.RW:
            
            if config_pixelformat == 'mono8':
                cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
            elif config_pixelformat == 'mono16':
                cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)
                
            print('Pixel format set to %s...' % cam.PixelFormat.GetCurrentEntry().GetSymbolic())
    
        else:
            print('Pixel format not available...')
            result = False
        
        # reset OffsetX and OffsetY first
        reset_offsets(cam)
        
        ###### CONFIGURE WIDTH ######    
        if cam.Width.GetAccessMode() == PySpin.RW and cam.Width.GetInc() != 0 and cam.Width.GetMax != 0:
            cam.Width.SetValue(config_width)
            print('Width set to %i...' % cam.Width.GetValue())

        else:
            print('Width not available...')
            result = False
        
        ###### CONFIGURE HEIGHT ######
        if cam.Height.GetAccessMode() == PySpin.RW and cam.Height.GetInc() != 0 and cam.Height.GetMax != 0:
            cam.Height.SetValue(config_height)
            print('Height set to %i...' % cam.Height.GetValue())

        else:
            print('Height not available...')
            result = False      
        
        ######## CONFIGURE OFFSET X ####
        if cam.OffsetX.GetAccessMode() == PySpin.RW:
            
            if config_offsetx >= cam.OffsetX.GetMin() & config_offsetx <= cam.OffsetX.GetMax():
                cam.OffsetX.SetValue(config_offsetx)
                print('Offset X set to %d...' % cam.OffsetX.GetValue())
                
            else:
                print('Offset X value not possible...')

        else:
            print('Offset X not available...')
            result = False
        
        ###### CONFIGURE OFFSET Y ######
        if cam.OffsetY.GetAccessMode() == PySpin.RW:
            
            if config_offsety >= cam.OffsetY.GetMin() & config_offsety <= cam.OffsetY.GetMax():
                cam.OffsetY.SetValue(config_offsety)
                print('Offset Y set to %d...' % cam.OffsetY.GetValue())
                
            else:
                print('Offset Y value not possible...')

        else:
            print('Offset Y not available...')
            result = False
        
        ###### CONFIGURE EXPOSURE ######
        if cam.ExposureAuto.GetAccessMode() != PySpin.RW:
            print('Unable to edit automatic exposure. Aborting...')
            return False
        
        if config_exposureauto == 'continuous':
            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)
        elif config_exposureauto == 'off':
            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            
        print('Automatic exposure set to %s...' % cam.ExposureAuto.GetCurrentEntry().GetSymbolic())
    
        # If auto exposure is continuous: set exposure time manually; exposure time recorded in microseconds
        if config_exposureauto == 'off':
            if cam.ExposureTime.GetAccessMode() != PySpin.RW:
                print('Unable to set exposure time. Aborting...')
                return False
    
            # Ensure desired exposure time does not exceed the maximum
            exposure_time_to_set = min(cam.ExposureTime.GetMax(), config_exposuretime)
            cam.ExposureTime.SetValue(exposure_time_to_set)
            print('Exposure time set to %f microseconds...' % exposure_time_to_set)
        
        ###### CONFIGURE GAIN ######
        if cam.GainAuto.GetAccessMode() != PySpin.RW:
            print('Unable to disable automatic gain. Aborting...')
            return False
            
        cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        print('Automatic gain disabled...')
        
        if cam.Gain.GetAccessMode() != PySpin.RW:
            print('Unable to set gain. Aborting...')
            return False
        
        gain_value = config_gain
        cam.Gain.SetValue(gain_value)
        print('Gain value set to %f...' % gain_value)
        
        ###### CONFIGURE GAMMA ######
        
        if cam.GammaEnable.GetAccessMode() != PySpin.RW:
            print('Unable to enable/disable gamma. Aborting...')
            return False
            
        if config_gammaenable == False:
            cam.GammaEnable.SetValue(False)
            print('Gamma disabled...')
            
        else:
            cam.GammaEnable.SetValue(True)
            print('Gamma enabled...')
            cam.Gamma.SetValue(config_gamma)
            print('Gamma value set to %f...' % config_gamma)
            
        ###### CONFIGURE TRIGGER SOURCE ######
            
        # Ensure trigger mode off
        # The trigger must be disabled in order to configure whether the source
        # is software or hardware.
        if cam.TriggerMode.GetAccessMode() != PySpin.RW:
            print('Unable to disable trigger mode (node retrieval). Aborting...')
            return False

        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
        
        # Select trigger source
        # The trigger source must be set to hardware or software while trigger
    	  # mode is off.
        if cam.TriggerSource.GetAccessMode() != PySpin.RW:
            print('Unable to get trigger source (node retrieval). Aborting...')
            return False
        
        if config_triggermode == 'on':
            
            if config_triggersource == 'software':
                cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
                print('Software trigger chosen...')
            elif config_triggersource == 'hardware':
                cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
                print('Hardware trigger chosen...')

            # Turn trigger mode on
            # Once the appropriate trigger source has been set, turn trigger mode
            # on in order to retrieve images using the trigger.
            cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
            print('Trigger mode turned on...')
        
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False
    
    return result  

#def live_view():
#    
    

def acquire_images(cam, nodemap, nodemap_tldevice, q, stoplive_event):
    """
    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    print('*** IMAGE ACQUISITION ***\n')
    try:
        # Begin acquiring images
        cam.BeginAcquisition()

        print('Acquiring image...')

#        disting_name = ''
#        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
#        if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
#            disting_name = node_device_serial_number.GetValue()

        if cam.AcquisitionMode.GetValue() == PySpin.AcquisitionMode_Continuous:
        
            # while loop until flag gets 1 from GUI func: 'stop_live'
            while(stoplive_event.is_set() == False):

                try:
                    image_result = cam.GetNextImage()
                    q.put(image_result.GetData(), block=True, timeout=0.2)
                    image_result.Release()
                    
                except PySpin.SpinnakerException as ex1:
                    print('Error: %s' % ex1)
                    return False
                
                except Queue.Full as ex2:
                    print('Error: %s' % ex2)
                    return False
        
        # send signal for finishing
        q.put('done')
        
        print('Image acquisition finished...')
        cam.EndAcquisition()
    
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

def run_camera(cam_num, config_with):
    """
    cam_num: int | [0,1,...]
    config_with: int | [0,1,2]
    """
    
    global config_sections
    
    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %d' % num_cameras)
    
    # Finish if there are no cameras
    if num_cameras == 0:

        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
    
    cam = cam_list[cam_num]
    
    # Retrieve TL device nodemap and print device information
    nodemap_tldevice = cam.GetTLDeviceNodeMap()

    print_device_info(nodemap_tldevice)

    # Initialize camera
    cam.Init()

    # Retrieve GenICam nodemap
    nodemap = cam.GetNodeMap()
    
    # load all configurations
    config_sections = cfg.load_config('ALL')
    # select one configuration section out of all
    configcam_with = config_sections[config_with]
    if not configure_custom_image_settings(cam, configcam_with):
        return False
    
    print('*** CAMERA CONNECTION ESTABLISHED ***\n')
    cam_nodes = [cam, nodemap, nodemap_tldevice, cam_list, system]
    
    return cam_nodes

def begin_acquisition(cam):
    
    try:
        cam.BeginAcquisition()
        return True
        
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

def end_acquisition(cam_nodes):
    cam = cam_nodes[0]
    
    # End camera acquisition
    cam.EndAcquisition()
    
    del cam
    release_camera(cam_nodes)
    
def release_camera(cam_nodes):
    cam = cam_nodes[0]
    cam_list = cam_nodes[3]
    system = cam_nodes[4]

    # Reset exposure
    reset_exposure(cam)
    
    # Reset trigger
    reset_trigger(cam)
    
    # Deinitialize camera
    cam.DeInit()
    
    # Release reference to camera
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()
    del cam_nodes[:]

    # Release system instance
    system.ReleaseInstance()
    
    print('Camera release successful')
     
def grab_next_image_by_trigger(cam):
    """
    This function acquires an image by executing the trigger node.

    :param cam: Camera to acquire images from.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    global trigger_flag
    
    try:
        result = True

        if cam.TriggerSource.GetValue() == PySpin.TriggerSource_Software:

            # Execute software trigger
            if cam.TriggerSoftware.GetAccessMode() != PySpin.WO:
                print('Unable to execute trigger. Aborting...')
                return False
            
            try:
                cam.TriggerSoftware.Execute()
                trigger_flag = 1
                image_result = cam.GetNextImage()
                image_array = PIL2array(image_result)
                image_result.Release()
                
                result = image_array
            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                result = False

            # TODO: Blackfly and Flea3 GEV cameras need 2 second delay after software trigger

        elif cam.TriggerSource.GetValue() == PySpin.TriggerSource_Line0:
            print('Use the hardware to trigger image acquisition.')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result

def save_profile(filename, profile):
    misc.imsave(filename, profile)
    
def save_image(disting_name):
    
    filename = '%s.tif' % disting_name

    # Save in image format
    misc.imsave(filename, image_array)
    # Save image array in txt
    #np.savetxt(('%s.txt' % disting_name), image_array, fmt = '%d')
    print('Image saved as %s' % filename)
    
#def stoplive():
#    global stoplive_flag
#    
#    stoplive_flag = 1
            
##########################################
##                                       #
##   FUNCTIONS USED SPECIFICALLY FOR GUI #
##                                       #
##########################################
#    
#def acquire_images_gui(cam, nodemap, nodemap_tldevice, gui_param):
#    """
#    :param cam: Camera to acquire images from.
#    :param nodemap: Device nodemap.
#    :param nodemap_tldevice: Transport layer device nodemap.
#    :type cam: CameraPtr
#    :type nodemap: INodeMap
#    :type nodemap_tldevice: INodeMap
#    :return: True if successful, False otherwise.
#    :rtype: bool
#    """
#    
#    global image_array
#    global disting_name
#    global stoplive_flag
#
#    print('*** IMAGE ACQUISITION ***\n')
#    try:
#        result = True
#        
#        # Retrieve gui parameters needed for image displaying
#        figplot = gui_param[0]
#        fig = gui_param[1]
#        canvas = gui_param[2]
#        single_feed = gui_param[3]
#        livefeed_img = gui_param[4]
#        livefeed_midline = gui_param[5]
#        
#        # Begin acquiring images
#        cam.BeginAcquisition()
#        
#        print('Acquiring image...')
#
#        disting_name = ''
#        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
#        if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
#            disting_name = node_device_serial_number.GetValue()
#            #print('Device serial number retrieved as %s...' % disting_name)
#
##        if cam.AcquisitionMode.GetValue() == PySpin.AcquisitionMode_SingleFrame:
##            NUM_IMAGES = 1
##        elif cam.AcquisitionMode.GetValue() == PySpin.AcquisitionMode_Continuous:
##            NUM_IMAGES = 1000
#            
#        try:
#            image_result = cam.GetNextImage()
#            image_array = PIL2array(image_result)
#            image_result.Release()
#            
#            ### SINGLE CAPTURE - IMAGE DISPLAY ###
#            if single_feed:
#                fig.imshow(image_array, aspect='auto', cmap='gray')
#                canvas.draw()
#                
#            ### LIVE IMAGE DISPLAY ON GUI ###
#            if livefeed_img:
#                fimshow = fig.imshow(image_array, aspect='auto', cmap='gray')
#                canvas.draw()
#                
#        except PySpin.SpinnakerException as ex:
#                print('Error: %s' % ex)
#                return False
#        
#        if cam.AcquisitionMode.GetValue() == PySpin.AcquisitionMode_Continuous:
#        
#            # Reset flag
#            stoplive_flag = 0
#            ctr = 0
#            
#            # while loop until flag gets 1 from GUI func: 'stop_live'
#            while(stoplive_flag == 0):
#                ctr += 1
#                
#                try:
#                    image_result = cam.GetNextImage()
#                    image_array = PIL2array(image_result)
#                    image_result.Release()
#                    
#                    ### LIVE IMAGE DISPLAY ON GUI ###
#                    if livefeed_img & (ctr % 20 == 0):
#                        fimshow.set_data(image_array)
#                        canvas.draw()
#                        
#                    ### MIDDLE LINE DISPLAY ON GUI ###
#                    if livefeed_midline & (ctr % 20 == 0):
#                        fig.clear()
#                        midline = (np.shape(image_array))[0]//2
#                        fig.plot(image_array[midline])
#                        canvas.draw()
#                        fig.clear()
#                        
#                    if ctr == 100:
#                        ctr = 0
#                    
#                except PySpin.SpinnakerException as ex:
#                    print('Error: %s' % ex)
#                    return False
#            
#            # stoplive_flag value became '1', so reset it
#            stoplive_flag = 0
#            
#        cam.EndAcquisition()
#        print('Image(s) acquisition completed...')
#    
#    except PySpin.SpinnakerException as ex:
#        print('Error: %s' % ex)
#        return False
#
#    return result
#
#def run_single_camera_gui(cam, gui_param):
#    """
#    :param cam: Camera to run on.
#    :type cam: CameraPtr
#    :return: True if successful, False otherwise.
#    :rtype: bool
#    """
#    try:
#        result = True
#
#        # Retrieve TL device nodemap and print device information
#        nodemap_tldevice = cam.GetTLDeviceNodeMap()
#
#        result &= print_device_info(nodemap_tldevice)
#
#        # Initialize camera
#        cam.Init()
#
#        # Retrieve GenICam nodemap
#        nodemap = cam.GetNodeMap()
#        
#        if not configure_custom_image_settings(cam):
#            return False
#        
#        # Acquire images
#        result &= acquire_images_gui(cam, nodemap, nodemap_tldevice, gui_param)
#        
#        # Reset exposure
#        result &= reset_exposure(cam)
#        
#        # Reset trigger
#        result &= reset_trigger(cam)
#
#        # Deinitialize camera
#        cam.DeInit()
#
#    except PySpin.SpinnakerException as ex:
#        print('Error: %s' % ex)
#        result = False
#
#    return result
#
#def run_camera_gui(cam_num, gui_param):
#    
#    # Retrieve singleton reference to system object
#    system = PySpin.System.GetInstance()
#
#    # Get current library version
#    version = system.GetLibraryVersion()
#    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))
#
#    # Retrieve list of cameras from the system
#    cam_list = system.GetCameras()
#
#    num_cameras = cam_list.GetSize()
#
#    print('Number of cameras detected: %d' % num_cameras)
#    
#    # Finish if there are no cameras
#    if num_cameras == 0:
#
#        # Clear camera list before releasing system
#        cam_list.Clear()
#
#        # Release system instance
#        system.ReleaseInstance()
#
#        print('Not enough cameras!')
#    
#    cam = cam_list[cam_num]
#    run_single_camera_gui(cam, gui_param)
#    #mpl.pyplot.pcolormesh(image_array,cmap='gray')
#    # Release reference to camera
#    del cam
#
#    # Clear camera list before releasing system
#    cam_list.Clear()
#
#    # Release system instance
#    system.ReleaseInstance()
#    
#def updateFrame(fig_img, image_array):
#    fig_img.set_data(image_array)
#    #canvas.draw()
#    
#def save_image_gui(savedir, img_case):
#    """
#    Wrapper function called from gui
#    """
#    # img_case is the name that corresponds to one of the four different images
#    # acquired from the interferometer
#    if img_case:
#        save_image(savedir + '\\' + img_case)
#    # disting_name is a global variable assigned at image acquisition process
#    else:
#        save_image(savedir + '\\' + disting_name)
#        
#def stoplive():
#    global stoplive_flag
#    
#    stoplive_flag = 1


    
    
   
                
    
    
    
