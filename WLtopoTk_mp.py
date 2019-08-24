# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:29:47 2019

@author: JarvanXIV
"""

try:
    import Tkinter as tk # Python 2
    from Tkinter.ScrolledText import ScrolledText
    from Tkinter import TkFileDialog
    import ttk
except ImportError:
    import tkinter as tk # Python 3
    from tkinter.scrolledtext import ScrolledText
    from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import matplotlib as mpl
import numpy as np
import os, sys
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import oscillate_piezo
from time import sleep
from multiprocessing import Process, Queue, Event, Value, Array
from threading import Thread
import ctypes
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

import mylibrary as mlb
import rw_config as cfg
import sync_PZTcam_methods as piezoscan
import calibratePZT_methods as calibrate
import get_profile_methods as analysis


"""
Parent/consumer process
"""
def create_cameraprocess(q, release_event, beginlive_event, stoplive_event,\
                         piezostep_event, updateconfig_event, plotmidline_event, stdout_queue):
    global a
    
#    a.set_ylim([0,1e4])
    button_initcamera.config(state='disabled')
    
    # Operate camera functions with a child process
    # Initialize objects that will be passed to the child process
    width = Value('i',  0)
    height = Value('i', 0)
    p2 = Process(target=camera_worker, args=(q, release_event, beginlive_event, stoplive_event,\
                                             updateconfig_event, width, height, tbox, stdout_queue))
    p2.start()
    sleep(2)
    
    # second thread for POI plotting
#    thread.start_new_thread( live_POIplot, () )
    
    # Parent process: Display images currently being retrieved by the child process
#    first_step = False
    while(True):
        
        if release_event.is_set():
            break
        else:
            if not beginlive_event.is_set():
                sleep(1)
                root.update()
            else:
                # Wait until image acquisition has begun from child process
                while(q.qsize() == 0):
                    try:
                        root.update()
                    except:
                        pass
                    sleep(0.5)
            
                # L I V E    F E E D
                # P L O T    I N T E R F E R O G R A M    P O I
                # P L O T    M I D D L E    L I N E    I N T E N S I T I E S
                while (width.value == 0 or height.value == 0):
                    sleep(0.01)
                imax = None
                image_bytes = q.get()
                while( not isinstance(image_bytes, str) ):
                    #----------------------------------------------------------
                    # live image display
                    img = Image.frombytes('L', (width.value, height.value), image_bytes)
#                    photo = ImageTk.PhotoImage(image=img)
                    if imax is None:
                        # param: vmin=0,vmax=65535
                        imax = ax.imshow(img,cmap='gray',vmin=0,vmax=65535)
                    else:
                        imax.set_data(img)
                        livefeed_canvas.draw()
                    if toggle_selector.RS.active:
                        toggle_selector.RS.update()
#                    livefeed_canvas.create_image(0,0,image=photo,anchor='nw')
                    
                    root.update()
                    #----------------------------------------------------------
                    # live POI plotting
                    if piezostep_event.is_set():
                        for i in range(len(POI)):
                            y_list[i].pop()
                            y_list[i].appendleft(img.getpixel(POI[i]))
                            line_list[i].set_data(x_list,y_list[i])
                            line_list[i].set_color(colors[i])
                        
                        canvas_plot.draw()
                        root.update()
                        piezostep_event.clear()
                        
                    #----------------------------------------------------------
                    # live midline plotting
                    if plotmidline_event.is_set():
                        a.clear()
                        midline = height.value//2
                        intensities = [ img.getpixel((col, midline)) for col in range(width.value)]
                        a.plot(intensities)
                        canvas_plot.draw()
                        a.clear()
                    #----------------------------------------------------------
                    # get next image for next loop
                    image_bytes = q.get()
        
                # after live feed stops
                stoplive_event.clear()
#                first_step = False
            
    p2.join()

"""
Starting function of child/worker process
"""

def camera_worker(q, release_event, beginlive_event, stoplive_event,\
                  updateconfig_event, width, height, tbox, stdout_queue):
    
    sys.stdout = StdoutQueue(stdout_queue)
    sys.stderr = StdoutQueue(stdout_queue)
    
    release_event.clear()
    section_num = 0
    init_camera(section_num, tbox)
    #--------------------------------------------------------------------------
    
    while( True ):
        # share width and height of image with the parent process
        width.value = eval(sections[2]['width'])
        height.value = eval(sections[2]['height'])
        
        if release_event.is_set():
            break
        
        if not beginlive_event.is_set():
            if updateconfig_event.is_set():
                load_config(tbox)
                updateconfig_event.clear()
            sleep(1)
        else:
            # Retrieve images from the camera continuously
            # until the stoplive_event is set
            live_feed(q, stoplive_event)
            
            beginlive_event.clear()
            stoplive_event.clear()
        
    #--------------------------------------------------------------------------
    release_event.clear()
    mlb.release_camera(cam_nodes)

#def camera_worker(q, release_event, beginlive_event, stoplive_event, width, height):
#    release_event.clear()
#    section_num = 0
#    init_camera(section_num)
#    #--------------------------------------------------------------------------
#    
#    # share width and height of image with the parent process
#    width.value = eval(sections[2]['width'])
#    height.value = eval(sections[2]['height'])
#    
#    # Wait until beginlive_event is set
#    while( not beginlive_event.is_set() ):
#        sleep(0.5)
#    
#    # Retrieve images from the camera continuously
#    # until the stoplive_event is set
#    live_feed(q, stoplive_event)
#    
#    beginlive_event.clear()
#    stoplive_event.clear()
#    
#    # Wait until release_event is set
#    while( not release_event.is_set() and not beginlive_event.is_set()):
#        sleep(1)
#    
#    release_event.clear()
#    mlb.release_camera(cam_nodes)

def init_camera(section_num, tbox):
    global cam_nodes
    
    load_config(tbox)
    cam_num = 0
    
    # Init camera, set configurations
    cam_nodes = mlb.run_camera(cam_num, section_num)
    
#def set_exposure():
#    global exposure_time
#    global sections
#    
#    exposure_time = str(slider_exposure.get())
#    # edit configurations of SectionProxy global var: sections
#    # no writing to disk
#    # edit all sections besides ('default':sections[0])
#    for i in [1,2]:
#        sections[i]['exposure auto'] = 'off'
#        sections[i]['exposure time'] = exposure_time
#    # cam: the initialized camera
#    cam = cam_nodes[0]
#    # now configure the camera with the settings of one of the non-default sections
#    mlb.configure_custom_image_settings(cam, sections[1])
#    
#def get_exposure(entry_exposure):
#    exposure_time = str(entry_exposure.widget.get())
#    slider_exposure.set(exposure_time)
    
def notify_releasecamera(release_event):
    button_release.config(state='disabled')
    release_event.set()
    button_initcamera.config(state='normal')
    
def notify_beginlive(beginlive_event):
    button_release.config(state='disabled')
    button_saveconfig.config(state='disabled')
    beginlive_event.set()
    
def notify_stoplive(stoplive_event):
    button_release.config(state='normal')
    button_saveconfig.config(state='normal')
    stoplive_event.set()
    
def toggle_displayMidline(plotmidline_event):
    global displaymidline_state
    
    displaymidline_state = not displaymidline_state
    if displaymidline_state == True:
        plotmidline_event.set()
    else:
        plotmidline_event.clear()
    
def live_feed(q, stoplive_event):
    # cam: the initialized camera
    # configure the camera with the 'live view' section: sections[2]
    cam = cam_nodes[0]
    mlb.configure_custom_image_settings(cam, sections[2])
    mlb.acquire_images(*cam_nodes[:3], q, stoplive_event)
    
def on_click(event):
    global x, y
    
    x = int(round(event.xdata))
    y = int(round(event.ydata))

def enable_POIsel():
    global oc_id, sp_id
    
    del POI[:]
    
    oc_id = livefeed_canvas.mpl_connect('button_press_event', on_click)
    sp_id = livefeed_canvas.get_tk_widget().bind('<Return>', save_POI)
    livefeed_canvas.get_tk_widget().focus_set()
    
def disable_POIsel():
    livefeed_canvas.mpl_disconnect(oc_id)
    livefeed_canvas.get_tk_widget().unbind('<Return>', sp_id)
    root.focus_set()
    
def save_POI(event):
    if len(POI) >= 0:
        POI.append((x, y))  # it must be a sequence ( , )
        print(POI)

def toggle_selector(event):
#    print(event.key)
    if event.key=='enter' and toggle_selector_RS.active:
#        print(' RectangleSelector deactivated.')
        print(ROI)
        toggle_selector_RS.set_visible(False)
        toggle_selector_RS.update()
        toggle_selector_RS.set_active(False)
        livefeed_canvas.mpl_disconnect(ts_id)
        root.focus_set()

def line_select_callback(eclick, erelease):
    global ROI
    
    'eclick and erelease are the press and release events'
    if toggle_selector_RS.active:
        x1, y1 = int(round(eclick.xdata)), int(round(eclick.ydata))
        x2, y2 = int(round(erelease.xdata)), int(round(erelease.ydata))
        ROI = [(x1,y1), (x2,y2)]

def enable_ROIsel():
    global ts_id
    
    del ROI[:]
    
    toggle_selector_RS.set_active(True)
    toggle_selector_RS.set_visible(True)
    ts_id = livefeed_canvas.mpl_connect('key_press_event', toggle_selector)
    livefeed_canvas.get_tk_widget().focus_set()


###############################################################################
#def enable_ROIsel():
#    global btn_funcid2
#    global rtr_funcid2
#    
#    del ROI[:]
#    
#    canvas_widget = livefeed_canvas.get_tk_widget()
#    btn_funcid2 = canvas_widget.bind('<Button 1>', get_pixels)
#    rtr_funcid2 = canvas_widget.bind('<Return>', save_ROI)
#    canvas_widget.focus_set()
#    
#def disable_ROIsel():
#    canvas_widget = livefeed_canvas.get_tk_widget()
#    canvas_widget.unbind('<Button 1>', btn_funcid2)
#    canvas_widget.unbind('<Return>', rtr_funcid2)
#    root.focus_set()
#    
#    # Also save the new ROI in the configuration text box (2nd tab)
#    
#    # Take string from text box
#    tbox_contents = text_config.get('1.0','end')
#    # Convert it to SectionProxy object
#    section_list = cfg.load_config_fromtkText('ALL', tbox_contents)
#    # Change dimensions and offsets
#    for i in range(1,3):
#        section_list[i]['height'] = str(ROI[1][1] - ROI[0][1])
#        section_list[i]['width'] = str(ROI[1][0] - ROI[0][0])
#        section_list[i]['offset x'] =str(ROI[0][0])
#        section_list[i]['offset y'] = str(ROI[0][1])
#    # Convert back to string
#    config_text = cfg.save_config_totkText(section_list)
#    # Insert the configuration list into the configuration text box
#    text_config.delete('1.0','end')
#    text_config.insert('end',config_text)
#    
#    # signal the camera worker to update its global variable 'sections'
#    update_config()
###############################################################################
    
#def save_ROI(event):
#    
#    if len(ROI) >= 0:
#        ROI.append([x,y])

#def toggleDispInterf_threaded():
#    global dispInterf_state
#    
#    dispInterf_state = 1 if dispInterf_state == 0 else 0
#    
#    if dispInterf_state == 0:
#        stop_event.set()
##        a.clear()
#        canvas.get_tk_widget().delete('all') 
#        del x[:], y[:]
#    else:
#        stop_event.clear()
#        Process(target=display_interf, args=(q,)).start()
        
def create_oscillationprocess(piezostep_event, stdout_queue):
    global line_list, x_list, y_list
    
    steps = piezosteps()
    
    # initialize the plot figure axes
    line_list = []
    a.clear()
    a.set_title('Light intensity vs. reference mirror displacement')
#    a.set_xlabel('Displacement in $\mu m$')
    a.set_xlabel('Piezo steps')
    a.set_ylabel('Digitized Intensity')
#    a.axis([0, np.amax(piezo_dispaxis), 0, 1e3])
    a.axis([0, steps, 0, 1e3])
    x_list = deque(np.arange(steps,0,-1))
#    x_list = deque(piezo_dispaxis[::-1])
    y_list = [ deque([-1]*steps) for i in range(len(POI)) ]
    
    # clear the plot figure each time the button 'oscillate piezo' is pressed
    for i in range(len(POI)):
        # a (axes) created with 'initialize' button
        line, = a.plot([],[])
        line_list.append(line)
        line_list[i].set_marker('.')
    
    # begin new process for piezo moving
    piezostep_event.clear()
    input_steps = eval(piezosteps_var.get())
    p3 = Process(target=oscillate_piezo.run, args=(piezostep_event, input_steps, stdout_queue))
    p3.start()
    
def update_config():
    tbox.value = bytes(text_config.get('1.0','end-1c'), 'utf8')
    updateconfig_event.set()

def load_config(tbox):
    global sections
    
    # wrapper
    # sections: str
#    sections = cfg.load_config('ALL')
    sections = cfg.load_config_fromtkText('ALL', tbox.value.decode('utf8'))
#    shared_data_mp.set_sections(sections)
    
    return sections

# color of plot lines
def lspec():
    
    colors = ['b', 'r', 'k', 'g', 'c', 'm', 'y', 'w']
    return colors

# for text highlighting
def search(text_widget, keyword, tag):
    pos = '1.0'
    while True:
        idx = text_widget.search(keyword, pos, 'end')
        if not idx:
            break
        pos = '{}+{}c'.format(idx, len(keyword))
        text_widget.tag_add(tag, idx, pos)

def update_tboxEmbellishments():
    search(text_config, '[DEFAULT]', 'section')
    search(text_config, '[IMAGE SEQUENCE CONFIGURATION]', 'section')
    search(text_config, '[LIVE VIEW CONFIGURATION]', 'section')
    root.after(2000, update_tboxEmbellishments)

def piezosteps():
    # Same is calculated in 'oscillate_piezo.py'
    max_steps = 4096 # minimum step: 1 mV, maximum step: 4096 mV
    steps = eval(piezosteps_var.get())
    increment = round(max_steps/steps)
    # rounding of increment may lead to extra steps
    extra = (increment*steps - max_steps)//increment
    extra = extra if extra > 0 else 0
    omit = 5
    plotted_steps = steps - extra - omit - 1
    
    return plotted_steps

def setMeasurementFolderName(event):
    global measurementfolder_name
    
    measurementfolder_name = entry_measfolder.widget.get()

def setCalibrationFolderName(event):
    global calibrationfolder_name
    
    calibrationfolder_name = entry_calibfolder.widget.get()

def prepare_piezoscan():
    steps = eval(piezosteps_var.get())
    mode = CheckVar.get()
    if mode == 'm':
        folder_name = measurementfolder_name
    else:
        folder_name = calibrationfolder_name
    piezoscan.run(steps, mode, folder_name)

def prepare_calibrate():
    steps = eval(piezosteps_var.get())
    calibrate.run(steps)

def prepare_analysis():
    steps = eval(piezosteps_var.get())
    analysis.run(calib_linear_region, steps)

def clear_outputtext():
    output_text.delete('1.0','end')

#class StdoutRedirector:
#    '''A class for redirecting stdout to the Text widget.'''
#    def __init__(self,text):
#        self.text = text
#        self.text.tag_config('stdout',foreground='thistle1')
#    
#    def write(self,str_):
#        self.text.insert('end',str_,'stdout')
#        
#class StderrRedirector:
#    '''A class for redirecting stderr to the Text widget.'''
#    def __init__(self,text):
#        self.text = text
#        self.text.tag_config('stderr',foreground='brown3')
#        
#    def write(self,str_):
#        self.text.insert('end',str_,'stderr')

# This function takes the text widget and a queue as inputs.
# It functions by waiting on new data entering the queue, when it 
# finds new data it will insert it into the text widget 
def text_catcher(text_widget, queue):
    while True:
        text_widget.insert('end', queue.get())

# This is a Queue that behaves like stdout
class StdoutQueue:
    def __init__(self, stdout_queue):
        self.queue = stdout_queue

    def write(self,msg):
        self.queue.put(msg)

    def flush(self):
        sys.__stdout__.flush()

def on_close():
     # To stop redirecting stdout:
    sys.stdout = sys.__stdout__
    # To stop redirecting stderr:
    sys.stderr = sys.__stderr__
    # Write configuration list from text box to 'config.ini'
    with open('config.ini','w') as configfile:
        configfile.write(text_config.get('1.0', 'end-1c'))
        configfile.close()
    # Destroy main window of GUI
    root.destroy()

"""
M A I N    G U I    T H R E A D
"""
def gui():

    """""""""""""""""""""
     G  L  O  B  A  L  S
    """""""""""""""""""""
#    global dispInterf_state
#    global stop_event
    
    global root, livefeed_canvas, imageplots_frame
    
    global slider_exposure, measurementfolder_name, calibrationfolder_name
    global button_initcamera, button_release, button_saveconfig
    global text_config, tbox, entry_measfolder, entry_calibfolder, piezosteps_var, CheckVar, output_text
    global a, colors, canvas_plot, line, ax
    
    global updateconfig_event
    
    global POI, ROI
    global displaymidline_state
    global piezo_dispaxis, calib_linear_region
    global toggle_selector_RS
    
    POI, ROI = [], []
    displaymidline_state = False
    measurementfolder_name = 'stack'
    calibrationfolder_name = 'stack'
    calib_linear_region = [19,None]
    
#    dispInterf_state = 0
    
    q = Queue()
    stdout_queue = Queue()
    
    beginlive_event = Event()
    stoplive_event = Event()
    release_event = Event()
#    stop_event = Event()
    piezostep_event = Event()
    updateconfig_event = Event()
    plotmidline_event = Event()
    
    """
    MAIN WINDOW
    """
    root = tk.Tk()
    root.iconbitmap('winicon.ico')
    #root.wm_attributes('-topmost', 1)
#    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    #root.geometry("%dx%d+0+0" % (w, h))
    root.title('White light interferometry: Topography')
    root.configure(background='grey')
    
    """
    MENU
    """
    menubar = tk.Menu(root)
    filemenu = tk.Menu(menubar, tearoff=0)
    #filemenu.add_command(label = 'Load image', command=openimage)
    filemenu.add_command(label = 'Save displayed image')
    menubar.add_cascade(label = 'File', menu=filemenu)
    
    optionsmenu = tk.Menu(menubar, tearoff=0)
    optionsmenu.add_command(label = 'Configure camera')
    menubar.add_cascade(label = 'Options', menu=optionsmenu)
    
    menubar.add_command(label = 'Help')
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            L        A        Y        O        U        T
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    """
    2 MAIN TABS
    """
    tabControl = ttk.Notebook(root)
    tab_ops = ttk.Frame(tabControl)
    tab_config = ttk.Frame(tabControl)
    tabControl.add(tab_ops, text='Operations')
    tabControl.add(tab_config, text='Camera configuration')
    tabControl.grid(row=0,sticky='we')
    
    """
    CAMERA CONFIGURATIONS FROM .INI FILE
    """
    text_config = tk.Text(tab_config, bg = 'gray18', fg='thistle1')
    text_config.grid(row=0,sticky='we')
    
    scrollb = tk.Scrollbar(tab_config, command=text_config.yview)
    scrollb.grid(row=0, column=1, sticky='nswe')
    text_config['yscrollcommand'] = scrollb.set
    
    root.config_ini = 'config.ini'
    file_contents = open(root.config_ini).read()
    text_config.insert('end',file_contents)
    
    tbox_contents = text_config.get('1.0', 'end')
    # c.char is a fixed array, so in case future editing needs more array space,
    # an expanded array is passed as an argument
    tbox = Array(ctypes.c_char, bytes(tbox_contents+'\n'*10, 'utf8'))
    
    text_config.tag_config('section', foreground='khaki1')
    update_tboxEmbellishments()
    
    """
    SAVE CONFIGURATION CHANGES
    """
    button_saveconfig = tk.Button(tab_config, 
                       text="Save changes",
                       bg="white",
                       fg="black",
                       command=update_config)
    
    button_saveconfig.grid(row=0,column=2,padx=10,pady=10)
    
    """
    CAMERA CONNECTION/DISCONNECTION FRAME
    """
    cameraonoff_frame = tk.Frame(tab_ops)
    cameraonoff_frame.grid(row=0,sticky='we')
    
    cameraonoff_frame.grid_rowconfigure(0, weight=1)
    cameraonoff_frame.grid_columnconfigure(0, weight=1)
    cameraonoff_frame.grid_columnconfigure(1, weight=1)
    
    """
    INITIALIZE CAMERA
    """
    button_initcamera = tk.Button(cameraonoff_frame,
                       text="INITIALIZE CAMERA",
                       bg="white",
                       fg="black",
                       command=lambda: create_cameraprocess(q, release_event, \
                            beginlive_event, stoplive_event, piezostep_event, \
                            updateconfig_event, plotmidline_event, stdout_queue))
    
    button_initcamera.grid(row=0,column=0,padx=10,pady=10)
    
    """
    RELEASE CAMERA
    """
    button_release = tk.Button(cameraonoff_frame,
                       text="RELEASE CAMERA",
                       bg="white",
                       fg="black",
                       command=lambda: notify_releasecamera(release_event))
    button_release.grid(row=0,column=1,padx=10,pady=10)
    
#    """
#    EXPOSURE TIME CONFIGURATION
#    """
#    label_exposure = tk.Label(cameraonoff_frame, text='Exposure time: ')
#    label_exposure.grid(row=0,column=2,padx=10,pady=10,sticky='we')
#    
#    slider_exposure = tk.Scale(cameraonoff_frame, from_=6, to=2000, orient='horizontal')
#    slider_exposure.grid(row=0,column=3,padx=10,pady=10,sticky='we')
#    
#    entry_exposure = tk.Entry(cameraonoff_frame)
#    entry_exposure.grid(row=0,column=4,padx=10,pady=10,sticky='we')
#    entry_exposure.delete(0, 'end')
#    entry_exposure.insert(0, '')
#    entry_exposure.bind("<Return>", get_exposure)
#    
#    button_exposure = tk.Button(cameraonoff_frame, 
#                       text="Set",
#                       bg="white",
#                       fg="black",
#                       command=set_exposure)
#    button_exposure.grid(row=0,column=5,padx=10,pady=10,sticky='we')
    
    root.rowconfigure(0,weight=1)
    root.columnconfigure(0,weight=1)
    
    tab_ops.rowconfigure(0,weight=1)
    tab_ops.columnconfigure(0,weight=1)
    
    tab_config.rowconfigure(0,weight=1)
    tab_config.columnconfigure(0,weight=1)
    
    main_frame = tk.Frame(tab_ops)
    main_frame.grid(row=1,column=0,sticky='nswe')
    
    main_frame.grid_rowconfigure(0, weight=1)
    main_frame.grid_rowconfigure(1, weight=1)
    main_frame.grid_rowconfigure(2, weight=1)
    main_frame.grid_rowconfigure(3, weight=1)
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_columnconfigure(1, weight=1)
    main_frame.grid_columnconfigure(2, weight=1)
    
    imageplots_frame = tk.Frame(main_frame)
    imageplots_frame.grid(row=0,column=0,sticky='nswe')
    
    imageplots_frame.grid_rowconfigure(0, weight=1)
    imageplots_frame.grid_columnconfigure(0, weight=1)
    
    oscillation_frame = tk.Frame(main_frame)
    oscillation_frame.grid(row=0,column=1,sticky='nswe')
    
    oscillation_frame.grid_rowconfigure(0, weight=1)
    oscillation_frame.grid_columnconfigure(0, weight=1)
    
    buttons_frame = tk.Frame(main_frame)
    buttons_frame.grid(row=0,column=2,sticky='nswe')
    
    buttons_frame.grid_rowconfigure(0, weight=1)
    buttons_frame.grid_columnconfigure(0, weight=1)
    
    piezosteps_frame = tk.Frame(buttons_frame)
    piezosteps_frame.grid(row=0,column=0,sticky='nswe')
    
    piezosteps_frame.grid_rowconfigure(0, weight=1)
    piezosteps_frame.grid_columnconfigure(0, weight=1)
    
    preperation_frame = tk.Frame(buttons_frame, borderwidth=1, relief='solid')
    preperation_frame.grid(row=1,column=0,sticky='nswe')
    
    preperation_frame.grid_rowconfigure(0, weight=1)
    preperation_frame.grid_columnconfigure(0, weight=1)
    
    measurement_frame = tk.Frame(buttons_frame, borderwidth=1, relief='solid')
    measurement_frame.grid(row=2,column=0,sticky='nswe', pady=10)
    
    measurement_frame.grid_rowconfigure(0, weight=1)
    measurement_frame.grid_columnconfigure(0, weight=1)
    
    selections_frame = tk.Frame(main_frame)
    selections_frame.grid(row=2,column=0,sticky='nswe')
    
    selections_frame.grid_rowconfigure(0, weight=1)
    selections_frame.grid_columnconfigure(0, weight=1)
    
    """
    CREATE CANVAS FOR IMAGE DISPLAY
    """
#    sections = cfg.load_config_fromtkText('ALL', text_config.get('1.0', 'end'))
#    h = eval(sections[2]['height'])
#    w = eval(sections[2]['width'])
    dpi = 96.0
#    f = Figure(figsize=(w/dpi,h/dpi))
    f = Figure(figsize=(500/dpi,500/dpi), dpi=96)
    f.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    ax = f.add_subplot(111)
    ax.set_axis_off()
    img = Image.frombytes('L', (500, 500), b'\x00'*250000)
    ax.imshow(img,cmap='gray',vmin=0,vmax=65535)
    livefeed_canvas = FigureCanvasTkAgg(f, master=imageplots_frame)
    livefeed_canvas.get_tk_widget().grid(row=0,column=0,sticky='nswe')
    livefeed_canvas.draw()
    
    """
    TOOLBAR - IMAGE SHOW
    """
    toolbarimshowFrame = tk.Frame(master=imageplots_frame)
    toolbarimshowFrame.grid(row=1,column=0)
    toolbarimshow = NavigationToolbar2Tk(livefeed_canvas, toolbarimshowFrame)
    toolbarimshow.update()
    
    # create global list of different plotting colors
    colors = lspec()
    
    """
    CREATE CANVAS FOR POI AND MIDLINE PLOTTING
    """
#    dpi = 96
#    fig = Figure(figsize=(imageplots_frame.winfo_height()/dpi, imageplots_frame.winfo_width()/3/dpi))
    fig = Figure(tight_layout=True)
    a = fig.add_subplot(111)
    canvas_plot = FigureCanvasTkAgg(fig, master=imageplots_frame)
    canvas_plot.get_tk_widget().grid(row=0,column=1,sticky='nswe')
    
    """
    TOOLBAR - LINE PLOT
    """
    toolbarplotFrame = tk.Frame(master=imageplots_frame)
    toolbarplotFrame.grid(row=1,column=1)
    toolbarplot = NavigationToolbar2Tk(canvas_plot, toolbarplotFrame)
    toolbarplot.update()
    
    label_preparation = tk.Label(preperation_frame, text='P R E P A R A T I O N')
    label_preparation.grid(row=0,columnspan=2)
    
    """
    POINTS OF INTEREST SELECTION BUTTONS
    """
    button_POIenable = tk.Button(preperation_frame, 
                       text="Select POI", 
                       fg="gold2",
                       bg='grey18',
                       command=enable_POIsel)
    button_POIenable.grid(row=1,column=0,padx=0,pady=10,sticky='ew')
    
    button_POIdisable = tk.Button(preperation_frame, 
                       text="OK!", 
                       fg="black",
                       command=disable_POIsel)
    button_POIdisable.grid(row=1,column=1,padx=0,pady=0,sticky='ew')
    
    """
    REGION OF INTEREST SELECT BUTTONS
    """
    button_ROIenable = tk.Button(preperation_frame, 
                       text="Select ROI", 
                       fg="chocolate2",
                       bg='grey18',
                       command=enable_ROIsel)
    button_ROIenable.grid(row=2,column=0,padx=0,pady=0,sticky='ew')
    
#    button_ROIdisable = tk.Button(preperation_frame, 
#                       text="OK!", 
#                       fg="black",
#                       command=disable_ROIsel)
#    button_ROIdisable.grid(row=2,column=1,padx=0,pady=0,sticky='ew')
    
    simplebuttons_frame = tk.Frame(selections_frame)
    simplebuttons_frame.grid(row=0,column=0,sticky='nswe')
    
    selections_frame.grid_rowconfigure(0, weight=1)
    selections_frame.grid_columnconfigure(0, weight=1)
    
    """
    OSCILLATE PIEZO BUTTON
    """
    button_oscillate = tk.Button(preperation_frame, 
                       text="Oscillate Piezo", 
                       fg="black",
                       command=lambda: create_oscillationprocess(piezostep_event, stdout_queue))
    button_oscillate.grid(row=3,column=0,padx=0,pady=0,sticky='ew')
    
    """
    PLOT MIDLINE BUTTON
    """
    button_oscillate = tk.Button(preperation_frame, 
                   text="Display intensity across\nhor/ntal line", 
                   fg="black",
                   command=lambda: toggle_displayMidline(plotmidline_event))
    button_oscillate.grid(row=3,column=1,padx=0,pady=0,sticky='ew')
    
    """
    OPTION LIST FOR NUMBER OF PIEZO STEPS - LABEL
    """
    label_piezosteps = tk.Label(piezosteps_frame, text='Piezo steps:')
    label_piezosteps.grid(row=0,column=0,padx=0,pady=0,sticky='ew')
    
    """
    OPTION LIST FOR NUMBER OF PIEZO STEPS - VALUES
    """
    piezosteps_options = ['100','200','300','400','500','600']
    piezosteps_var = tk.StringVar()
    piezosteps_var.set(piezosteps_options[-1]) # default value
    
    optionmenu_piezosteps = tk.OptionMenu(piezosteps_frame, piezosteps_var, *piezosteps_options)
    optionmenu_piezosteps.grid(row=0,column=1,padx=0,pady=0,sticky='ew')
    
    label_measurement = tk.Label(measurement_frame, text='M E A S U R E M E N T')
    label_measurement.grid(row=0,columnspan=2)
    
    """
    EXECUTE PIEZO SCAN FOR INTERFEROGRAM STACK CAPTURING
    """
    button_oscillate = tk.Button(measurement_frame, 
                   text="Piezo scan\nCapture interferograms", 
                   fg="khaki1",
                   bg='grey18',
                   command=prepare_piezoscan)
    button_oscillate.grid(row=1,column=0,columnspan=2,padx=0,pady=10,sticky='ew')
    
    CheckVar = tk.StringVar()
    CheckVar.set('m')
    C1 = tk.Radiobutton(measurement_frame, text = 'Measurement', variable = CheckVar, value='m')
    C2 = tk.Radiobutton(measurement_frame, text = 'Calibration', variable = CheckVar, value='c')
    C1.grid(row=2,column=0,padx=0,pady=0,sticky='we')
    C2.grid(row=2,column=1,padx=0,pady=0,sticky='we')
    
    """
    EXECUTE PIEZO SCAN FOR CALIBRATION PROCESS
    """
    button_oscillate = tk.Button(measurement_frame, 
                   text="Calibrate", 
                   fg="khaki1",
                   bg='grey18',
                   command=prepare_calibrate)
    button_oscillate.grid(row=5,column=0,padx=0,pady=0,sticky='ew')
    
    """
    INSERT MEASUREMENT IMAGE STACK FOLDER NAME
    """
    label_folder = tk.Label(measurement_frame, text='Measurement\nfolder name:')
    label_folder.grid(row=3,column=0,padx=0,pady=0,sticky='ew')
    
    entry_measfolder = tk.Entry(measurement_frame)
    entry_measfolder.grid(row=3,column=1,padx=0,pady=0,sticky='we')
    entry_measfolder.delete(0, 'end')
    entry_measfolder.insert(0, 'stack')
    entry_measfolder.bind("<Return>", setMeasurementFolderName)
    
    """
    INSERT CALIBRATION IMAGE STACK FOLDER NAME
    """
    label_folder = tk.Label(measurement_frame, text='Calibration\nfolder name:')
    label_folder.grid(row=4,column=0,padx=0,pady=0,sticky='ew')
    
    entry_calibfolder = tk.Entry(measurement_frame)
    entry_calibfolder.grid(row=4,column=1,padx=0,pady=0,sticky='we')
    entry_calibfolder.delete(0, 'end')
    entry_calibfolder.insert(0, 'stack')
    entry_calibfolder.bind("<Return>", setCalibrationFolderName)
    
    """
    EXECUTE INTERFEROGRAM STACK ANALYSIS FOR SURFACE ELEVATION MAP EXTRACTION
    """
    button_oscillate = tk.Button(measurement_frame, 
                   text="Analyze", 
                   fg="khaki1",
                   bg='grey18',
                   command=prepare_analysis)
    button_oscillate.grid(row=5,column=1,padx=0,pady=0,sticky='ew')
    
#    """
#    DISPLAY POI INTERFEROGRAMS
#    """
#    button_toggleDispInterf = tk.Button(oscillation_frame, 
#                       text="Display/Hide\nPOI Interferograms", 
#                       fg="black",
#                       command=toggleDispInterf_threaded)
#    button_toggleDispInterf.grid(row=2,column=1,padx=0,pady=0,sticky='ew')

    """
    LIVE FEED BUTTON
    """
    button_live = tk.Button(simplebuttons_frame, 
                       text="Live!", 
                       fg="red",
                       bg='grey18',
                       command=lambda: notify_beginlive(beginlive_event))
    button_live.grid(row=0,column=0,padx=10,pady=10,sticky='ew')
    
    """
    STOP LIVE BUTTON
    """
    button_reset = tk.Button(simplebuttons_frame, 
                       text="Stop Live Feed", 
                       fg="red",
                       bg='grey18',
                       command=lambda: notify_stoplive(stoplive_event))
    button_reset.grid(row=0,column=1,padx=10,pady=10,sticky='we')
    
    """""""""
     || || ||
     || || ||
     VV VV VV
    """""""""
    piezo_dispaxis = np.loadtxt('Mapping_Steps_Displacement_2.txt')
    
    """
    OUTPUT TEXT
    """
    redirectstdout_frame = tk.Frame(main_frame)
    redirectstdout_frame.grid(row=3,column=0,sticky='nswe')
    
    output_text = ScrolledText(redirectstdout_frame,bg='gray18',fg='thistle1',width=75, height=10)
    output_text.see('end')
    output_text.grid(row=0,padx=10,pady=10,sticky='nswe')
    
    """
    CLEAR OUTPUT TEXT BOX
    """
    button_cleartbox = tk.Button(redirectstdout_frame, 
                       text="Clear", 
                       fg="lawn green",
                       bg='grey18',
                       command=clear_outputtext)
    button_cleartbox.grid(row=0,column=1,padx=10,pady=10,sticky='we')
    
    """
    RECTANGLE SELECTOR OBJECT - for ROI selection
    """
    toggle_selector_RS = RectangleSelector(ax, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    toggle_selector_RS.set_active(False)
    toggle_selector_RS.set_visible(False)
    
    """
    STDOUT REDIRECTION
    """
    sys.stdout = StdoutQueue(stdout_queue)
    sys.stderr = StdoutQueue(stdout_queue)
    
    # Instantiate and start the text monitor
    monitor = Thread(target=text_catcher, args=(output_text, stdout_queue))
    monitor.daemon = True
    monitor.start()
    
#    sys.stdout = StdoutRedirector(output_text)
#    sys.stderr = StderrRedirector(output_text)
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.config(menu=menubar)
    root.mainloop()
    
    
if __name__ == "__main__":
    gui()
    
    
### DUMPSTER ###

#    livefeed_canvas = tk.Canvas(imageplots_frame, height = h, width = w)
#    livefeed_canvas.grid(row=0,column=0,padx=0,pady=0,sticky='nswe')
#    
#    img = Image.frombytes('L', (w, h), b'\xff'*(w*h))
#    photo = ImageTk.PhotoImage(image=img)
#    livefeed_canvas.create_image(0,0,image=photo,anchor='nw')




#                    # live POI plotting
#                    if piezostep_event.is_set():
#                        if first_step == False:
#                            ctr = 0
#                            first_step = True
#                        else:
#                            ctr = 0 if ctr == max_ctr-1 else ctr
#                            
#                        if ctr == 0:
#                            a.clear()
#                            xp = []
#                            yp = [ [] for i in range(len(POI)) ]
#                        
#                        xp.append(ctr)
#                        
#                        for i in range(len(POI)):
#                            yp[i].append(img.getpixel(POI[i]))
#                            a.plot(xp, yp[i], colors[i]+'o-')
#                        
#                        canvas_plot.draw()
#                        root.update()
#                        ctr += 1
#                        piezostep_event.clear()
















































