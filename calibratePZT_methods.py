# -*- coding: utf-8 -*-
"""
Created on Fri May 31 20:51:32 2019

@author: Evangelos Tzardis
"""

import matplotlib.pyplot as plt
import numpy as np

try: 
    import tkFileDialog
except:  
    import tkinter.filedialog as tkFileDialog

try:
    import Tkinter
except:
    import tkinter as Tkinter

import imageio as io
import glob
from skimage.feature import peak_local_max
from matplotlib.lines import Line2D
import os

def read_IsIrIb():
    global Is
    global Ir
    global Ib
    folder = ResultDir + '\\IsIrIb'
    Is = io.imread(folder + '\\Is' + '.tiff')
    Ir = io.imread(folder + '\\Ir' + '.tiff')
    Ib = io.imread(folder + '\\Ib' + '.tiff')

def normalize_image(image):
    eps = np.finfo(np.float64).eps
    nom = 2.0*np.sqrt(Is)*np.sqrt(Ir)
    nom[np.where(nom==0)] = eps
    image = (image-Is-Ir-Ib)/nom
    
    return image

def calc_phase(image):
    f = np.fft.fft2(image)
    f = np.fft.fftshift(f)
    f_abs = np.abs(f)
    
    # Find the coordinates of local maxima
    coeff_max = peak_local_max(f_abs, min_distance=3, threshold_rel=0.1, num_peaks=3)
    
    # 1st maximum corresponds to the lowest dominant frquency of the image : 
    #                                                           coeff_max[1]
    # 2nd maximum corresponds to the fringe modulation frequency :
    #                                           coeff_max[0] or coeff_max[2]
    
    # We want to calculate the phase difference of the fourier coefficients
    # indicated by the 2nd maxima of each fourier norm image.
    cf = f[coeff_max[0,0],coeff_max[0,1]]
    
    p = np.angle(cf)
    
    return p

def im2odd_dim(image):
    rows, cols = np.shape(image)
    if rows % 2 == 0: # crop the last row if rows even
        image = image[:-1,:]
        
    if cols % 2 == 0: # crop the last column if columns even
        image = image[:,:-1]
        
    return image

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

# sort file names by numerical order indicated by name
def sort_filelist(filelist):
    numlist = []
    
    for f in filelist:
        f = f[::-1]
        idx = f.find('/')
        f = f[:idx]
        chars = []
        for i in range(len(f)):
            chars.append(f[i])
        
        dig_str = [c for c in chars if c.isdigit()]
        dig_str = dig_str[::-1]
        numstr= ''
        for dig_char in dig_str:
            numstr += dig_char
        
        num = int(numstr)
        numlist.append(num)
        sorted_idx = (np.argsort(numlist))
        sorted_filelist = [filelist[i] for i in sorted_idx]
        
    return sorted_filelist

def onpick1(event, fig):
    global sel
    global mapStart
    global mapEnd
    
    if isinstance(event.artist, Line2D):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ind = event.ind
        
        Xsel = np.take(xdata, ind)[0]
        if sel == 0:
            mapStart = int(np.round(Xsel))
            sel += 1
        elif sel == 1:
            mapEnd = int(np.round(Xsel))
            plt.close(fig)
        
        
def evaluate_mapping(incremental):
    global sel
    global mapStart
    global mapEnd 
    
    sel = 0
    
    fig, ax = plt.subplots()
    ax.set_title('click on points', picker=True)
    line, = ax.plot(incremental, 'o', picker=5)
    
    fig.canvas.mpl_connect('pick_event', lambda event: onpick1(event, fig))
    
    return incremental[mapStart:mapEnd+1]

#if 'allfiles' in globals():
#    allfiles.clear()
    
def run(steps):
    root = Tkinter.Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    
    
    ResultDir = tkFileDialog.askdirectory(parent=root,title='Pick the folder that contains the interferometer images')
    
    if ResultDir == '':
        return
    
    FileList1 = glob.glob(ResultDir+'\\'+'*.bmp')
    FileList2 = glob.glob(ResultDir+'\\'+'*.tif')
    FileList3 = glob.glob(ResultDir+'\\'+'*.tiff')
    
    allfiles = [FileList1, FileList2, FileList3]
    if len(allfiles) == 0: 
        raise Exception("No interferometer files were found")
    
    if FileList1:
        file_list = FileList1
#        ftype = '.bmp'
    elif FileList2:
        file_list = FileList2
#        ftype = '.tif'
    elif FileList3:
        file_list = FileList3
#        ftype = '.tiff'
        
    file_list = sort_filelist(file_list)
        
    # read images of intensities of sample, reference and background
    #read_IsIrIb()
        
    listlen = len(file_list)
    phases = np.zeros(listlen)
    
    image = (io.imread(file_list[0])).astype(float)
    #image = normalize_image(image)
    rows, cols = np.shape(image)
    
    image = im2odd_dim(image)
    phases[0] = calc_phase(image)   
    
    for j in range(1,listlen):
        image = ((io.imread(file_list[j])).astype(float))
    #    image = normalize_image(image)
        image = im2odd_dim(image)
        phases[j] = calc_phase(image)
            
        print('%d/%d' %(j+1,listlen))
    
    uwphases = np.unwrap(phases)
    dp = np.diff(uwphases)
    
    # Phase difference converted to displacement of fringes.
    # In michelson interferometers, displacement of the sample is equal to
    # half the optical path difference
    lamda = 670.0 # in nm
    displacements = dp/(2*np.pi)*lamda/2 # in nm
    displacements = np.concatenate([[0],displacements]) # zero as the first position
    
    incremental = np.cumsum(displacements)
    grad = np.gradient(incremental)
    grad = reject_outliers(grad, m = 2.)
    dp_mean = np.mean(grad)
    
    incremental = - incremental if dp_mean < 0 else incremental
    
    #mapping = evaluate_mapping(incremental)
    mapping = incremental
    
    
    
    """
    SAVE MAPPING
    """
    pardirs = 2
    for i in range(pardirs):
        ResultDir = os.path.abspath(os.path.join(ResultDir, os.pardir))
    
    samenamelist1 = glob.glob(ResultDir+'\\'+'Mapping_Steps_Displacement*.txt')
    
    len_snlist = len(samenamelist1)
    np.savetxt('Mapping_Steps_Displacement_'+steps+'_{}.txt'.format(str(len_snlist+1)), mapping)
    #--------
    # FOR MEASURING VIBRATIONS -- ONLY WHEN FLAT MIRRORS ON BOTH ARMS AND PZT NOT MOVING
#    np.savetxt('Displacements.txt', displacements)
#    np.savetxt('Incremental.txt', incremental)
    #--------
    
    """
    PLOT MAPPING
    """
    x = np.arange(1,listlen+1)
    plt.title('PZT incremental displacement')
    plt.xlabel('# of step')
    plt.ylabel('Displacement in $nm$')
    plt.plot(x,incremental,'-o')



















## display results
#fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
#ax = axes.ravel()
#ax[0].imshow(f_abs, cmap=plt.cm.gray)
#ax[0].axis('off')
#ax[0].set_title('Original')
#
#ax[1].imshow(image_max, cmap=plt.cm.gray)
#ax[1].axis('off')
#ax[1].set_title('Maximum filter')
#
#ax[2].imshow(f_abs, cmap=plt.cm.gray)
#ax[2].autoscale(False)
#ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
#ax[2].axis('off')
#ax[2].set_title('Peak local max')
#
#fig.tight_layout()
#
#plt.show()















#df = f2/f
#exp = np.log(df)
#shift = np.abs(exp)
#gx, gy = np.gradient(shift)
#rot_angle = np.arctan(gy/gx)
#norm = np.sqrt(gx**2 + gy**2)

########################



#sum_d = 0
#for i in range(rows):
#    peaks, _ = find_peaks(image_g[i])
#    d = np.diff(peaks)
#    d = np.average(d)
#    sum_d = sum_d + d
#    
#d = sum_d/rows
#
#rot = np.deg2rad(52.67)
#theta = (rot - np.pi/2) if (rot >= np.pi/2) else (np.pi/2 - rot)
#x = d*np.cos(theta)
#pixel = lamda/x # 'x' pixels --> 'lamda' <=> 1 pixel --> 'lamda'/'x'


#x = np.arange(cols)
#dx = np.linspace(-x[-1], x[-1], 2*cols-1)
#xcorr = correlate(image[rows//2], image2[rows//2])
#pixel_shift = dx[xcorr.argmax()]





################################

#dt = 1e-2;
#t = np.arange(0,20,dt);
#y1 = np.sin(t); h1 = hilbert(y1);
#y2 = np.sin(t+1); h2 = hilbert(y2);
#p1 = np.angle(h1); p2 = np.angle(h2); 
#p = np.unwrap(p2)-np.unwrap(p1);      
#
#fig = plt.figure()
#ax = fig.add_subplot(211)
#ax.plot(t,p1,'r',t,p2,'b');
#ax2 = fig.add_subplot(212)
#ax2.plot(t,p,'k');





