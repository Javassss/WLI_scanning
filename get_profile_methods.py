# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 20:35:53 2018

@author: Evangelos Tzardis
"""

from scipy.signal import hilbert
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
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
from scipy.signal import hilbert
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

#def gaussian_function(fwhm, width):
#    # fwmh, width in pixels
#    x = np.arange(width)
#    a = 1
#    b = width/2
#    c = fwhm/2
#    res = a*np.e**(-((x-b)**2)/(2*c**2))
#    
#    return res

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    idx_remaining = np.argwhere(s<m)
    return [data[s<m], idx_remaining]    

def plane(coords, a, b, c):
    X, Y = coords
    Z = a*X + b*Y + c
    return Z.ravel()

def gaus(x,a,x0,sigma,c):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+c

def gaus_residu(x,n,y):
    a = x[0]
    x0 = x[1]
    sigma = x[2]
    return a*np.exp(-(n-x0)**2/(2*sigma**2)) - y

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

def run(calib_linear_region, steps):
    if 'FileList1' in globals():
        FileList1.clear();
        del FileList1
    if 'FileList2' in globals():
        FileList2.clear();
        del FileList2
    if 'FileList3' in globals():
        FileList3.clear();
        del FileList3
    
    root = Tkinter.Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    
    
    ResultDir = tkFileDialog.askdirectory(parent=root,title='Pick the folder that contains the interferometer images')
    
    if ResultDir == '':
        return
    
    FileList1=glob.glob(ResultDir+'\\'+'*.bmp')
    FileList2=glob.glob(ResultDir+'\\'+'*.tif')
    FileList3=glob.glob(ResultDir+'\\'+'*.tiff')
    
    if len([FileList1, FileList2, FileList3]) == 0: 
        raise Exception("No interferometer files were found")
    
    if FileList1:
        file_list = FileList1
    #    ftype = '.bmp'
    elif FileList2:
        file_list = FileList2
    #    ftype = '.tif'
    elif FileList3:
        file_list = FileList3
    #    ftype = '.tiff'
        
    file_list = sort_filelist(file_list)
    
    # starting and ending index of mapping array   
    #start = 141
    #end = 580
    zlen = len(file_list)
    
    ROI = [0,None,0,None]
    image = ((io.imread(file_list[0]))[ROI[0]:ROI[1],ROI[2]:ROI[3]]).astype(float)
    rows, cols = np.shape(image)
    
    image_stack = np.zeros([zlen, rows, cols])
    image_stack[0] = image
    
    for i in range(1,zlen):
        image = ((io.imread(file_list[i]))[ROI[0]:ROI[1],ROI[2]:ROI[3]]).astype(float)
        image_stack[i] = image
        
    profile = np.zeros([rows, cols])
    mapping = np.loadtxt(file_list[0]+'\\..\\..\\..\\' + 'Mapping_Steps_Displacement'+steps+'.txt')
    nmap = np.arange(len(mapping))
    # start and end points of PZT almost linear movement
#    nearly_linear_region = [19,None]
    s = calib_linear_region[0]
    e = calib_linear_region[1]
    fit_coeff = np.polyfit(nmap[s:e], mapping[s:e] ,5)
    curve = np.poly1d(fit_coeff)
    
    n = np.arange(zlen)
    
    for i in range(rows):
        for j in range(cols):
            z = image_stack[:,i,j]
            zm = z - np.median(z)
            za = np.abs(zm)
            zg = gaussian_filter1d(za, sigma=30)
    #        zgn = zg/np.amax(zg)
    #        zgm = zg - np.median(zg)
            
            
    #        mean = np.sum(n*zg)/np.sum(zg)
            nmax = np.argmax(zg)
            amp = zg[nmax]
    #        sigma = np.sqrt(np.sum((n-mean)**2*zg/np.sum(zg)))
            
            try:
                sigma = 50
                c = np.median(zg)
                guess = [amp,nmax,sigma,c]
                popt, _ = curve_fit(gaus,n,zg,p0=guess)#,\
    #                                bounds=[[0.99*amp,nmax-2,1.0*sigma],[1.01*amp,nmax+2,2.0*sigma]])
                peak = popt[1]
    #            res_robust = least_squares(gaus_residu, guess, loss='soft_l1', f_scale=0.1, args=(n, zg))
    #            peak = res_robust.x[1]
                profile[i,j] = curve(peak) # with mapping
    #            profile[i,j] = peak # without mapping
            except:
                profile[i,j] = None
        print(i)
        
    profile = - profile
        
    """
    #################### tilt correction
    """
    
    x = np.arange(rows)
    y = np.arange(cols)
    x, y = np.meshgrid(y, x)
    
    """
    mask NaN values
    """
    prof = np.ma.masked_invalid(profile)
    """
    get only the valid values
    """
    x1 = x[~prof.mask]
    y1 = y[~prof.mask]
    nprof = prof[~prof.mask]
    
    """
    interpolate NaN values
    """
    prof = griddata((x1, y1), nprof.ravel(), (x, y), method='cubic')
    
    planefit_rows = 20
    planefit_cols = cols
    areafit = prof[:planefit_rows,:planefit_cols]
    #"""
    #remove possible outliers
    #"""
    #areafit, idx_remaining = reject_outliers(areafit.ravel(), m = 2.)
    #
    #"""
    #interpolate NaN values again
    #"""
    #prof = griddata((x1, y1), areafit, (x, y), method='cubic')
    
    xx = np.arange(planefit_rows)
    yy = np.arange(planefit_cols)
    xx, yy = np.meshgrid(yy, xx)
    
    po, _ = curve_fit(plane, (xx,yy), areafit.ravel())
    plane_fit = plane((x,y), *po).reshape(np.shape(prof))
    
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(x, y, profile, cmap='gnuplot')
    #fig.colorbar(surf)
    
    flat_profile = prof - plane_fit
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.set_zlim(0, 3e6)
    surf = ax.plot_surface(x, y, flat_profile, cmap='gnuplot')
    plt.show()
    fig.colorbar(surf)
    """
    ####################
    """



#"""
#################### NO tilt correction
#"""
#
#sp = np.shape(profile)
#ROI = [0,sp[0]-1,0,sp[1]-1]
##ROI = [5,49,9,49]
#area = profile[ROI[0]:ROI[1]+1,ROI[2]:ROI[3]+1]
#
#x = np.arange(ROI[0],ROI[1]+1)
#y = np.arange(ROI[2],ROI[3]+1)
#x, y = np.meshgrid(y, x)
#"""
#mask NaN values
#"""
#prof = np.ma.masked_invalid(area)
#"""
#get only the valid values
#"""
#x1 = x[~prof.mask]
#y1 = y[~prof.mask]
#nprof = prof[~prof.mask]
#
#"""
#interpolate NaN values
#"""
#prof = griddata((x1, y1), nprof.ravel(), (x, y), method='cubic')
#
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
##ax.set_zlim(0, 3e6)
#surf = ax.plot_surface(x, y, prof, cmap='gnuplot')
#plt.show()
#fig.colorbar(surf)
#
#
#"""
####################
#"""





























