# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 20:35:53 2018

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

import imageio
import glob
from scipy import fftpack
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.signal import fftconvolve, medfilt2d, correlate
from skimage.feature import peak_local_max
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
from time import time
from multiprocessing import Process, Pool, RawArray, cpu_count, log_to_stderr
import ctypes

import profile_processing as postproc
import logging

#lts = log_to_stderr()
#lts.setLevel(logging.INFO)

#%%
#def gaussian_function(fwhm, width):
#    # fwmh, width in pixels
#    x = np.arange(width)
#    a = 1
#    b = width/2
#    c = fwhm/2
#    res = a*np.e**(-((x-b)**2)/(2*c**2))
#    
#    return res

#%%
def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    idx_remaining = np.argwhere(s<m)
    
#    plt.figure()
#    plt.imshow(data)
#    plt.figure()
#    plt.imshow(fixed)
    
    data[s<m]
    
    return [data[s<m], idx_remaining]

##%%
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

#%%
"""
Run the analysis. If interferogram stack folder is given, execute on that, else ask for directory.
If 'pp' is True execute profile post-processing
"""
def run(*args, pp=True):
    
    if len(args) == 1:
        ResultDir = args[0]
    elif len(args) == 2:
        ResultDir = args[0]
        mask_center = args[1]
    else:
        root = Tkinter.Tk()
        root.wm_attributes('-topmost', 1)
        root.withdraw()
        
        
        ResultDir = tkFileDialog.askdirectory(parent=root,
                                              title='Pick the folder that contains the interferometer images')
    
#    if 'FileList1' in globals():
#        FileList1.clear();
#        del FileList1
#    if 'FileList2' in globals():
#        FileList2.clear();
#        del FileList2
#    if 'FileList3' in globals():
#        FileList3.clear();
#        del FileList3
    
    FileList1=glob.glob(ResultDir+'\\'+'*.bmp')
    FileList2=glob.glob(ResultDir+'\\'+'*.tif')
    FileList3=glob.glob(ResultDir+'\\'+'*.tiff')
    
    if len([FileList1, FileList2, FileList3]) == 0: 
        raise Exception("No interferometer files were found")
    
    if FileList1:
        file_list = FileList1
    elif FileList2:
        file_list = FileList2
    elif FileList3:
        file_list = FileList3
        
    file_list = sort_filelist(file_list)
    
    # starting and ending index of mapping array   
    #start = 141
    #end = 580
    zlen = len(file_list)
    
    # ROI selection
    """""""""
    |   |   |
    |   |   |
    V   V   V
    """""""""
#    ROI = [170,190,140,150]
    ROI = [0,None,0,None]
    image = ((imageio.imread(file_list[0]))[ROI[0]:ROI[1],ROI[2]:ROI[3]]).astype(float)
    rows, cols = np.shape(image)
    
    image_stack = np.zeros([zlen, rows, cols])
    image_stack[0] = image
    
    for i in range(1,zlen):
        image = ((imageio.imread(file_list[i]))[ROI[0]:ROI[1],ROI[2]:ROI[3]]).astype(float)
        image_stack[i] = image
        
    profile = np.zeros([rows, cols])
    # Calibration file
    """""""""
    |   |   |
    |   |   |
    V   V   V
    """""""""
#    mapping = np.loadtxt(file_list[0]+'\\..\\..\\..\\' + 'Mapping_Steps_Displacement_AVERAGE.txt')
    mapping = np.loadtxt(file_list[0]+'\\..\\..\\..\\' + 'Mapping_Steps_Displacement_afterMirrorMeasurement.txt')
    nmap = np.arange(len(mapping))
    
    # start and end points of piezo actuator 'almost linear' movement
    """""""""
    |   |   |
    |   |   |
    V   V   V
    """""""""
    nearly_linear_region = [19,580]
    
    s = nearly_linear_region[0]
    e = nearly_linear_region[1]
    fit_coeff = np.polyfit(nmap[s:e], mapping[s:e] ,5)
    curve = np.poly1d(fit_coeff)
    
    n = np.arange(zlen)
    
    # get folder name
    idx = ResultDir.rfind('/')
    pname = ResultDir[idx+1:]
    
    """
    GET RAW
    """
    #--------------------------------------------------------------------------
    # UNCOMMENT FOR MULTI-CORE ANALYSIS 4
    
#    t = time()
#    #TODO
#    profile, nonNaN_idx = analysis4(image_stack)
#    
#    for k in nonNaN_idx:
#        i, j = k[0], k[1]
#        noncalibrated_value = profile[i,j]
#        calibrated_value = curve(noncalibrated_value)
#        profile[i,j] = calibrated_value
#        
#    elapsed_time = time() - t
#    print(elapsed_time)
#            
#    profile = - profile
#    
#    np.savetxt('raw_profile_' + pname + '_mp.txt', profile)
#    imageio.imsave('raw_profile_' + pname + '_mp.tiff', profile)    
    
    #--------------------------------------------------------------------------
    # UNCOMMENT FOR SINGLE-CORE ANALYSIS
    
    #TODO
    t = time()
    profile = analysis_masked(image_stack, rows, cols, curve, n, profile, pname, mask_center)
#    profile = analysis(image_stack, rows, cols, curve, n, profile)
    elapsed_time = time() - t
    print(elapsed_time)
    
    profile = - profile
    #--------------------------------------------------------------------------
    
    """
    SAVE RAW
    """
    np.savetxt('raw_profile_' + pname + '.txt', profile)
    imageio.imsave('raw_profile_' + pname + '.tiff', profile)
    
    """
    SAVE interpolated w/o NaNs
    """
    interp_prof = postproc.interpolate_nan(profile)
    np.savetxt('profile_' + pname + '.txt', interp_prof)
    imageio.imsave('profile_' + pname + '.tiff', interp_prof)
    
    """
    POST-PROCESS
    """
    if pp == True:
        postproc.profile_editing(interp_prof)



#%%

#sigma_h = 100
#s_h = sigma_h*np.ones(len(n))
#z = image_stack[:,15,15]
#n = np.arange(len(z))
#zm = z - np.median(z)
#za = np.abs(zm)
#zg = gaussian_filter1d(za, sigma=30)
#
#
#nmax = np.argmax(zg)
#amp = zg[nmax]
#
#sigma = 60 # pixels
#c = np.median(zg)
#guess = [amp,nmax,sigma,c]
#popt, pc = curve_fit(gaus,n,zg,p0=guess, sigma=s_h ,absolute_sigma=True)
#
#peak = popt[1] # pixels
#h_mean = curve(peak)
#perr = np.sqrt(np.diag(pc))
#nmax_err = perr[1]
#nmaxErr_nm = abs(h_mean-curve(peak-nmax_err))
#
#if ~np.isinf(nmaxErr_nm):
#    h_uc = nmaxErr_nm
#








#%%
#TODO

def gaus(x,a,x0,sigma,c):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+c

def analysis(image_stack, rows, cols, curve, n, profile):
    global h_uc, uc_median

    """""""""
    |   |   |
    |   |   |
    V   V   V
    """""""""
    # h : height -- nm
    sigma_h = 15
    s_h = sigma_h*np.ones(len(n))
    # uncertainty in elevation is the average of all the individual uncertainties per pixel
    h_uc = []
    
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
                sigma = 60 # pixels
                c = np.median(zg)
                guess = [amp,nmax,sigma,c]
                popt, pc = curve_fit(gaus,n,zg,p0=guess, sigma=s_h ,absolute_sigma=False)
                
                peak = popt[1] # pixels
                h_mean = curve(peak)
                perr = np.sqrt(np.diag(pc))
                nmax_err = perr[1]
                nmaxErr_nm = abs(h_mean-curve(peak-nmax_err))
                
                if ~np.isinf(nmaxErr_nm) and ~np.isnan(nmaxErr_nm):
                    h_uc.append(nmaxErr_nm)
                profile[i,j] = h_mean
            except:
                profile[i,j] = None
        print(i)
    
    h_uc = np.asarray(h_uc)
    huc_median = np.median(h_uc)
    print('elevation uc: %f'%huc_median)
    
    return profile

#%%
#TODO
def create_circular_mask(pname, img, h, w, radius=None, center=None):
    
    if center is None:
        plt.imshow(img)
        c = plt.ginput(1)
        center = (c[0][0], c[0][1])

#    if center is None: # use the middle of the image
#        center = (int(w/2), int(h/2))
    
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    
    f = open("mask_radii.txt","a+")
    f.write('Sample: %s, %d px\n'%(pname, radius))
    
    return mask

def analysis_masked(image_stack, rows, cols, curve, n, profile, pname, mask_center):
    global h_uc, uc_median

    """""""""
    |   |   |
    |   |   |
    V   V   V
    """""""""
    # h : height -- nm
    sigma_h = 15
    s_h = sigma_h*np.ones(len(n))
    # uncertainty in elevation is the average of all the individual uncertainties per pixel
    h_uc = []
    
    imstack_depth = np.shape(image_stack)[0]
    img = image_stack[imstack_depth//2]
    mask = create_circular_mask(pname, img, rows, cols, center=mask_center)
    
    for i in range(rows):
        for j in range(cols):
            
            if mask[i,j]:
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
                    sigma = 60 # pixels
                    c = np.median(zg)
                    guess = [amp,nmax,sigma,c]
                    popt, pc = curve_fit(gaus,n,zg,p0=guess, sigma=s_h ,absolute_sigma=False)
                    
                    peak = popt[1] # pixels
                    h_mean = curve(peak)
                    perr = np.sqrt(np.diag(pc))
                    nmax_err = perr[1]
                    nmaxErr_nm = abs(h_mean-curve(peak-nmax_err))
                    
                    if ~np.isinf(nmaxErr_nm) and ~np.isnan(nmaxErr_nm):
                        h_uc.append(nmaxErr_nm)
                    profile[i,j] = h_mean
                except:
                    profile[i,j] = None
                
        print(i)
    
    h_uc = np.asarray(h_uc)
    huc_median = np.median(h_uc)
    print('elevation uc: %f'%huc_median)
    
    return profile

#%%

def gaus2(x,x0,c):
    sigma = 60
    a = 1.0
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+c    

def analysis2(image_stack, rows, cols, curve, n, profile):

    for i in range(rows):
        for j in range(cols):
            z = image_stack[:,i,j]
            zm = z - np.median(z)
            za = np.abs(zm)
            zg = gaussian_filter1d(za, sigma=30)
    #        zgn = zg/np.amax(zg)
    #        zgm = zg - np.median(zg)
            zn = zg/np.amax(zg)
            
            nmax = np.argmax(zn)
#            mean = np.sum(n*zg)/np.sum(zg)
#            sigma = np.sqrt(np.sum((n-mean)**2*zg/np.sum(zg)))/2
            sigma = 60
            c = 0.0
#            amp = 1.0
            guess = [nmax,c]
            area = np.arange(nmax-sigma, nmax+sigma)
            
            try:
                popt, _ = curve_fit(gaus2,n[area],zn[area],p0=guess)#,\
    #                                bounds=[[0.99*amp,nmax-2,1.0*sigma],[1.01*amp,nmax+2,2.0*sigma]])
                peak = popt[0]
    #            res_robust = least_squares(gaus_residu, guess, loss='soft_l1', f_scale=0.1, args=(n, zg))
    #            peak = res_robust.x[1]

                profile[i,j] = curve(peak) # with mapping
    #            profile[i,j] = peak # without mapping
            except:
                profile[i,j] = None
#        print(i)
    
    return profile

#%%

# A global dictionary storing the variables passed from the initializer.
global var_dict, profile
var_dict = {}

def perpixel(i, j):
    
    print(1)
    IS_np = np.frombuffer(var_dict['IS']).reshape(var_dict['IS_shape'])
    N_np = np.frombuffer(var_dict['N'])
    
    z = IS_np[:,i,j]
    zm = z - np.median(z)
    za = np.abs(zm)
    zg = gaussian_filter1d(za, sigma=30)
    zn = zg/np.amax(zg)
    
    nmax = np.argmax(zn)
    sigma = 60
    c = 0.0
    guess = [nmax,c]
    area = np.arange(nmax-sigma, nmax+sigma)
    
    try:
        print(2)
        popt, _ = curve_fit(gaus2,N_np[area],zn[area],p0=guess)
        peak = popt[0]

        elevation = peak # curve(peak) will be computed in the parent process
    except:
        print(3)
        elevation = None
    
    return (i, j, elevation)

def collect_result(elevation_results):
    
    print(4)
    i, j, elevation = elevation_results
    print(5)
#    profile[i,j] = elevation

def init_worker(IS, IS_shape, N):
    global var_dict
    
    print(6)
    var_dict['IS'] = IS
    var_dict['IS_shape'] = IS_shape
#    var_dict['P'] = P
    var_dict['N'] = N
    print(7)

def analysis3(image_stack):
    
    global all_results, profile
    
    # Create shared memory arrays
    IS_shape = np.shape(image_stack)
    P_shape = IS_shape[1:]
    n = np.arange(IS_shape[0])
    profile = np.zeros(P_shape).astype(float)
    IS = RawArray(ctypes.c_float, image_stack.ravel())
#    P = RawArray('f', IS_shape[1] * IS_shape[2])
    N = RawArray(ctypes.c_int, n.ravel())
    # Start the process pool and do the computation.
    # Here we pass IS and IS_shape to the initializer of each worker.
    # (Because IS_shape is not a shared variable, it will be copied to each
    # child process.)
    with Pool(processes=cpu_count(), initializer=init_worker, initargs=(IS, IS_shape, N)) as pool:
#        for i in range(IS_shape[1]):
#            for j in range(IS_shape[2]):
        for i in range(1):
            for j in range(1):
                all_results = pool.apply_async(perpixel, args=(i, j), callback=collect_result)
                print(i)
#        pool.close()
#        pool.join()          
#        result = pool.map(worker_func, range(IS_shape[0]))
    
    
#    for i in range(rows):
#        for j in range(cols):
#            pool.apply_async(perpixel, args=(i, j), callback=collect_result)
#    
#    pool.close()
#    pool.join()
    
#    results.sort(key=lambda x: x[0])
#    results_final = [r for i, r in results]

#%%
#TODO

def per_segment(IS, rows, cols, depth, P, MU):
    
    n = np.arange(depth)
    seg = np.frombuffer(IS,dtype=ctypes.c_float).reshape([depth, rows, cols])
    
    for i in range(rows):
        for j in range(cols):
            z = seg[:,i,j]
            zm = z - np.median(z)
            za = np.abs(zm)
            zg = gaussian_filter1d(za, sigma=30)
            zn = zg/np.amax(zg)
            
            nmax = np.argmax(zn)
            sigma = 60
            c = 0.0
            guess = [nmax,c]
            area = np.arange(nmax-sigma, nmax+sigma)
            
            try:
                popt, _ = curve_fit(gaus2,n[area],zn[area],p0=guess)
                peak = popt[0]
                mark_unreliability = 0
        
                elevation = peak # curve(peak) will be computed in the parent process
#                print('ok: %d %d: %.2f' % (i,j, elevation))
            except:
#                print('not ok: %d %d' % (i,j))
                elevation = 0.0 # it will be marked as unreliable and changed to None in the parent process
                mark_unreliability = 1
            
            # Store result in the shared array (single-indexed)
            P[i*cols + j] = elevation
            MU[i*cols + j] = mark_unreliability
            
#        print(i)

def analysis4(image_stack):
    
    Nprocs = cpu_count()
    IS_shape = np.shape(image_stack)
    P_shape = IS_shape[1:]
    
    if Nprocs == 2:
        seg1 = image_stack[:, 0:P_shape[0]//2, 0:P_shape[1]]
        seg2 = image_stack[:, P_shape[0]//2:P_shape[0], 0:P_shape[1]]
        
        IS_1 = RawArray(ctypes.c_float, seg1.ravel())
        IS_2 = RawArray(ctypes.c_float, seg2.ravel())
        
    elif Nprocs == 4:
        seg1 = image_stack[:, 0:P_shape[0]//2, 0:P_shape[1]//2]
        seg2 = image_stack[:, 0:P_shape[0]//2, P_shape[1]//2:P_shape[1]]
        seg3 = image_stack[:, P_shape[0]//2:P_shape[0], 0:P_shape[1]//2]
        seg4 = image_stack[:, P_shape[0]//2:P_shape[0], P_shape[1]//2:P_shape[1]]
        
        IS_1 = RawArray(ctypes.c_float, seg1.ravel())
        IS_2 = RawArray(ctypes.c_float, seg2.ravel())
        IS_3 = RawArray(ctypes.c_float, seg3.ravel())
        IS_4 = RawArray(ctypes.c_float, seg4.ravel())
        
        r1, c1 = P_shape[0]//2, P_shape[1]//2
        r2, c2 = P_shape[0]//2, P_shape[1]-P_shape[1]//2
        r3, c3 = P_shape[0]-P_shape[0]//2, P_shape[1]//2
        r4, c4 = P_shape[0]-P_shape[0]//2, P_shape[1]-P_shape[1]//2
        d = IS_shape[0]
        
        P_1 = RawArray(ctypes.c_float, r1 * c1)
        P_2 = RawArray(ctypes.c_float, r2 * c2)
        P_3 = RawArray(ctypes.c_float, r3 * c3)
        P_4 = RawArray(ctypes.c_float, r4 * c4)
        
        MU_1 = RawArray(ctypes.c_short, r1 * c1)
        MU_2 = RawArray(ctypes.c_short, r2 * c2)
        MU_3 = RawArray(ctypes.c_short, r3 * c3)
        MU_4 = RawArray(ctypes.c_short, r4 * c4)
        
        p1 = Process(target=per_segment, args=(IS_1, r1, c1, d, P_1, MU_1))
        p2 = Process(target=per_segment, args=(IS_2, r2, c2, d, P_2, MU_2))
        p3 = Process(target=per_segment, args=(IS_3, r3, c3, d, P_3, MU_3))
        p4 = Process(target=per_segment, args=(IS_4, r4, c4, d, P_4, MU_4))
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        
        p1.join()
        p2.join()
        p3.join()
        p4.join()
        
        P1_np = np.frombuffer(P_1,dtype=ctypes.c_float).reshape([r1, c1])
        P2_np = np.frombuffer(P_2,dtype=ctypes.c_float).reshape([r2, c2])
        P3_np = np.frombuffer(P_3,dtype=ctypes.c_float).reshape([r3, c3])
        P4_np = np.frombuffer(P_4,dtype=ctypes.c_float).reshape([r4, c4])
        
        MU1_np = np.frombuffer(MU_1,dtype=ctypes.c_short).reshape([r1, c1])
        MU2_np = np.frombuffer(MU_2,dtype=ctypes.c_short).reshape([r2, c2])
        MU3_np = np.frombuffer(MU_3,dtype=ctypes.c_short).reshape([r3, c3])
        MU4_np = np.frombuffer(MU_4,dtype=ctypes.c_short).reshape([r4, c4])
        
        idxUnr1_i, idxUnr1_j = np.where(MU1_np==1)
        idxUnr2_i, idxUnr2_j = np.where(MU2_np==1)
        idxUnr3_i, idxUnr3_j = np.where(MU3_np==1)
        idxUnr4_i, idxUnr4_j = np.where(MU4_np==1)
        
        if len(idxUnr1_i):
            P1_np[idxUnr1_i, idxUnr1_j] = np.nan
        if len(idxUnr2_i):
            P2_np[idxUnr2_i, idxUnr2_j] = np.nan
        if len(idxUnr3_i):
            P3_np[idxUnr3_i, idxUnr3_j] = np.nan
        if len(idxUnr4_i):
            P4_np[idxUnr4_i, idxUnr4_j] = np.nan
        
        p1 = np.concatenate((P1_np, P2_np), axis=1)
        p2 = np.concatenate((P3_np, P4_np), axis=1)
        Profile = np.concatenate((p1,p2), axis=0)
        
        nonNaN_idx = np.argwhere(~np.isnan(Profile))
        
        return [Profile, nonNaN_idx]  
        
        

#%%
#def analysis2(image_stack, rows, cols, curve, n, profile):
#
#    for i in range(rows):
#        for j in range(cols):
#            z = image_stack[:,i,j]
#            zm = z - np.median(z)
#            za = np.abs(zm)
#            zg = gaussian_filter1d(za, sigma=30)
#    #        zgn = zg/np.amax(zg)
#    #        zgm = zg - np.median(zg)
#            
#            
#    #        mean = np.sum(n*zg)/np.sum(zg)
#            nmax = np.argmax(zg)
#            amp = zg[nmax]
#    #        sigma = np.sqrt(np.sum((n-mean)**2*zg/np.sum(zg)))
#            
#            sigma = 50
#            c = np.median(zg)
#            guess = [amp,nmax,sigma,c]
#            popt, _ = curve_fit(gaus,n,zg,p0=guess)#,\
##                                bounds=[[0.99*amp,nmax-2,1.0*sigma],[1.01*amp,nmax+2,2.0*sigma]])
#            peak = popt[1]
##            res_robust = least_squares(gaus_residu, guess, loss='soft_l1', f_scale=0.1, args=(n, zg))
##            peak = res_robust.x[1]
#            profile[i,j] = curve(peak) # with mapping
##            profile[i,j] = peak # without mapping
#            profile[i,j] = None
#        print(i)
#        
#    np.savetxt('PROFILE.txt', profile)
#    
#    return profile
#
##%%
#def analysis3(image_stack, rows, cols, curve, n, profile):
#    
#        p, r, c = np.shape(image_stack) # pillars, rows, columns
#        
#        """
#        Select a point of interest to obtain absolute point elevation
#        """
#        
#        # double index in "2D layer" matrix = image_stack[x,:,:] 
#        poi_di = [r//2, c//2]
#        # single index (2D layer flattened)
#        poi_si = poi_di[0]*c + poi_di[1]
#        
#        """
#        Absolute elevation analysis for POI
#        """
#        i, j = poi_di
#        z = image_stack[:,i,j]
#        
#        z = z2
#        
#        zm = z - np.median(z)
#        za = np.abs(zm)
#        zg = gaussian_filter1d(za, sigma=30)
##        zgn = zg/np.amax(zg)
##        zgm = zg - np.median(zg)
#        
#        
##        mean = np.sum(n*zg)/np.sum(zg)
#        nmax = np.argmax(zg)
#        amp = zg[nmax]
##        sigma = np.sqrt(np.sum((n-mean)**2*zg/np.sum(zg)))
#        
#        try:
#            sigma = 50
#            c = np.median(zg)
#            guess = [amp,nmax,sigma,c]
#            popt, _ = curve_fit(gaus,n,zg,p0=guess)#,\
##                                bounds=[[0.99*amp,nmax-2,1.0*sigma],[1.01*amp,nmax+2,2.0*sigma]])
#            peak = popt[1]
##            res_robust = least_squares(gaus_residu, guess, loss='soft_l1', f_scale=0.1, args=(n, zg))
##            peak = res_robust.x[1]
#            p2 = curve(peak) # with mapping
##            profile[i,j] = peak # without mapping
#        except:
#            p2 = None
#        
#        """
#        Relative elevation analysis - with reference to POI
#        """
#        for k in range():
#            pass
            
            
            
#%%
#TODO
def run_multiple():
    
#    root = Tkinter.Tk()
#    root.wm_attributes('-topmost', 1)
#    root.withdraw()
#    
#    
#    ResultDir = tkFileDialog.askdirectory(parent=root,title='Pick the folder that contains the interferometer images')
    
#    directory = 'C://Users//JarvanXIV//Documents//NotRandomFiles//Projects//VisionOptics//forth_uoc//javas//PZT Interferometer//images_measurement//'
    directory = 'C:/Users/JarvanXIV/Documents/NotRandomFiles/Projects/VisionOptics/forth_uoc/javas/PZT Interferometer/_images_measurement/'
    stack_names = ['200310_2015VT EKSMA_wedge05d_1_12', '200310_2015VT EKSMA_wedge05d_1_13',
                   '200310_2015VT EKSMA_wedge05d_2_11', '200310_2015VT EKSMA_wedge05d_2_12',
                   '200310_2015VT EKSMA_wedge1d_1_11', '200310_2015VT EKSMA_wedge1d_1_12',
                   '200310_2015VT EKSMA_wedge1d_1_13', '200310_2015VT EKSMA_wedge1d_1_21',
                   '200310_2015VT EKSMA_wedge1d_1_22', '200310_2015VT EKSMA_wedge1d_1_23']

    rn = range(len(stack_names))
    stack_directory = [ '' for x in rn ]
    for i in rn:
        stack_directory[i] = directory + stack_names[i]
        print(stack_directory[i])
        run(stack_directory[i], predefcenter, pp='False')
            
        
#             # Current POI
#            i, j = poi_di
#            z = image_stack[:,i,j]
#            f = np.fft.fft(z)
#            f = np.fft.fftshift(f)
#            f_abs = np.abs(f)
#            coeff_max = peak_local_max(f_abs, min_distance=3, threshold_rel=0.1, num_peaks=3)
#            cf = f[coeff_max[1]]
#            p1 = np.angle(cf)
#            
#            # Next POI (adjecent to the current one)
#            poi_si = ( poi_di[0]*c + poi_di[1] ) - 1
#            poi_di = int(poi_si/c) + poi_si%c
#            i, j = poi_di
#            z = image_stack[:,i,j]
#            f = np.fft.fft(z)
#            f = np.fft.fftshift(f)
#            f_abs = np.abs(f)
#            coeff_max = peak_local_max(f_abs, min_distance=3, threshold_rel=0.1, num_peaks=3)
#            cf = f[coeff_max[1]]
#            p2 = np.angle(cf)
            
            #%%
#            # Current POI
#            i, j = poi_di
#            z1 = image_stack[:,i,j]
#            
#            # Next POI (adjecent to the current one)
#            poi_si = ( poi_di[0]*c + poi_di[1] ) - 100
#            poi_di = [int(poi_si/c), poi_si%c]
#            i, j = poi_di
#            z2 = image_stack[:,i,j]
#            
#            dz = np.argmax(correlate(z1,z2))
#            dzz = np.argmax(correlate(z2,z1))
            

#%%
#def remove_hotpxl(toFix):
##    toFix = profile
#    #toFix = flat_profile
#    #toFix = fixed_image
#    blurred = medfilt2d(toFix)
#    difference = toFix - blurred
#    threshold = 0.01*np.std(difference)
#    
#    #find the hot pixels, but ignore the edges
#    hot_pixels = np.nonzero((np.abs(difference[1:-1,1:-1])>threshold) )
#    hot_pixels = np.array(hot_pixels) + 1 #because we ignored the first row and first column
#    
#    fixed_image = np.copy(toFix) #This is the image with the hot pixels removed
#    for y,x in zip(hot_pixels[0],hot_pixels[1]):
#        fixed_image[y,x]=blurred[y,x]
#    
#    ###
#    height,width = np.shape(toFix)
#    
#    ###Now get the pixels on the edges (but not the corners)###
#    
#    #left and right sides
#    for index in range(1,height-1):
#        #left side:
#        med  = np.median(toFix[index-1:index+2,0:2])
#        diff = np.abs(toFix[index,0] - med)
#        if diff>threshold: 
#            hot_pixels = np.hstack(( hot_pixels, [[index],[0]]  ))
#            fixed_image[index,0] = med
#    
#        #right side:
#        med  = np.median(toFix[index-1:index+2,-2:])
#        diff = np.abs(toFix[index,-1] - med)
#        if diff>threshold: 
#            hot_pixels = np.hstack(( hot_pixels, [[index],[width-1]]  ))
#            fixed_image[index,-1] = med
#    
#    #Then the top and bottom
#    for index in range(1,width-1):
#        #bottom:
#        med  = np.median(toFix[0:2,index-1:index+2])
#        diff = np.abs(toFix[0,index] - med)
#        if diff>threshold: 
#            hot_pixels = np.hstack(( hot_pixels, [[0],[index]]  ))
#            fixed_image[0,index] = med
#    
#        #top:
#        med  = np.median(toFix[-2:,index-1:index+2])
#        diff = np.abs(toFix[-1,index] - med)
#        if diff>threshold: 
#            hot_pixels = np.hstack(( hot_pixels, [[height-1],[index]]  ))
#            fixed_image[-1,index] = med
#    
#    ###Then the corners###
#    
#    #bottom left
#    med  = np.median(toFix[0:2,0:2])
#    diff = np.abs(toFix[0,0] - med)
#    if diff>threshold: 
#        hot_pixels = np.hstack(( hot_pixels, [[0],[0]]  ))
#        fixed_image[0,0] = med
#    
#    #bottom right
#    med  = np.median(toFix[0:2,-2:])
#    diff = np.abs(toFix[0,-1] - med)
#    if diff>threshold: 
#        hot_pixels = np.hstack(( hot_pixels, [[0],[width-1]]  ))
#        fixed_image[0,-1] = med
#    
#    #top left
#    med  = np.median(toFix[-2:,0:2])
#    diff = np.abs(toFix[-1,0] - med)
#    if diff>threshold: 
#        hot_pixels = np.hstack(( hot_pixels, [[height-1],[0]]  ))
#        fixed_image[-1,0] = med
#    
#    #top right
#    med  = np.median(toFix[-2:,-2:])
#    diff = np.abs(toFix[-1,-1] - med)
#    if diff>threshold: 
#        hot_pixels = np.hstack(( hot_pixels, [[height-1],[width-1]]  ))
#        fixed_image[-1,-1] = med 
#    ###
#    
#    display2d(toFix)
#    display2d(fixed_image)
#    
#    
#    
#    return fixed_image

#%%


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
    
#"""
####################
#"""

#%%
## Select ROI and LOI for linear and surface profile rotation
#"""""""""
#|  |  |
#|  |  |
#V  V  V
#"""""""""
#r, c = np.shape(profile)
#"""
#full ROI and LOI
#"""
##rroi = [0,r]
##croi = [0,c]
##lroi = [0,c]
##l = r//2
#"""
#custom
#"""
#rroi = [0,354]
#croi = [250,392]
##lroi = [0,c]
##l = r//2
###
#cropped = profile[rroi[0]:rroi[1], croi[0]:croi[1]]
#prof = interpolate_nan(cropped)
#prof = interpolate_nan(profile)
#fixed = remove_hotpxl(prof)
#flat = rotate(fixed, [rroi[0], rroi[1]], [croi[0], croi[1]])
#flatline = rotate_line(fixed[l], [lroi[0], lroi[1]])






#%%

#m=400.
#prof = interpolate_nan(profile)
#data = prof
#d = np.abs(data - np.median(data))
#mdev = np.median(d)
#s = d/mdev if mdev else 0.
#fixed = data
#fixed[s>m] = np.nan
#fixed = interpolate_nan(fixed)
#plt.plot(prof[130]);plt.plot(fixed[130])
#
#
#
##profile = np.loadtxt('C:/Users/JarvanXIV/Documents/NotRandomFiles/Projects/VisionOptics/forth_uoc/javas/PZT Interferometer/profile_polymer_Ti3070_rpm4000_1o2_5.txt')
##profile = profile[50:]
##prof = interpolate_nan(profile)
##fixed = remove_hotpxl(prof)
##fixed = remove_hotpxl(fixed)
##fixed = remove_hotpxl(fixed)
###fixed = remove_hotpxl(fixed)
##np.savetxt('fixed5.txt',fixed)
#flat = rotate(fixed, [0, 178], [0, 100])
#np.savetxt('flat.txt',flat)
##flat5 = flat




#image = flat1
#r, c = np.shape(image)
#x = np.arange(r)
#y = np.arange(c)
#x, y = np.meshgrid(y, x)
#
#xx = np.arange(, )
#yy = np.arange(, )
#xx, yy = np.meshgrid(yy, xx)
#
#po, _ = curve_fit(plane, (xx,yy), areafit.ravel())
#plane_fitpolymer = plane((x,y), *po).reshape(np.shape(image))
#plane_fitglass = plane((x,y), *po).reshape(np.shape(image))


#px = 10
#px = 10.426
#
#rows, cols = np.shape(flat1)
#avg1 = np.average(flat1[:,120:240],axis=1)
#avg2 = np.average(flat2[:,120:240],axis=1)
#avg3 = np.average(flat3[:,120:240],axis=1)
#avg4 = np.average(flat4[:,120:240],axis=1)
#avg5 = np.average(flat5[:,120:240],axis=1)
#tavg = np.average(np.vstack((avg1, avg2, avg3, avg4, avg5)), axis=0)
#x = np.arange(rows)*px
#l1= plt.plot(x,avg1,'+', label = 'Average profile, meas. #1')
#l2= plt.plot(x,avg2,'+', label = 'Average profile, meas. #2')
#l3= plt.plot(x,avg3,'+', label = 'Average profile, meas. #3')
#l4= plt.plot(x,avg4,'+', label = 'Average profile, meas. #4')
#l5 = plt.plot(x,avg5,'+', label = 'Average profile, meas. #5')
#l6= plt.plot(x, tavg, 'b-', label = 'Total average')
#plt.title('Average of 150 adjecent individual vertical linear profiles(columns) taken from the 2D profile(608 columns)')
#plt.xlabel('Transverse x-axis ($\mu m$)')
#plt.ylabel('Elevation ($nm$)')
#h = 2950
#l = 3340
#plt.plot([h, l], [avg1[h//10], avg1[l//10]], 'rD')
#plt.plot([h, l], [avg2[h//10], avg2[l//10]], 'rD')
#plt.plot([h, l], [avg3[h//10], avg3[l//10]], 'rD')
#plt.plot([h, l], [avg4[h//10], avg4[l//10]], 'rD')
#plt.plot([h, l], [avg5[h//10], avg5[l//10]], 'rD')
#
#ph, _ = curve_fit(line, x[0:h//px], tavg[0:h//px])
#hl = line(x[0:h//px], *ph)
#l7= plt.plot(x[0:h//px], hl, 'b-.', label = 'Linear fit: higher level')
#pl, _ = curve_fit(line, x[l//px:], tavg[l//px:])
#ll = line(x[l//px:], *pl)
#l8= plt.plot(x[l//px:], ll, 'b-.', label = 'Linear fit: lower level')
#
#mhl = np.mean(hl)
#mll = np.mean(ll)
#l9 = plt.plot(x, mhl*np.ones(rows), 'k--', label = 'Mean elevation of higher level')
#l9 = plt.plot(x, mll*np.ones(rows), 'k--', label = 'Mean elevation of lower level')
#
#plt.text(0.5, 0.5, 'Average step : %.f $nm$\n' %(mhl-mll), fontsize=12)
#
#plt.legend()


#flat1 = np.loadtxt('C:/Users/JarvanXIV/Documents/NotRandomFiles/Projects/VisionOptics/forth_uoc/javas/PZT Interferometer/results/5_polymers/Ti3070_rpm5000_1o2_5measurements/flat1.txt')
#flat2 = np.loadtxt('C:/Users/JarvanXIV/Documents/NotRandomFiles/Projects/VisionOptics/forth_uoc/javas/PZT Interferometer/results/5_polymers/Ti3070_rpm5000_1o2_5measurements/flat2.txt')
#flat3 = np.loadtxt('C:/Users/JarvanXIV/Documents/NotRandomFiles/Projects/VisionOptics/forth_uoc/javas/PZT Interferometer/results/5_polymers/Ti3070_rpm5000_1o2_5measurements/flat3.txt')
#flat4 = np.loadtxt('C:/Users/JarvanXIV/Documents/NotRandomFiles/Projects/VisionOptics/forth_uoc/javas/PZT Interferometer/results/5_polymers/Ti3070_rpm5000_1o2_5measurements/flat4.txt')
#flat5 = np.loadtxt('C:/Users/JarvanXIV/Documents/NotRandomFiles/Projects/VisionOptics/forth_uoc/javas/PZT Interferometer/results/5_polymers/Ti3070_rpm5000_1o2_5measurements/flat5.txt')
#avg1 = np.average(flat1[:,100:350],axis=1)
#avg2 = np.average(flat2[:,80:250],axis=1)
#avg3 = np.average(flat3[:,80:250],axis=1)
#avg4 = np.average(flat4[:,80:250],axis=1)
#avg5 = np.average(flat5[:,80:250],axis=1)
#tavg = np.average(np.vstack((avg1, avg2, avg3, avg4, avg5)), axis=0)
#np.savetxt('total_average_1D.txt', tavg)

#%%
#flat = np.loadtxt('C:/Users/JarvanXIV/Documents/NotRandomFiles/Projects/VisionOptics/'+
#                  'forth_uoc/javas/PZT Interferometer/results/5_polymers/SZ3070_rpm5000_1measurement/flat1.txt')
#plt.figure()
#plt.imshow(flat)
#
#roi = flat[135:169,230:297]
#n = np.arange(np.shape(roi)[0])
#filt = gaussian_filter(roi, 11)
#plt.figure()
#plt.plot(n,roi[:,np.shape(roi)[1]//2],n,filt[:,np.shape(roi)[1]//2])
#rough = filt - roi
#np.std(rough)
















if __name__ == "__main__":
#    run(pp=False)
    run(pp=True)