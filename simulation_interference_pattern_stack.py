# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 19:32:38 2019

@author: Evangelos Tzardis
"""


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os
from visualizations import display_imageStack, plot_3d

#%%
"""
DEFINITIONS
""" 
def gaussian_2d(x_axis, y_axis, x_points, y_points, mu_s, sigma_s):
    x, y = np.meshgrid(np.linspace(x_axis[0],x_axis[1],x_points), np.linspace(y_axis[0],y_axis[1],y_points))
    d = np.sqrt(x*x+y*y)
    return np.exp(-( (d-mu_s)**2 / ( 2.0 * sigma_s**2 ) ) )

def gaussian_1d(x,x0,sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))

# X,Y AXES IN um
# Z AXIS IN nm

#%%
"""
GENERATE 2D SAMPLE
"""
# e.g a gaussian bell

#-----------------------------
# transverse plane
x_axis = [-500.0,500.0] # um
y_axis = [-500.0,500.0] # um
x_points = 101 # linspace points
y_points = 101

# sample parameters
mu_s, sigma_s = 0.0, 2000.0 # um, um | gaussian parameters

# z-axis scaling factor
scale_z = 10000.0 # nm
#-----------------------------
sample_name = 'gaussbell'
# 2D gaussian bell
sample = gaussian_2d(x_axis, y_axis, x_points, y_points, mu_s, sigma_s)

#-----------------------------
 
sample = sample*scale_z
#%%

"""
GENERATE IMAGE STACK OF INTERFERENCE PATTERNS WITH Z-AXIS SCANNING
"""
# I_tot = I_amb + gamma*cos(2*pi/lamda*OPD)
# OPD = 2*elevation

lamda = 55.0 # nm | wavelength or mean wavelength of illuminating source
intensity_scale = 20000.0 # assuming 16-bit depth per pixel | max = 65535
bckg_percent = 0.05
sample_percent = 0.05
ref_percent = 0.9
#------------------------------------------------------------------------------
# Camera intensity variation across the field
mu_field = 0.0
sigma_field = min(np.amax(x_axis)-np.amin(x_axis), np.amax(y_axis)-np.amin(y_axis))

field_intensity = gaussian_2d(x_axis, y_axis, x_points, y_points, mu_field, sigma_field)
#------------------------------------------------------------------------------
# Axis and sigma of coherence function or gamma
sigma_g = 5000.0 # nm
scan_length = 500 # number of scans in z-axis

#scan_range = [np.amin(sample), np.amax(sample)]
scan_range = [0.0, 15000.0] # based on knowledge of the sample range of heights
# simulated reference arm movement reduces OPD with each step made
# Linear movement of reference arm
#scan_axis = np.linspace(scan_range[1],scan_range[0],scan_length)
# Non-linear movement of reference arm
a = 1.0
scan_axis = np.linspace(scan_range[1],scan_range[0],scan_length)**a
#------------------------------------------------------------------------------
# Ambient intensity modulation across X,Y axes
# Assume ambient intensity modulation is the same across all the z-planes
bckg_scale = bckg_percent*intensity_scale
I_b = bckg_scale*field_intensity
#I_b = np.repeat(I_amb[np.newaxis, :, :], OPD_length, axis=0)
#------------------------------------------------------------------------------
# Sample intensity
sample_scale = sample_percent*intensity_scale
I_s = sample_scale*field_intensity
#------------------------------------------------------------------------------
# Reference intensity
ref_scale = ref_percent*intensity_scale
I_r = ref_scale*field_intensity
#------------------------------------------------------------------------------
# No-fringe added intensities
I_srb = I_s + I_r + I_b
#------------------------------------------------------------------------------
# Image stack of interference patterns
I_tot = np.zeros([scan_length, x_points, y_points])

for i in range(y_points):
    for j in range(x_points):
        elevation = sample[i,j]
        OPD_onScan0 = 2*elevation
        OPD_axis = 2*scan_axis - OPD_onScan0
        
        gamma = 2*np.sqrt(I_s[i,j])*np.sqrt(I_r[i,j])*gaussian_1d(2*scan_axis,OPD_onScan0,sigma_g)
        I_tot[:,i,j] = I_srb[i,j] + gamma*np.cos(2*np.pi/lamda*OPD_axis)
#------------------------------------------------------------------------------
# Additive White Gaussian Noise
mu_n = 0.0
noise_pcnt = 0.0
sigma_n = noise_pcnt*intensity_scale

I_totN = np.zeros([scan_length, x_points, y_points])

for k in range(scan_length):
    noise = np.random.normal(mu_n,sigma_n,sample.shape)
    I_totN[k] = I_tot[k] + noise

#------------------------------------------------------------------------------
##%%
## Save image stack to disk
#    
#subfolder = '\\simulation'
#stack_name = '\\' + sample_name + '_' \
#             + (str(lamda)+'nm').replace('.','_') + '_' \
#             + ('tempCohSigma'+str(sigma_g)+'nm').replace('.','_') + '_' \
#             + (str(scan_length)+'zscans') + '_' \
#             + (str(noise_pcnt).replace('.','_')+'pcnt_awgn')
#
#folder = os.path.dirname(os.path.realpath(__file__)) + subfolder + stack_name
#folder_srb = folder + '\\IsIrIb'
#if not os.path.exists(folder):
#    os.makedirs(folder)
#    
#for k in range(scan_length):
#    fn = '\\' +  str(k) + '.tiff'
#    imageio.imsave(folder + fn, I_totN[k])
#
##------------------------------------------------------------------------------
##%%
## Display image sequence
display_imageStack(I_totN)



# Random displaying of examples 1D, 2D, 3D
# ss = I_tot[100]
# plot_3d(ss, x_axis, y_axis, x_points, y_points, [np.amin(ss),np.amax(ss)])
    
# plt.plot(I_tot[:,25,25]);plt.plot(I_tot[:,6,1])
    
# plt.imshow(I_tot[11])











































