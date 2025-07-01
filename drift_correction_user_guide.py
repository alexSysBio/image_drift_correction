# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 13:01:17 2025

@author:  Alexandros Papagiannakis, Stanford University, 2025
"""

import remove_image_drift as drft
      
ndtwo_path = ".../images.nd2"

# Open the fifth XY position and the Phase channel form the .nd2 file
images_dict = drft.load_image_arrays(ndtwo_path, 5, 'Phase')

# Calculate image drift
phase_drift = drft.generate_drift_sequence(images_dict, resolution=10, hard_threshold=27500, mask_show=False)

# Get the cumulative drift and smooth
# Univariate spline smoothing 
cor_images_dict, cum_phase_drift = drft.apply_phase_correction(images_dict, phase_drift, 'univar', (2,50)) # kappa and s parameters 
# or polynomial fit smoothing
cor_images_dict, cum_phase_drift = drft.apply_phase_correction(images_dict, phase_drift, 'poly', 6) # polynomial degree
# or moving average smoothing
cor_images_dict, cum_phase_drift = drft.apply_phase_correction(images_dict, phase_drift, 'rolling', 3) # rolling window
# or no smoothing
cor_images_dict, cum_phase_drift = drft.apply_phase_correction(images_dict, phase_drift, 'none', 'none') 


# plot the corrected frames
crop_pad = (300, 550, 1020, 1270) # minx, miny, maxx, maxy
scale = 0.066  # um/px
time_stamp_pos = (40,60) # in pixels
scale_bar_pos = (580,660) # in pixels
time_interval= 4 # min
fonts_sizes = 22
fonts_color = 'white'
save_path = "...\Drift_corrected_images"


drft.create_movies(cor_images_dict, crop_pad, time_interval, scale, 
                   time_stamp_pos, scale_bar_pos, fonts_sizes, fonts_color, save_path, show=False)