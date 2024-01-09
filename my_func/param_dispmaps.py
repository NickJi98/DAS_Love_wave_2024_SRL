#!/usr/bin/env python3

"""
Parameters for dispersion measurements (Figures 9)

Author: Qing Ji
"""

# Load python packages
from my_func.param_profiles import get_param_profile
from my_func.preprocess import select_correlations, bpfilter
import numpy as np


# Window parameters for phase velocity measurement
def get_window_params(wave_type='L'):
    
    # Love wave (Case 2, Parallel)
    if wave_type == 'L':
        window_param = {"ref_vel":160, "dv":30, "min_win":0.3, "taper":0.25}
        
    # Rayleigh wave (Case 1 & 4, Inline)
    elif wave_type =='R':
        window_param = {"ref_vel":278.25, "dv":0, "min_win":1.0, "taper":0.25}
    
    return window_param


# Get profiles for dispersion measurements
def get_disp_profile(xcf_info, xcf_list, index=1):
    
    # Bandpass filter
    filter_range = [0.5, 10]
    
    # Love, Group (Figure 9a)
    if index == 1:
        
        # Case 1, Parallel
        param_profile = get_param_profile(xcf_info, case=1)
        ind_pair = xcf_info[param_profile['mask2']]
        ind_pair = ind_pair.sort_values(**param_profile['sort2'])
        gather = select_correlations(ind_pair, xcf_list)
        gather = bpfilter(gather, filter_range[0], filter_range[1])
        
    # Love, Phase (Figure 9b)
    elif index == 2:
        
        # Case 2, Oblique
        param_profile = get_param_profile(xcf_info, case=2)
        ind_pair = xcf_info[param_profile['mask1']]
        ind_pair = ind_pair.sort_values(**param_profile['sort1'])
        gather = select_correlations(ind_pair, xcf_list)
        gather = bpfilter(gather, filter_range[0], filter_range[1])
        
    # Rayleigh, Phase, Fundamental (Figure 9c)
    elif index == 3:
        
        # Case 4, Inline (Blue segment)
        param_profile = get_param_profile(xcf_info, case=4)
        ind_pair = xcf_info[param_profile['mask1']]
        ind_pair = ind_pair.sort_values(**param_profile['sort1'])
        gather = select_correlations(ind_pair, xcf_list)
        gather = bpfilter(gather, filter_range[0], filter_range[1])
        
    # Rayleigh, Phase, Overtone (Figure 9d)
    elif index == 4:
        
        # Case 1, Inline (Green segment)
        param_profile = get_param_profile(xcf_info, case=1)
        ind_pair = xcf_info[param_profile['mask1']]
        ind_pair = ind_pair.sort_values(**param_profile['sort1'])
        gather = select_correlations(ind_pair, xcf_list)
        gather = bpfilter(gather, filter_range[0], filter_range[1])
        
    return ind_pair, gather


# Get frequency samples for measurement
def get_measure_freqs(index):
    
    # Love, Group (Figure 9a)
    if index == 1:
        return np.arange(1.5, 5.5+0.1, 0.2)
    
    # Love, Phase (Figure 9b)
    elif index == 2:
        return np.arange(1.5, 5.5+0.1, 0.2)
    
    # Rayleigh, Phase, Fundamental (Figure 9c)
    elif index == 3:
        return np.arange(2.0, 6.5, 0.2)
    
    # Rayleigh, Phase, Overtone (Figure 9d)
    elif index == 4:
        return np.arange(3.0, 6.5, 0.2)
    
    else:
        return np.arange(1.0, 7.1, 0.1)