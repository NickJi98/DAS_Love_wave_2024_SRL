#!/usr/bin/env python3

"""
Functions for plotting cross-correlation profiles (and channel pair properties)

Author: Qing Ji
"""

# Load python packages
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib import cm
from matplotlib.colors import ListedColormap

from my_func.param_global import dt, tlen, Npts
from my_func.preprocess import normfunc, get_amps


# Plot juxtaposed profiles (Figure 4-6)
def plot_two_profiles(ind_pair1, ind_pair2, gather1, gather2, 
                      kind='profile', case=1):
    
    # Reference velocity for Rayleigh and Love waves
    # For plotting time indicators
    ref_vel_R = 278.25
    ref_vel_L = 170.75
    
    # Concatenate two profiles
    plot_gathers = np.concatenate((gather1, gather2), axis=0)
    plot_pairs = pd.concat([ind_pair1, ind_pair2])
    
    # Timestamps
    timestamp = np.linspace(-tlen/2, tlen/2, num=Npts, endpoint=True)
    
    # Non-dimensional x-axis
    N1, N2 = len(ind_pair1), len(ind_pair2)
    mid_line = 1
    x1_data = np.linspace(0, mid_line, N1+1)[:-1]
    x2_data = np.linspace(mid_line, 2*mid_line, N2+1)[1:]
    x_data = np.concatenate((x1_data, x2_data))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 12),
                                        gridspec_kw={'height_ratios': [1, 1, 2.5]})
    
    # Panel 1: Channel pair offsets
    ax1.plot(x_data, plot_pairs['Distance'], 'k-')
    ax1.axvline(mid_line, linestyle='dashed', color='k', linewidth=2)
    ax1.set_ylabel('Offset (m)')
    secax1 = ax1.secondary_yaxis('right')
    secax1.set_ylabel('Offset (m)')
    
    # Panel 2: Angular response
    ax2_2 = ax2.twinx()
    obj_R = ax2.scatter(x1_data, ind_pair1['Rayleigh'], s=10, c='k', marker='^')
    ax2_2.scatter(x2_data, ind_pair2['Rayleigh'], s=10, c='k', marker='^')
    
    obj_L_p, obj_L_m = _scatter_sign(x1_data, ind_pair1['Love'], 
                                       ax=ax2, s=10, marker='s')
    _scatter_sign(x2_data, ind_pair2['Love'], 
                  ax=ax2_2, s=10, marker='s')
    ax2.axvline(mid_line, linestyle='dashed', color='k', linewidth=2)
    
    y1_limit, y2_limit = _get_response_ylims(case)
    ax2.set_ylim(y1_limit)
    ax2_2.set_ylim(y2_limit)
    ax2.set_ylabel('Angular Response')
    ax2_2.set_ylabel('Angular Response')
    ax2.legend([(obj_L_m, obj_L_p), obj_R], ['Love (-/+)', 'Rayleigh'], 
               loc='lower left', markerscale=2, bbox_to_anchor=(0.1,0), 
               handler_map={tuple: HandlerTuple(ndivide=None)})
    
    # Panel 3: Profile / Envelope
    if kind == 'profile':
        clim = 2e-1
        ax3.pcolormesh(x_data, timestamp, np.flip(plot_gathers.T, axis=0),
                       vmin=-clim, vmax=clim, cmap='seismic', shading='gouraud')
        ax3.axvline(mid_line, linestyle='dashed', color='k', linewidth=2)
        time_indicator_color = 'k'
        
    elif kind == 'envelope':
        clim = 1
        plot_envs = np.abs(signal.hilbert(plot_gathers, axis=1))
        plot_envs = (normfunc(plot_envs.T)).T
        ax3.pcolormesh(x_data, timestamp, np.flip(plot_envs.T, axis=0), 
                       vmin=0, vmax=clim, cmap='inferno', shading='gouraud')
        ax3.axvline(mid_line, linestyle='dashed', color='w', linewidth=2)
        time_indicator_color = 'lime'
        
    ax3.set_xlim([0, 2*mid_line])
    ax3.set_ylim([-6, 6])
    ax3.set_ylabel('Lag Time (s)')
    secax3 = ax3.secondary_yaxis('right')
    secax3.set_ylabel('Lag Time (s)')
    
    # Labels for channel pair index
    x1_index, x2_index = _get_xlabels(N1, N2, case)

    ax3.set_xticks(np.concatenate((x1_data[x1_index-1], x2_data[x2_index-1])))
    ax3.set_xticklabels(np.concatenate((x1_index, x2_index)))
    ax3.set_xlabel('Channel Pair Index')

    # Time indicators
    x_sparse = np.linspace(0, 2*mid_line, 20)
    interp_offset = interp1d(x_data, plot_pairs['Distance'])
    offset_sparse = interp_offset(x_sparse)
    ref_line_size = 10

    ref_time_L = offset_sparse / ref_vel_L
    ax3.scatter(x_sparse, ref_time_L, 
                s=ref_line_size, c=time_indicator_color, marker='s')
    ax3.scatter(x_sparse, -ref_time_L, 
                s=ref_line_size, c=time_indicator_color, marker='s')

    ref_time_R = offset_sparse / ref_vel_R
    ax3.scatter(x_sparse, ref_time_R, 
                s=ref_line_size, c=time_indicator_color, marker='s')
    ax3.scatter(x_sparse, -ref_time_R, 
                s=ref_line_size, c=time_indicator_color, marker='s')
    
    fig.tight_layout(pad=1.0)
    
    
# Get channel pair index labels
def _get_xlabels(N1, N2, case=1):
    
    if case == 1:
        x1_index = np.array([5, 25, 50, N1])
        x2_index = np.array([20, 50, 100, N2])
        
    elif case in [2, 3]:
        x1_index = np.array([5, 20, 40, N1])
        x2_index = np.array([5, 20, 40, N2])
        
    else:
        x1_index = np.rint(np.linspace(1, N1, 4, endpoint=True)).astype(int)
        x2_index = np.rint(np.linspace(1, N2, 4, endpoint=True)).astype(int)
    
    return x1_index, x2_index


# Get response plotting range
def _get_response_ylims(case=1):
    
    if case == 1:
        y1_limit = [-0.05, 1.05]
        y2_limit = [-0.0125, 0.2625]
        
    elif case in [2,3]:
        y1_limit = [0.1, 0.3]
        y2_limit = [0.1, 0.3]
        
    else:
        y1_limit = [0.0, 1.0]
        y2_limit = [0.0, 1.0]
                    
    return y1_limit, y2_limit


# Scatter plot (magnitude) with different colors for +/- values
def _scatter_sign(x, y, colors=['r', 'b'], ax=None, **kwargs):
    
    if ax is None:
        obj_p = plt.scatter(x[y>=0], y[y>=0], c=colors[0], **kwargs)
        obj_m = plt.scatter(x[y<0], -y[y<0], c=colors[1], **kwargs)
        
    else:
        obj_p = ax.scatter(x[y>=0], y[y>=0], c=colors[0], **kwargs)
        obj_m = ax.scatter(x[y<0], -y[y<0], c=colors[1], **kwargs)
        
    return obj_p, obj_m


# Plot single profile
def plot_profile(gather, ax=None):
    
    # Number of traces
    N = gather.shape[0]
    
    # Timestamps
    timestamp = np.linspace(-tlen/2, tlen/2, num=Npts, endpoint=True)
    
    clim = 2e-1
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.pcolormesh(np.arange(N), timestamp, np.flip(gather.T, axis=0), 
                      vmin=-clim, vmax=clim, cmap='seismic', shading='gouraud')
        ax.set_xlim([0, N-1])
        ax.set_ylim([-6, 6])
        ax.set_xlabel('Channel Pair Index')
        ax.set_ylabel('Lag Time (s)')   
        return fig, ax
    
    else:
        ax.pcolormesh(np.arange(N), timestamp, np.flip(gather.T, axis=0), 
                      vmin=-clim, vmax=clim, cmap='seismic', shading='gouraud')
        ax.set_xlim([0, N-1])
        ax.set_ylim([-6, 6])
        ax.set_xlabel('Channel Pair Index')
        ax.set_ylabel('Lag Time (s)')
        return ax


# Plot amplitude ratio 
def plot_amp_ratio(gather, ind_pair, case=1):
    
    # Reference velocities
    vref = np.array([140, 230, 350])
    
    # Obtain amplitudes (AL and AR)
    amp_L, amp_R = get_amps(gather, ind_pair)
    
    # Order by Love amplitudes (AL)
    order_L = np.flip(np.argsort(amp_L))
    
    # Create x-axis & Timestamps
    N = len(amp_L)
    x_data = np.arange(1, N+1)
    timestamp = np.linspace(-tlen/2, tlen/2, num=Npts, endpoint=True)
    
    # Envelope of profile
    envelope = np.abs(signal.hilbert(gather, axis=1))
    
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8, 10),
                                   gridspec_kw={'height_ratios': [1.4, 1.1]})
    ax1.pcolormesh(x_data, timestamp, np.flip(envelope[order_L,:].T, axis=0), 
                   vmin=0, vmax=0.1, cmap='inferno', shading='gouraud')
    ax1.plot(x_data, ind_pair['Distance'].iloc[order_L]/vref[0], 'w--')
    ax1.plot(x_data, ind_pair['Distance'].iloc[order_L]/vref[1], 'w--')
    ax1.plot(x_data, ind_pair['Distance'].iloc[order_L]/vref[1], 'w--')
    ax1.plot(x_data, ind_pair['Distance'].iloc[order_L]/vref[2], 'w--')
    ax1.set_ylim([0, 6])
    ax1.set_ylabel('Lag Time [s]')
    ax1.get_xaxis().set_visible(False)
    
    ax2.scatter(x_data, (amp_L/amp_R)[order_L], s=10, c='k', marker='s', 
                zorder=10, label='Obs.')
    ax2.scatter(x_data, (ind_pair['Love']/ind_pair['Rayleigh']).to_numpy()[order_L], 
               s=10, c='r', marker='s', alpha=0.7, edgecolors=None, label='Theory')
    ax2.set_ylabel('$|A_L/A_R|$')
    ax2.set_ylim(0.4, 3.3)
    ax2.set_xlim([0, N+1])
    ax2.legend(ncol=2, columnspacing=0.05, handletextpad=0.01, 
               markerscale=2, loc='upper right')
    
    if case == 1:
        x_index = np.array([1, 50, 100, N])
    elif case in [2, 3]:
        x_index = np.array([1, 20, 40, N])
    else:
        x_index = np.rint(np.linspace(1, N, 4, endpoint=True)).astype(int)
    
    ax2.set_xticks(x_index)
    ax2.set_xlabel('Channel Pair Index, Re-sorted by $|A_L|$')
    fig.tight_layout(pad=1.0)
    
    
# Plot channel pair properties (Figure 3)
def plot_channel_info(xcf_info, index=1):
    
    if index == 1:
        prop_mat = meta2tri(xcf_info, 'Distance')
        x_label = 'Distance (m)'
        im_kwargs = {'vmin': 0, 'vmax': 1250}
        
        cmap = cm.get_cmap("jet").copy()
        cmap.set_bad('w')
        cmap = ListedColormap(cmap(np.linspace(0, 1, 256)))
        
    elif index == 2:
        prop_mat = meta2tri(xcf_info, 'Rayleigh')
        x_label = 'Rayleigh response $(A_R)$'
        im_kwargs = {'vmin': 0.0, 'vmax': 1.0}
        cmap = cm.get_cmap("seismic").copy()
        cmap = ListedColormap(cmap(np.linspace(0.5, 1, 256)))
        
    elif index == 3:
        prop_mat = meta2tri(xcf_info, 'Love')
        x_label = 'Love response $(A_L)$'
        im_kwargs = {'vmin': -0.25, 'vmax': 0.25}
        cmap = cm.get_cmap("seismic").copy()
        
    else:
        raise Exception("Index should be one of [1, 2, 3]!")
    
    # Plot
    fig, ax = plt.subplots(figsize=(7.2, 9))
    im = ax.imshow(prop_mat, cmap=cmap, interpolation='none', **im_kwargs)
    ax.set_xlabel('Channel number', labelpad=10)
    ax.set_ylabel('Channel number', labelpad=20, rotation=270)
    ax.xaxis.set_ticks_position('top') 
    ax.xaxis.set_label_position('top') 
    ax.yaxis.set_ticks_position('right') 
    ax.yaxis.set_label_position('right')
    ax.grid()
    ax.set_facecolor([0, 0, 0, 0.08])
    cb = fig.colorbar(im, orientation='horizontal', pad=0.02)
    cb.ax.set_xlabel(x_label)
    
    # Two corners
    ax.vlines(x=70, ymin=0, ymax=70, colors='k', linestyles='--', linewidth=2)
    ax.vlines(x=130, ymin=0, ymax=130, colors='k', linestyles='--', linewidth=2)
    ax.hlines(y=70, xmin=70, xmax=200, colors='k', linestyles='--', linewidth=2)
    ax.hlines(y=130, xmin=130, xmax=200, colors='k', linestyles='--', linewidth=2)
    
    ax.set_xlim(0, 200)
    ax.set_ylim(200, 0)
    fig.tight_layout(pad=1.0)
    return fig, ax


# Read channel pair property into triangular matrix
def meta2tri(df, col):
    
    # Obtain indices and quantity
    ind1 = np.array(df['Ind1'])
    ind2 = np.array(df['Ind2'])
    prop = np.array(df[col])
    
    # Create empty matrix
    N = np.max([ind1, ind2])
    mat = np.zeros((N+1, N+1))
    
    # Fill the matrix
    mat[ind1, ind2] = prop
    
    # Mask out lower triangle
    mat = np.ma.array(mat, mask=np.tri(N+1))
    
    return mat