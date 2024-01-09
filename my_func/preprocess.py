#!/usr/bin/env python3

"""
Functions for trace selection and pre-processing

Author: Qing Ji
"""

# Load python packages
import numpy as np
from scipy import signal
from obspy import Trace, Stream, UTCDateTime
from my_func.param_global import dt, tlen, Npts


# Extract cross-correlation traces
def select_correlations(ind_pair, xcf_list):

    select_gather = []
    N_select = ind_pair.shape[0]

    for i in range(N_select):
        
        tr_data = np.load(xcf_list[ind_pair['Ind1'].iloc[i]])[ind_pair['Ind2'].iloc[i]]
        tr_data = np.append(tr_data, 0.0)
        select_gather.append(tr_data)

    select_gather = np.asarray(select_gather)
    return select_gather


# Convert stream object to numpy array
def stream_to_gather(stream):
    gather = []
    
    for trace in stream:
        gather.append(np.flip(trace.data))
    
    gather = np.asarray(gather)
    return gather


# Convert numpy array object to stream
def gather_to_stream(gather, ind_pair):
    stream = Stream()
    
    for i in range(gather.shape[0]):
        
        tr = Trace(data=gather[i, ::-1])
        tr.stats.delta = dt
        tr.stats.starttime = UTCDateTime(2000, 1, 1, 0, 0, 0)
        tr.stats.distance = ind_pair.iloc[i,2]
        stream += tr

    return stream


# Normalization
def normfunc(X):
    return np.asarray([X[:,i]/(abs(X[:,i]).max()+1e-20) for i in range(X.shape[1])]).T


# Bandpass filter (Consistent with Obspy)
def bpfilter(data, bp_low, bp_high):
    # Default order: 4
    z, p, k = signal.iirfilter(4, [bp_low, bp_high], btype='band', 
                               ftype='butter', output='zpk', fs=1/dt)
    sos = signal.zpk2sos(z, p, k)
    return signal.sosfiltfilt(sos, data, axis=-1)


# Apply Gaussian window
def window_trace(gather, offset, ref_vel, dv, min_win=0.25, taper=0.25):
    
    # Timestamps
    time = np.linspace(-tlen/2, tlen/2, num=Npts, endpoint=True)
    
    win_gather = []
    for data, dist in zip(gather, offset):
        
        # Create flat window with Gaussian taper
        shift = dist / ref_vel
        win_len = dist * dv / ref_vel**2
        win_len = np.max((win_len, min_win))
        sigma = win_len * taper
        win_p = (time > shift+win_len) * (np.exp(-(time-shift-win_len)**2 / (2*sigma**2)) - 1) \
              + (time < shift-win_len) * (np.exp(-(time-shift+win_len)**2 / (2*sigma**2)) - 1) \
              + 1
        win_n = (time < -shift-win_len) * (np.exp(-(time+shift+win_len)**2 / (2*sigma**2)) - 1) \
              + (time > -shift+win_len) * (np.exp(-(time+shift-win_len)**2 / (2*sigma**2)) - 1) \
              + 1
        win_func = np.maximum(win_p, win_n)
        
        # Windowed trace
        win_gather.append(data * win_func)
        
    win_gather = np.asarray(win_gather)
    return win_gather


# Simple measurement of Love / Rayleigh amplitudes (Figure 8)
def get_amps(gather, ind_pair):
    
    # Number of traces
    N = ind_pair.shape[0]
    
    # Reference velocities
    vref = np.array([140, 230, 350])
    
    # Timestamps
    time = np.linspace(-tlen/2, tlen/2, num=Npts, endpoint=True)
    
    # Match the traces with the time axis (profile or envelope)
    # Note: Figure 8 accidentally used the profile, instead of its absolute values
    # or envelope, to measure amplitudes. However, the results are consistent, 
    # as previously I also tried to use the RMS amplitude. 
    # In addition, Figure 8 only serves as a qualitative comparison of AL/AR 
    # between observation and theory
    amp_gather = np.flip(gather, axis=1)
    
    # Use the envelope to measure amplitude
    # amp_gather = np.flip(np.abs(signal.hilbert(gather, axis=1)), axis=1)
    
    # Love wave amplitudes
    amp_L = np.nan * np.ones((N,))
    for i in range(0, N):
        dist_cur = ind_pair['Distance'].iloc[i]
        mask_cur = (time <= dist_cur/vref[0]) & (time >= dist_cur/vref[1])
        amp_L[i] = np.max(amp_gather[i][mask_cur])
        
    # Rayleigh wave amplitudes
    amp_R = np.nan * np.ones((N,))
    for i in range(0, N):
        dist_cur = ind_pair['Distance'].iloc[i]
        mask_cur = (time <= dist_cur/vref[1]) & (time >= dist_cur/vref[2])
        amp_R[i] = np.max(amp_gather[i][mask_cur])
    
    return amp_L, amp_R