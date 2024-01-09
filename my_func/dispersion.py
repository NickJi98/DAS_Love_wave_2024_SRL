#!/usr/bin/env python3

"""
Functions for dispersion measurement

Author: Qing Ji (FK method is from Ariel Lellouch)
"""


# Load python packages
import numpy as np
import math, scipy
import matplotlib.pyplot as plt
from tqdm import tqdm

from my_func.preprocess import normfunc
from my_func.xwt import cwt
from my_func.param_global import dt
from my_func.param_dispmaps import get_measure_freqs

from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
from unwrap import unwrap


### FK method from Ariel Lellouch ###

# FK transform
def map_fk(gather, offset_pair, dx=5, dt=1./50., pos=True):
    
    half_t = gather.shape[1]//2
    d_off = dx
    
    if pos:
        interp_offs, interp_data = map_to_offset(gather[:, :half_t], offset_pair, d_off)
    else:
        interp_offs, interp_data = map_to_offset(gather[:, -1:half_t:-1], offset_pair, d_off)
    
    return scipy.fft.fftshift(scipy.fft.fft2(interp_data))
  

# Convert k to phase velocity  
def map_fv(data, dx, dt, freqs, vels):
    
    (nch, nt) = np.shape(data)
    nscanv = np.size(vels)
    nf = 2**(1+math.ceil(math.log(nt, 2)))
    nk = 2**(1+math.ceil(math.log(nch, 2)))

    fft_f = np.arange(-nf/2, nf/2)/nf/dt
    fft_k = np.arange(-nk/2, nk/2)/nk/dx

    fk_res = scipy.fft.fftshift(scipy.fft.fft2(data, s=[nk, nf]))
    fk_res = np.absolute(fk_res)

    # Transpose needed for interp2 definition
    interp_fun = interp2d(fft_k, fft_f, fk_res.T, kind='cubic')

    # interp_fun = scipy.interpolate.RectBivariateSpline(fft_k, fft_f, fk_res)  # Supposed to be faster. So far isn't

    ones_arr = np.ones(shape=(nscanv,))
    fv_map = np.zeros(shape=(len(freqs), len(vels)), dtype=np.float32)
    for ind, fr in enumerate(freqs):
        fv_map[ind, :] = np.squeeze(interp_fun(np.divide(ones_arr*fr, vels), fr))

    return fv_map.T


# Interpolate traces to uniform spacing offset grid defined by d_off
def map_to_offset(data, offsets, d_off):
    n_chan, nt = np.shape(data)

    max_off = np.amax(offsets)
    min_off = np.amin(offsets)
    interp_offs = d_off * np.arange(int(np.floor(min_off/d_off))+1, int(np.ceil(max_off/d_off)))
    n_int_offs = np.size(interp_offs)

    sorting_ord = np.argsort(offsets)
    sorted_off = offsets[sorting_ord]
    sorted_data = data[sorting_ord, :]

    interpolator = scipy.interpolate.interp1d(sorted_off, np.arange(0, n_chan))
    interp_chans = interpolator(interp_offs)

    interp_data = np.zeros(shape=(n_int_offs, nt))
    interp_counter = np.zeros(shape=(n_int_offs,))

    for i in range(n_int_offs):
        interp_val = interp_chans[i] - np.floor(interp_chans[i])
        flr_chan = int(np.floor(interp_chans[i]))
        if interp_val == 0.0:
            interp_data[i, :] += sorted_data[flr_chan, :]
            interp_counter[i] += 1.0
        else:
            interp_data[i, :] = (1.0-interp_val) * sorted_data[flr_chan, :] + interp_val * sorted_data[flr_chan+1, :]
            interp_counter[i] += 1.0

    for i in range(n_int_offs):
        interp_data[i, :] /= interp_counter[i]

    return interp_offs, interp_data


# Calculate phase velocity dispersion
def calc_vph_disp(gather, offset_pair, freqs, vels, dt=1./50., pos=True):

    half_t = gather.shape[1]//2
    d_off = 5
    
    if pos:
        interp_offs, interp_data = map_to_offset(gather[:, :half_t], offset_pair, d_off)
    else:
        interp_offs, interp_data = map_to_offset(gather[:, -1:half_t:-1], offset_pair, d_off)

    fv_map_arr = map_fv(interp_data, d_off, dt, freqs, vels)
    
    return normfunc(fv_map_arr)


### Obtain phase velocity dispersion map by averaging over two branches

def get_ave_vph_map(gather, offset, freqs, vels, dt=1./50.):
    
    fv_map_p = calc_vph_disp(gather, offset, freqs, vels, dt=dt, pos=True)
    fv_map_n = calc_vph_disp(gather, offset, freqs, vels, dt=dt, pos=False)
    
    return (fv_map_p + fv_map_n) / 2.


### Obtain group velocity dispersion map by wavelet transform

def stack_psd(gather, ind_pair, time_samp, vg_samp, freq_range, Nfreqs, 
              wave_type=None, smooth=True):
    
    interp_type = 'linear'
    Nvg = len(vg_samp)
    
    # Distance range
    print('Maximum distance : %.2f m' %np.max(ind_pair['Distance']))
    print('Minimum distance : %.2f m' %np.min(ind_pair['Distance']))
    print('Average distance : %.2f m' %np.average(ind_pair['Distance']))
    print('Median distance : %.2f m' %np.median(ind_pair['Distance']))
    
    stack_cfs_p = np.zeros((Nfreqs, Nvg))
    stack_cfs_n = np.zeros((Nfreqs, Nvg))
    coi_p = np.zeros(Nvg)
    coi_n = np.zeros(Nvg)
    
    for cur_ind in tqdm(range(ind_pair.shape[0])):
    
        # Wavelet analysis
        trace = gather[cur_ind, ::-1]
        _, cc_cfs, freqs, coi = cwt(trace, 1/dt, ns=3, nt=0.25, vpo=12, 
                                    freqmin=freq_range[1], # To make output 'freqs' in incresing order
                                    freqmax=freq_range[0], 
                                    nptsfreq=Nfreqs, smooth=smooth)
        
        # Positve branch
        mask_p = (time_samp > 0)
        cc_cfs_p = cc_cfs[:, mask_p]
        cc_cfs_p = cc_cfs_p / np.max(cc_cfs_p)
        
        # Negative branch
        mask_n = (time_samp < 0)
        cc_cfs_n = cc_cfs[:, mask_n]
        cc_cfs_n = cc_cfs_n / np.max(cc_cfs_n)
        
        # Interpolation to reference velocity samples
        fp = interp1d(ind_pair.iloc[cur_ind]['Distance']/time_samp[mask_p], cc_cfs_p, 
                      kind=interp_type, bounds_error=False, fill_value=0)
        if wave_type is None:
            stack_cfs_p = stack_cfs_p + fp(vg_samp)
        else:
            stack_cfs_p = stack_cfs_p + fp(vg_samp) * np.abs(ind_pair.iloc[cur_ind][wave_type])
        
        fn = interp1d(-ind_pair.iloc[cur_ind]['Distance']/time_samp[mask_n], cc_cfs_n, 
                      kind=interp_type, bounds_error=False, fill_value=0)
        if wave_type is None:
            stack_cfs_n = stack_cfs_n + fn(vg_samp)
        else:
            stack_cfs_n = stack_cfs_n + fn(vg_samp) * np.abs(ind_pair.iloc[cur_ind][wave_type])
        
        # Cone of instability (Min. frequency resolved at each time)
        fp_coi = interp1d(ind_pair.iloc[cur_ind]['Distance']/time_samp[mask_p], coi[mask_p], 
                          kind=interp_type, bounds_error=False, fill_value=0)
        coi_p = np.max(np.array([coi_p, 1/fp_coi(vg_samp)]), axis=0)
        
        fn_coi = interp1d(-ind_pair.iloc[cur_ind]['Distance']/time_samp[mask_n], coi[mask_n], 
                          kind=interp_type, bounds_error=False, fill_value=0)
        coi_n = np.max(np.array([coi_n, 1/fn_coi(vg_samp)]), axis=0)
        
    # Normalization & Flipping to fv_map convention
    stack_cfs_p = np.flip(normfunc(stack_cfs_p.T), axis=0)
    stack_cfs_n = np.flip(normfunc(stack_cfs_n.T), axis=0)
    
    return freqs, stack_cfs_p, stack_cfs_n, coi_p, coi_n


### Correction for instantaneous frequency (e.g., Bensen et al., 2007, GJI) ###

def stack_psd_correction(gather, ind_pair, time_samp, vg_samp, filter_range,
                         Nfreqs, wave_type=None):
    
    interp_type = 'linear'
    Nvg = len(vg_samp)
    
    # Distance range
    print('Maximum distance : %.2f m' %np.max(ind_pair['Distance']))
    print('Minimum distance : %.2f m' %np.min(ind_pair['Distance']))
    print('Average distance : %.2f m' %np.average(ind_pair['Distance']))
    print('Median distance : %.2f m' %np.median(ind_pair['Distance']))
    
    stack_cfs_p = np.zeros((Nfreqs, Nvg))
    stack_cfs_n = np.zeros((Nfreqs, Nvg))
    coi_p = np.zeros(Nvg)
    coi_n = np.zeros(Nvg)
    freqs = np.linspace(filter_range[0], filter_range[1], Nfreqs)
    
    # Extend frequency axis to account for potential instantaneous correction
    _freq_range = [filter_range[0], filter_range[1]]
    _Nfreqs = np.ceil(Nfreqs*1.2).astype(int)
    
    for cur_ind in tqdm(range(ind_pair.shape[0])):
    
        # Wavelet analysis
        trace = gather[cur_ind, ::-1]
        cc_cwt, cc_cfs, _freqs, coi = cwt(trace, 1/dt, ns=3, nt=0.25, vpo=12, 
                                          freqmin=_freq_range[1], # To make output '_freqs' in incresing order
                                          freqmax=_freq_range[0], 
                                          nptsfreq=_Nfreqs, smooth=False)
        
        # Positve branch
        mask_p = (time_samp > 0)
        cc_cfs_p = cc_cfs[:, mask_p]
        cc_cfs_p = cc_cfs_p / np.max(cc_cfs_p)
        
        # Negative branch
        mask_n = (time_samp < 0)
        cc_cfs_n = cc_cfs[:, mask_n]
        cc_cfs_n = cc_cfs_n / np.max(cc_cfs_n)
        
        ### Ridgeline (group arrival time)
        tg_p_ind = np.nanargmax(cc_cfs_p, axis=1)
        tg_n_ind = np.nanargmax(cc_cfs_n, axis=1)

        ### Correction for instantaneous frequency
        cwt_phase = unwrap(np.angle(cc_cwt))
        phi1_p = (cwt_phase[:,mask_p])[np.arange(0, _Nfreqs), tg_p_ind-1]
        phi2_p = (cwt_phase[:,mask_p])[np.arange(0, _Nfreqs), tg_p_ind]
        freqs_ins_p = (phi2_p - phi1_p) / dt / (2*np.pi)
        idx_p = np.argsort(freqs_ins_p)
        freqs_ins_p = freqs_ins_p[idx_p]

        phi1_n = (cwt_phase[:,mask_n])[np.arange(0, _Nfreqs), tg_n_ind-1]
        phi2_n = (cwt_phase[:,mask_n])[np.arange(0, _Nfreqs), tg_n_ind]
        freqs_ins_n = (phi2_n - phi1_n) / dt / (2*np.pi)
        idx_n = np.argsort(freqs_ins_n)
        freqs_ins_n = freqs_ins_n[idx_n]
        
        # Interpolation to reference velocity samples
        if not np.all(np.diff(freqs_ins_p) > 0.0):
            print('Trace %d positive branch skipped.' %cur_ind)
        else: 
            fp = RectBivariateSpline(freqs_ins_p, 
                                     np.flip(ind_pair.iloc[cur_ind]['Distance']/time_samp[mask_p]), 
                                     np.flip(cc_cfs_p[idx_p, :], axis=1))
            if wave_type is None:
                stack_cfs_p = stack_cfs_p + fp(freqs, vg_samp)
            else:
                stack_cfs_p = stack_cfs_p + fp(freqs, vg_samp) * np.abs(ind_pair.iloc[cur_ind][wave_type])
        
        if not np.all(np.diff(freqs_ins_n) > 0.0):
            print('Trace %d negative branch skipped.' %cur_ind)
        else: 
            fn = RectBivariateSpline(freqs_ins_n, 
                                     -ind_pair.iloc[cur_ind]['Distance']/time_samp[mask_n], 
                                     cc_cfs_n[idx_n, :])
            if wave_type is None:
                stack_cfs_n = stack_cfs_n + fn(freqs, vg_samp)
            else:
                stack_cfs_n = stack_cfs_n + fn(freqs, vg_samp) * np.abs(ind_pair.iloc[cur_ind][wave_type])
        
        # Cone of instability (Min. frequency resolved at each time)
        fp_coi = interp1d(ind_pair.iloc[cur_ind]['Distance']/time_samp[mask_p], coi[mask_p], 
                          kind=interp_type, bounds_error=False, fill_value=0)
        coi_p = np.max(np.array([coi_p, 1/fp_coi(vg_samp)]), axis=0)
        
        fn_coi = interp1d(-ind_pair.iloc[cur_ind]['Distance']/time_samp[mask_n], coi[mask_n], 
                          kind=interp_type, bounds_error=False, fill_value=0)
        coi_n = np.max(np.array([coi_n, 1/fn_coi(vg_samp)]), axis=0)
        
    # Normalization & Flipping to fv_map convention
    stack_cfs_p = np.flip(normfunc(stack_cfs_p.T), axis=0)
    stack_cfs_n = np.flip(normfunc(stack_cfs_n.T), axis=0)
    
    return freqs, stack_cfs_p, stack_cfs_n, coi_p, coi_n


### Extract dispersion map ridgeline, optionally based on a reference curve ###

def extract_ridge(freq, vel, fv_map, func_vel=None, sigma=25):
    
    # Velocity unit is m/s: vel, func_vel, sigma
    # Shape of fv_map: (Nvel, Nfreq)
    
    # No reference curve
    if func_vel is None:
        return vel[np.argmax(fv_map, axis=0)]
    
    # Extract ridgeline around the given curve
    else:
        # Reference velocity
        vel_ref = func_vel(freq)
        
        # Mask dispersion map
        vel_2d = np.tile(vel[::-1], (len(freq), 1)).T
        mask = (vel_2d > (vel_ref - sigma)) & (vel_2d < (vel_ref + sigma))
        mask_fv_map = np.ma.masked_array(fv_map, mask=~mask)
        
        # Dispersion curve
        return vel[np.argmax(mask_fv_map, axis=0)]
    

### Measure dispersion curves with uncertainties (optionally based on a reference curve) ###

def measure_dispersion(freq, vel, fv_map, freq_samples, threshold=0.9, 
                       func_vel=None, vel_offset=20):
    
    if freq[1] - freq[0] < 0:
        raise Exception('Frequency points should be of increasing order!')
    
    dv = vel[1] - vel[0]
    if dv < 0:
        raise Exception('Velocity points should be of increasing order!')
        
    # Frequency samples
    ind_freq = np.array([np.argmin(np.abs(freq - freq_samp)) 
                         for freq_samp in freq_samples])
    
    # Peak point
    if func_vel is None:
        ind_vel_mean = np.argmax(fv_map[:, ind_freq], axis=0)
        vel_mean = vel[ind_vel_mean]
        amp_peak = fv_map[ind_vel_mean, ind_freq]
        
    else:
        # Reference velocity
        vel_ref = func_vel(freq)
        
        # Mask dispersion map
        vel_2d = np.tile(vel, (len(freq), 1)).T
        mask = (vel_2d > (vel_ref - vel_offset)) & (vel_2d < (vel_ref + vel_offset))
        mask_fv_map = np.ma.masked_array(fv_map, mask=~mask)
        
        ind_vel_mean = np.argmax(mask_fv_map[:, ind_freq], axis=0)
        vel_mean = vel[ind_vel_mean]
        amp_peak = fv_map[ind_vel_mean, ind_freq]
        
    # Uncertainty (default: above 0.9 * max)
    ind_min = np.array([ind_v - np.argmax(fv_map[ind_v::-1, ind_f] < amp*threshold) 
                         for (ind_v, ind_f, amp) in zip(ind_vel_mean, ind_freq, amp_peak)])
    ind_max = np.array([ind_v + np.argmax(fv_map[ind_v:, ind_f] < amp*threshold) 
                         for (ind_v, ind_f, amp) in zip(ind_vel_mean, ind_freq, amp_peak)])

    # Output measurements as a dict
    dispcurve = {'freqs': freq[ind_freq], 'vels': vel_mean, 
                 'upper': vel[ind_max], 'lower': vel[ind_min]}
    return dispcurve


def get_dispcurve(freqs, vels, dispmap, index):
    
    # Reference Love wave group velocity
    if index == 1:
        ref_vg_freq = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        ref_vg = np.array([225, 180, 160, 150, 150, 150])
        func_vel = interp1d(ref_vg_freq, ref_vg, fill_value="extrapolate")
        
    # Reference Rayleigh wave phase velocity (overtone)
    # Only to avoid the influence of fundamental mode around 6 Hz
    elif index == 4:
        ref_vph_freq = np.array([2.5, 3.0, 4.0, 4.6, 6.0])
        ref_vph = np.array([500, 450, 380, 340, 310])
        func_vel = interp1d(ref_vph_freq, ref_vph, fill_value="extrapolate")
    
    else:
        func_vel = None
    
    # Dispersion curve
    dispcurve = measure_dispersion(freqs, vels, np.flip(dispmap, axis=0), 
                                   get_measure_freqs(index), func_vel=func_vel)
    return dispcurve


### Plot dispersion map ###

def plot_dispmap(fv_map, extent, vel_type='phase'):
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(fv_map, aspect='auto', vmin=0, vmax=1, cmap='jet', 
              interpolation='bicubic', origin='upper', extent=extent)
    ax.grid(c='k')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('%s velocity (m/s)' %vel_type.capitalize())
    ax.set_xlim([1, 7])
    ax.set_ylim([100, 500])
    
    return fig, ax


### Plot dispersion curve ###
def plot_dispcurve(dispcurve, vel_type='phase', 
                   xaxis='freq', ax=None, **kwargs):
    
    # X-axis: Frequency or Period
    if xaxis == 'period':
        x_data = 1 / dispcurve['freqs']
        x_label = 'Period (s)'
        x_limit = [1/7, 1]
        
    elif xaxis == 'freq':
        x_data = dispcurve['freqs']
        x_label = 'Frequency (Hz)'
        x_limit = [1, 7]
        
    else:
        raise Exception("Argument xaxis should be 'freq' or 'period'!")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.errorbar(x_data, (dispcurve['upper']+dispcurve['lower'])/2, 
                    yerr=np.abs(np.stack((dispcurve['lower'], dispcurve['upper']))
                                - (dispcurve['upper']+dispcurve['lower'])/2),
                    **kwargs)
        ax.grid(c='k')
        ax.set_xlabel(x_label)
        ax.set_ylabel('%s velocity (m/s)' %vel_type.capitalize())
        ax.set_xlim(x_limit)
        ax.set_ylim([100, 500])
        return fig, ax
    
    else:
        ax.errorbar(x_data, (dispcurve['upper']+dispcurve['lower'])/2, 
                    yerr=np.abs(np.stack((dispcurve['lower'], dispcurve['upper']))
                                - (dispcurve['upper']+dispcurve['lower'])/2),
                    **kwargs)
        ax.grid(c='k')
        ax.set_xlabel(x_label)
        ax.set_ylabel('%s velocity (m/s)' %vel_type.capitalize())
        ax.set_xlim(x_limit)
        ax.set_ylim([100, 500])
        return ax
    