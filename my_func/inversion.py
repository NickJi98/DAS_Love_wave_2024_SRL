#!/usr/bin/env python3

"""
Functions for subsurface Vs inversion

Author: Qing Ji

This part uses the following python package (version 2.0.1):

    Luu, K. (2021). evodcinv: Inversion of dispersion curves using 
    evolutionary algorithms, doi: 10.5281/zenodo.5785565.
    
    Github page: https://github.com/keurfonluu/evodcinv/tree/v2.0.1

Reference:
    
    Luu, K., M. Noble, A. Gesret, N. Belayouni, and P.-F. Roux (2018). 
    A parallel competitive Particle Swarm Optimization for non-linear 
    first arrival travel time tomography and uncertainty quantification, 
    Comput. Geosci. 113, 81–93.
"""


# Load python packages
from evodcinv import Curve
from my_func.dispersion import get_dispcurve

import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from disba import depthplot, surf96
from disba._common import ifunc
from joblib import Parallel, delayed

from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import Normalize


# Read dispersion curves (4 curves)
def read_curves_from_dispmap(dir_dispmap='./disp_maps/'):
    
    # Files for dispersion maps
    disp_files = ['love_group_0_correction.npz', 
                  'love_phase_0.npz',
                  'rayleigh_phase_0.npz',
                  'rayleigh_phase_1.npz']
    
    # Types of dispersion curves
    curve_kwargs = [{'wave': 'love', 'type': 'group', 'mode': 0},
                    {'wave': 'love', 'type': 'phase', 'mode': 0},
                    {'wave': 'rayleigh', 'type': 'phase', 'mode': 0},
                    {'wave': 'rayleigh', 'type': 'phase', 'mode': 1}]
    
    # Create a list of evodcinv.Curve objects
    curves = []
    for i in range(4):
        
        # Read dispersion map
        disp_data = np.load(os.path.join(dir_dispmap, disp_files[i]))
        if i == 0:
            disp_map = disp_data['cwt_map']
        else:
            disp_map = disp_data['fv_map']
        
        # Obtain dispersion curve
        disp_curve = get_dispcurve(disp_data['freqs'], disp_data['vels'], 
                                   disp_map, index=i+1)
        
        # Curve object
        curve = Curve(np.flip(1/disp_curve['freqs']),  # Periods (incresing)
                      np.flip((disp_curve['upper'] + disp_curve['lower'])/2e3),
                      uncertainties=np.flip((disp_curve['upper'] - disp_curve['lower'])/2e3),
                      weight=1.0, **curve_kwargs[i])
        curves.append(curve)
        
    return curves


# Plot evodcinv.Curve object
def plot_curve(curve, xaxis='freq', scale=1e3, ax=None, **kwargs):
    
    # Note: scale = 1e3 converts from km/s to m/s
    
    # X-axis: Frequency or Period
    if xaxis == 'period':
        x_data = curve.period
        x_label = 'Period (s)'
        x_limit = [1/7, 1]
        
    elif xaxis == 'freq':
        x_data = 1/curve.period
        x_label = 'Frequency (Hz)'
        x_limit = [1, 7]
        
    else:
        raise Exception("Argument xaxis should be 'freq' or 'period'!")
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))
        if curve.uncertainties is None:
            ax.scatter(x_data, curve.data*scale, **kwargs)
        else:
            ax.errorbar(x_data, curve.data*scale, 
                        yerr=curve.uncertainties*scale, 
                        fmt='o', ms=5, **kwargs)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Velocity (m/s)')
        ax.set_xlim(x_limit)
        ax.set_ylim([100, 500])
        return fig, ax
    
    else:
        if curve.uncertainties is None:
            ax.scatter(x_data, curve.data*scale, **kwargs)
        else:
            ax.errorbar(x_data, curve.data*scale, 
                        yerr=curve.uncertainties*scale, 
                        fmt='o', ms=5, **kwargs)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Velocity (m/s)')
        ax.set_xlim(x_limit)
        ax.set_ylim([100, 500])
        return ax


# Plot the range of model parameters
def plot_model_range(model, plot_args=None, ax=None):
    
    d1 = np.array([])
    vs1 = np.array([])
    d2 = np.array([])
    vs2 = np.array([])
    
    for layer in model.layers:
        d1 = np.append(d1, layer.thickness[1])
        vs1 = np.append(vs1, layer.velocity_s[0])
        d2 = np.append(d2, layer.thickness[0])
        vs2 = np.append(vs2, layer.velocity_s[1])
    
    # Plot arguments
    plot_args = plot_args if plot_args is not None else {}
    _plot_args = {"color": "black", "linewidth": 2}
    _plot_args.update(plot_args)
    
    d2[-1] = np.sum(d1) - np.sum(d2[:-1])
    depthplot(d1*1e3, vs1*1e3, None, plot_args=_plot_args, ax=ax)
    depthplot(d2*1e3, vs2*1e3, None, plot_args=_plot_args, ax=ax)
    
    
# Plot inverted models
def plot_models(inv_result, parameter, show="best", stride=1, percent=10, 
                zmax=None, plot_args=None, ax=None, 
                cmap_on=False, cmap_args=None, cmap_range=None):
    
    parameters = {
            "velocity_p": 1,
            "velocity_s": 2,
            "density": 3,
            "vp": 1,
            "vs": 2,
            "rho": 3,
    }
    
    if parameter not in parameters:
        raise ValueError()
    i_param = parameters[parameter]
        
    # Plot arguments
    plot_args = plot_args if plot_args is not None else {}
    _plot_args = {"cmap": "gist_ncar", "color": "black", "linewidth": 2}
    _plot_args.update(plot_args)

    cmap = _plot_args.pop("cmap")
    
    if show == 'percentage':
        # Select top percentage models
        idx = np.argsort(inv_result.misfits)[:]
        models = inv_result.models[idx]
        misfits = inv_result.misfits[idx]
        
        n_select = np.floor((percent/100)*idx.shape[0]).astype(int)
        models = models[n_select::-stride]
        misfits = misfits[n_select::-stride]
        print('Plot %d models.' %misfits.shape[0])

        # Make colormap
        if cmap_range is None:
            norm = Normalize(misfits.min(), misfits.max())
        else:
            norm = Normalize(cmap_range[0], cmap_range[1])
        smap = ScalarMappable(norm, cmap)
        smap.set_array([])

        # Plot models
        for model, misfit in zip(models, misfits):
            tmp = {k: v for k, v in _plot_args.items()}
            tmp["color"] = smap.to_rgba(misfit)
            depthplot(model[:, 0]*1e3, model[:, i_param]*1e3, zmax, plot_args=tmp, ax=ax)

    elif show == "best":
        model = inv_result.model
        depthplot(model[:, 0]*1e3, model[:, i_param]*1e3, zmax, plot_args=_plot_args, ax=ax)
        
    elif show == "mean":
        # Select top percentage models
        idx = np.argsort(inv_result.misfits)[:]
        models = inv_result.models[idx]
        misfits = inv_result.misfits[idx]
        
        n_select = np.floor((percent/100)*idx.shape[0]).astype(int)
        models = models[:n_select+1]
        print('Plot mean of %d models.' %models.shape[0])
        print('Misfit range: %.4f, %.4f.' %(misfits[0], misfits[n_select+1]))
        
        # Plot mean model
        model_mean = np.squeeze(np.mean(models, axis=0))
        depthplot(model_mean[:, 0]*1e3, model_mean[:, i_param]*1e3, zmax, plot_args=_plot_args, ax=ax)
        
    # Customize axes
    gca = ax if ax is not None else plt.gca()
    labels = {
        "velocity_p": "P-wave velocity [m/s]",
        "velocity_s": "S-wave velocity [m/s]",
        "density": "Density [$kg/m^3$]",
        "vp": "$V_p$ [m/s]",
        "vs": "$V_s$ [m/s]",
        "rho": "$\\rho$ [$kg/m^3$]",
    }
    xlabel = labels[parameter]
    ylabel = "Depth [m]"
    gca.set_xlabel(xlabel)
    gca.set_ylabel(ylabel)
    
    # Colorbar
    if cmap_on:
        cmap_args = cmap_args if cmap_args is not None else {}
        _cmap_args = {"orientation": "vertical", "label": "Log Misfit", "location": "right"}
        _cmap_args.update(cmap_args)
        plt.colorbar(smap, **_cmap_args)
        
        
# Plot predicted dispersion curves
def plot_predicted_curve(inv_result, period, mode, wave, type, 
                         show="best", stride=1, percent=10, 
                         plot_args=None, ax=None):
    
    if type not in {"phase", "group", "ellipticity"}:
        raise ValueError()
    
    # Default parameters
    n_jobs = -1
    dc = 0.001
    dt = 0.01
    itype = {"phase": 0, "group": 1}
    units = {"frequency": "Hz", "period": "s"}
    
    # Model dispersion curves
    def get_y(thickness, velocity_p, velocity_s, density):
        c = surf96(period, thickness, velocity_p, velocity_s, density, mode, 
                   itype[type], ifunc["dunkin"][wave], dc, dt)
        idx = c > 0.0
        return c[idx]
    
    # Plot arguments
    plot_args = plot_args if plot_args is not None else {}
    _plot_args = {"type": "line", "xaxis": "period", "yaxis": "velocity", "cmap": "Oranges_r"}
    _plot_args.update(plot_args)

    plot_type = _plot_args.pop("type")
    xaxis = _plot_args.pop("xaxis")
    yaxis = _plot_args.pop("yaxis")
    cmap = _plot_args.pop("cmap")
    
    plot_type = plot_type if plot_type != "line" else "plot"
    plot = getattr(plt if ax is None else ax, plot_type)
    x = 1.0 / period if xaxis == "frequency" else period
    
    if show == 'percentage':
        # Select top percentage models
        idx = np.argsort(inv_result.misfits)[:]
        models = inv_result.models[idx]
        misfits = inv_result.misfits[idx]
        
        n_select = np.floor((percent/100)*idx.shape[0]).astype(int)
        models = models[n_select::-stride]
        misfits = misfits[n_select::-stride]
        print('Plot curves from %d models.' %misfits.shape[0])

        # Make colormap
        norm = Normalize(misfits.min(), misfits.max())
        smap = ScalarMappable(norm, cmap)
        smap.set_array([])

        # Generate and plot curves
        curves = Parallel(n_jobs=n_jobs)(delayed(get_y)(*model.T) for model in models)
        for curve, misfit in zip(curves, misfits):
            y = (1.0 / curve if yaxis == "slowness" else curve*1e3)
            plot(x[: len(y)], y, color=smap.to_rgba(misfit), **_plot_args)

    elif show == "best":
        curve = get_y(*inv_result.model.T)
        y = y = (1.0 / curve if yaxis == "slowness" else curve*1e3)
        plot(x[: len(y)], y, **_plot_args)
        
    # Customize axes
    gca = ax if ax is not None else plt.gca()

    xlabel = f"{xaxis.capitalize()} [{units[xaxis]}]"
    ylabel = f"{type.capitalize()} "
    ylabel += f"{yaxis} [m/s]"
    gca.set_xlabel(xlabel)
    gca.set_ylabel(ylabel)

    # Disable exponential tick labels
    gca.xaxis.set_major_formatter(ScalarFormatter())
    gca.xaxis.set_minor_formatter(ScalarFormatter())


# Obtain the mean model for top percentage
def get_mean_model(inv_result, percent=30):
    
    # Select top percentage models
    idx = np.argsort(inv_result.misfits)[:]
    models = inv_result.models[idx]
    misfits = inv_result.misfits[idx]

    n_select = np.floor((percent/100)*idx.shape[0]).astype(int)
    models = models[:n_select+1]
    print('Get mean of %d models.' %models.shape[0])
    print('Misfit range: %.4f, %.4f.' %(misfits[0], misfits[n_select+1]))

    # Return mean model
    model_mean = np.squeeze(np.mean(models, axis=0))
    return model_mean