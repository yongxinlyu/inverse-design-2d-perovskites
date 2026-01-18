import numpy as np
import matplotlib as mpl


import pandas as pd
import seaborn.objects as so
from seaborn import axes_style
from scipy.optimize import curve_fit
import seaborn as sns

from variables import COLOR_PALETTE
sns.set_context('paper')

def plot_vasp_gaussian_comparison(dataframe_list, color='ringcount'):
    dataframe = pd.concat(dataframe_list, axis=1).dropna()
    dataframe['organic_HOMO_LUMO_gap'] = dataframe['organic_LUMO'] - dataframe['organic_HOMO']

    def func(x, a, b):
        return a*x + b
    param_HOMO, param_cov_HOMO = curve_fit(func, dataframe['organic_HOMO'], dataframe['HOMO'])
    dataframe['HOMO_fitted'] = dataframe['organic_HOMO'] * param_HOMO[0] + param_HOMO[1]
    print('HOMO: ', param_HOMO)

    param_LUMO, param_cov_LUMO = curve_fit(func, dataframe['organic_LUMO'], dataframe['LUMO'])
    dataframe['LUMO_fitted'] = dataframe['organic_LUMO'] * param_LUMO[0] + param_LUMO[1]
    print('LUMO: ', param_LUMO)

    param_gap, param_cov_gap = curve_fit(func, dataframe['organic_HOMO_LUMO_gap'], dataframe['HOMO_LUMO_gap'])
    dataframe['HOMO_LUMO_gap_fitted'] = dataframe['organic_HOMO_LUMO_gap'] * param_gap[0] + param_gap[1]
    print('HOMO_LUMO_gap: ', param_gap)

    fig = mpl.figure.Figure(figsize=(7, 3))
    subfigure_HOMO, subfigure_LUMO, subfigure_gap = fig.subfigures(1,3)

    (
        so.Plot(data=dataframe, x='organic_HOMO', y='HOMO', color=color)
        .add(so.Dots())
        .add(so.Line(color=".2"), so.PolyFit(order=1), color=None, y='HOMO_fitted')
        .theme(axes_style("ticks"))
        #.layout(engine='tight') 
        .on(subfigure_HOMO)
        .plot()
    )
    (
        so.Plot(data=dataframe, x='organic_LUMO', y='LUMO', color=color)
        .add(so.Dots())
        .add(so.Line(color=".2"), so.PolyFit(order=1), color=None, y='LUMO_fitted')
        .theme(axes_style("ticks"))
        #.layout(engine='tight')
        .on(subfigure_LUMO)
        .plot()
    )
    (
        so.Plot(data=dataframe, x='organic_HOMO_LUMO_gap', y='HOMO_LUMO_gap', color=color)
        .add(so.Dots())
        .add(so.Line(color=".2"), so.PolyFit(order=1), color=None, y='HOMO_LUMO_gap_fitted')
        .limit(x=[1.5,7.5], y=[1.5,7.5])
        .theme(axes_style("ticks"))
        #.layout(engine='tight')
        .on(subfigure_gap)
        .plot()
    )
    for subfigure in [subfigure_HOMO, subfigure_LUMO, subfigure_gap,]:
        ax = subfigure.axes[0]
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

    return fig

def plot_interlayer_vs_energy_level_alignment(dataframe):
    dataframe_m = pd.melt(
        dataframe, id_vars='d_interlayer', 
        value_vars=[ 'inorganic_cbm_z','inorganic_vbm_z', 'inorganic_cbm_gamma', 'inorganic_vbm_gamma','organic_LUMO','organic_HOMO'],
        var_name='type', value_name='energy')

    fig = mpl.figure.Figure(figsize=(7, 3))
    subfigure1 = fig.subfigures(1,1)
    (
        so.Plot(data=dataframe_m, x='d_interlayer', y='energy', color='type')
        .add(so.Dots())
        .limit(y=(-5,6))
        .theme(axes_style("ticks"))
        .scale(color=['#92c5de','#92c5de','#2166ac', '#2166ac','#d95f0e','#d95f0e'])
        .layout(engine='tight')
        .on(subfigure1)
        .plot()
        
    )
    (
        so.Plot(data=dataframe_m, y='energy', color='type')
        .add(so.Bars(), so.Hist())
        .limit(y=(-5,6))
        .theme(axes_style("ticks"))
        .scale(color=['#92c5de','#92c5de','#2166ac', '#2166ac','#d95f0e','#d95f0e'])
        .layout(engine='tight')
        #.on(subfigure2)
        #.plot()
    )
    return fig

def plot_inorganic_band_edge(dataframe):
    fig = mpl.figure.Figure(figsize=(7, 3))
    subfigure_gap, subfigure_dispersion,= fig.subfigures(1,2)

    dataframe['iodide_interaction'] = dataframe['inorganic_cbm_gamma'] - dataframe['inorganic_cbm_z'] - dataframe['inorganic_vbm_gamma'] + dataframe['inorganic_vbm_z']
    dataframe['inorganic_bandgap'] = dataframe['inorganic_cbm_gamma'] - dataframe['inorganic_vbm_gamma']
    dataframe_m = pd.melt(
        dataframe, id_vars='inorganic_bandgap',value_vars=['angleXMX_average', 'angleMXM_average'],
        value_name='angle',var_name='type')
    
    (
        so.Plot(data=dataframe_m,x='angle',y='inorganic_bandgap',color='type')
        .add(so.Dots())
        .theme(axes_style("ticks"))
        .label(x='Angle (degree)', y='Energy gap (eV)', color='')
        .on(subfigure_gap)
        .plot()
    )

    (
        so.Plot(data=dataframe, x='d_interlayer', y='iodide_interaction')
        .add(so.Dots())
        .theme(axes_style("ticks"))
        .label(x='Interlayer distance (angstrom)', y='Energy dispersion (eV)')
        .on(subfigure_dispersion)
        .plot()
        )
    return fig


def plot_energy_level_alignment_bars(dataframe_list, identifier_list):
    dataframe = pd.concat(dataframe_list, axis=1)
    dataframe = dataframe.loc[identifier_list]
    dataframe['id_vars'] = np.arange(len(identifier_list))

    dataframe_m = pd.melt(
        dataframe, id_vars='id_vars', 
        value_vars=['inorganic_cbm_z','inorganic_vbm_z', 'inorganic_cbm_gamma','inorganic_vbm_gamma', 'organic_LUMO', 'organic_HOMO'],
        var_name='type', value_name='energy'       
    )
    dataframe_upper = pd.melt(
        dataframe, id_vars='id_vars', 
        value_vars=['inorganic_cbm_z', 'organic_LUMO'],
        var_name='type', value_name='energy',
    )
    dataframe_upper.replace('inorganic_cbm_z', 'inorganic_cbm_z_bar',inplace=True)
    dataframe_upper.replace('organic_LUMO', 'organic_LUMO_bar',inplace=True)

    dataframe_lower = pd.melt(
        dataframe, id_vars='id_vars', 
        value_vars=['inorganic_vbm_z', 'organic_HOMO'],
        var_name='type', value_name='energy'
    )
    dataframe_lower.replace('inorganic_vbm_z', 'inorganic_vbm_z_bar',inplace=True)
    dataframe_lower.replace('organic_HOMO', 'organic_HOMO_bar',inplace=True)

    bar_range = [-6, 6]
    warm000 = COLOR_PALETTE['warm000']
    warm002 = COLOR_PALETTE['warm002']
    cold000 = COLOR_PALETTE['cold000']
    cold001 = COLOR_PALETTE['cold001']
    cold002 = COLOR_PALETTE['cold002']

    fig = mpl.figure.Figure(figsize=(10, 5))
    subfigure1 = fig.subfigures(1,1)

    (
        so.Plot(data=dataframe_m, x='id_vars', y='energy')
        .add(so.Bars(baseline=bar_range[1]), data=dataframe_upper, x='id_vars', y='energy',color='type')
        .add(so.Bars(baseline=bar_range[0]), data=dataframe_lower, x='id_vars', y='energy',color='type')
        .add(so.Dot(), data=dataframe_m, x='id_vars', y='energy', color='type')
        .theme(axes_style("ticks"))
        .scale(color=[cold002, warm002, cold002, warm002, cold001, cold001, cold000, cold000, warm000, warm000])
        .on(subfigure1)
        .plot()
    )

    return fig

def plot_energy_level_alignment_quadrant(dataframe_list):
    dataframe = pd.concat(dataframe_list, axis=1)
    dataframe['lumo_cbm_mismatch'] = dataframe['organic_LUMO'] - dataframe['inorganic_cbm_z']
    dataframe['vbm_homo_mismatch'] = dataframe['inorganic_vbm_z'] - dataframe['organic_HOMO']
    dataframe['type'] = np.where(dataframe['generation']<=2, 'training', 'prediction')

    fig = mpl.figure.Figure(figsize=(5, 5))
    subfigure = fig.subfigures(1,1)
    (
        so.Plot(data=dataframe, x='lumo_cbm_mismatch', y='vbm_homo_mismatch')
        .add(so.Dots(), color='type')
        .add(so.Line(linestyle='--', color='.5'), 
             data=pd.DataFrame(data={'x':[0,0], 'y': [-2,6]}), 
             x='x', y='y')
        .add(so.Line(linestyle='--', color='.5'), 
             data=pd.DataFrame(data={'x':[-2,6], 'y': [0,0]}), 
             x='x', y='y')
        .limit(x=(-2,6), y=(-2,6))
        .theme(axes_style("ticks"))
        .on(subfigure)
        .plot()
    )
    ax = subfigure.axes[0]
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    return fig