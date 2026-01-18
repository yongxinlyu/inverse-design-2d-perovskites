import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn.objects as so
from seaborn import axes_style

from variables import IDENTIFIER_DICT, MORPHING_PATHWAY_DICT, COLUMNS_DICT

def plot_color_bar(x_range, vcenter,ax):

    top = mpl.colormaps['Blues_r'].resampled(128)
    bottom = mpl.colormaps['Oranges'].resampled(128)

    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
    newcmp = ListedColormap(newcolors, name='BlueOrange')

    bar_x = np.arange(x_range[0], x_range[1], 0.01)
    bars = ax.bar(bar_x,-100,color="r",width=0.01)
    norm = mcolors.CenteredNorm(vcenter=vcenter)
    for i, bar in enumerate(bars):
        bar.set_color(newcmp(norm(bar_x[i])))

def plot_HOMO_prediction_histogram(dataframe_list, base_identifier=26):
    dataframe = pd.concat(dataframe_list, axis=1)
    fig, ax = plt.subplots(figsize=(10,2))

    HOMO_base_value = dataframe.loc[base_identifier, 'HOMO_predicted']
    HOMO_cutoff = -11.20

    sns.histplot(dataframe['HOMO_predicted'],bins=250,binrange=[-18,-6],ax=ax,element="step",alpha=0.5,color='tab:gray')
    sns.histplot(dataframe.query('ringcount <=2')['HOMO_predicted'],bins=250,binrange=[-18,-6],ax=ax,element="step",alpha=0.5, color='tab:blue')
    ax.axvline(HOMO_base_value, c='black', alpha=0.5)
    ax.axvline(HOMO_cutoff, c='black',linestyle='--', alpha=0.5)
    #plot_color_bar(x_range=ax.get_xlim(), vcenter=HOMO_base_value, ax=ax)
    #ax.set_xlim(x_lim)
    return fig


def plot_LUMO_prediction_histogram(dataframe_list, base_identifier=26):
    dataframe = pd.concat(dataframe_list, axis=1)
    fig, ax = plt.subplots(figsize=(10,2))

    LUMO_base_value = dataframe.loc[base_identifier, 'LUMO_predicted']
    LUMO_cutoff = -11.20

    sns.histplot(dataframe['LUMO_predicted'],bins=250,binrange=[-13,-3],ax=ax,element="step",alpha=0.5,color='tab:gray')
    sns.histplot(dataframe.query('ringcount <=1')['LUMO_predicted'],bins=250,binrange=[-13,-3],ax=ax,element="step",alpha=0.5, color='tab:blue')
    ax.axvline(LUMO_base_value, c='black', alpha=0.5)
    ax.axvline(LUMO_cutoff, c='black',linestyle='--', alpha=0.5)
    #plot_color_bar(x_range=ax.get_xlim(), vcenter=LUMO_base_value, ax=ax)
    return fig

def get_morphing_pathway_dataframe(input_dataframe, pathway_key='existing'):
    dataframe = pd.DataFrame(data=MORPHING_PATHWAY_DICT[pathway_key]).T
    dataframe_m = pd.melt(dataframe, value_name='identifier', var_name='pathway_index').dropna()
    dataframe_m.set_index('identifier', inplace=True)
    for identifier in dataframe_m.index:
        dataframe_m.at[identifier, 'd1_jitter'] = input_dataframe.at[identifier, 'd1_jitter']
        dataframe_m.at[identifier, 'd2_jitter'] = input_dataframe.at[identifier, 'd2_jitter']
    morphing_pathway_dataframe = dataframe_m.reset_index()
    return morphing_pathway_dataframe

def plot_chemical_space(dataframe_list):
    zoom_range = [-2.8, -2.3, -0.9, -0.4]
    dataframe = pd.concat(dataframe_list, axis=1)
    for identifier in set(sum(MORPHING_PATHWAY_DICT['existing'], [])):
        dataframe.loc[identifier,'exist'] = 'PATHWAY'
    for identifier in IDENTIFIER_DICT['existing_2d']:
        dataframe.loc[identifier,'exist'] = '2D'
        dataframe.loc[identifier, 'text'] = str(int(identifier))
    for identifier in IDENTIFIER_DICT['existing']:
        dataframe.loc[identifier,'exist'] = 'DJ'
        dataframe.loc[identifier, 'text'] = str(int(identifier))

    dataframe = dataframe.query('generation <=4')
    morphing_pathway_dataframe = get_morphing_pathway_dataframe(dataframe, pathway_key='existing')

    fig = mpl.figure.Figure(figsize=(7, 5))
    subfigure = fig.subfigures(1,1)
    (
        so.Plot()
        .add(so.Dots(), data=dataframe, x='d1_jitter', y='d2_jitter', color='ringcount')
        .add(so.Line(color='.1', linewidth=1, alpha=0.2), data=morphing_pathway_dataframe,  x='d1_jitter', y='d2_jitter', group='pathway_index')

        .add(so.Dots(color='.1', alpha=0.1, fill=None), data=dataframe.query("exist=='PATHWAY'"), x='d1_jitter', y='d2_jitter')
        .add(so.Dots(color='.5'), data=dataframe.query("exist=='2D'"), x='d1_jitter', y='d2_jitter')
        .add(so.Dots(color='.1'), data=dataframe.query("exist=='DJ'"), x='d1_jitter', y='d2_jitter')
        .add(so.Text(fontsize=8, valign='bottom', offset=4), data=dataframe, x='d1_jitter', y='d2_jitter', text='text')

        .add(so.Paths(linestyle='-', color='.1', linewidth=1, alpha=0.5), 
             data=pd.DataFrame(
                 data={'x':[zoom_range[0],zoom_range[0],zoom_range[1],zoom_range[1],zoom_range[0]], 
                       'y': [zoom_range[2],zoom_range[3],zoom_range[3],zoom_range[2],zoom_range[2]]}), 
             x='x', y='y')
        .theme(axes_style("ticks"))
        .scale(color='flare')
        .on(subfigure)
        .plot()
    )
    return fig

def plot_chemical_space_zoom(dataframe_list):
    zoom_range = [-2.8, -2.3, -0.9, -0.4]
    dataframe = pd.concat(dataframe_list, axis=1)
    for identifier in set(sum(MORPHING_PATHWAY_DICT['existing'], [])):
        dataframe.loc[identifier,'exist'] = 'PATHWAY'
    for identifier in IDENTIFIER_DICT['existing_2d']:
        dataframe.loc[identifier,'exist'] = '2D'
        dataframe.loc[identifier, 'text'] = str(int(identifier))
    for identifier in IDENTIFIER_DICT['existing']:
        dataframe.loc[identifier,'exist'] = 'DJ'
        dataframe.loc[identifier, 'text'] = str(int(identifier))

    dataframe = dataframe.query('generation <=4')
    morphing_pathway_dataframe = get_morphing_pathway_dataframe(dataframe, pathway_key='existing')

    fig = mpl.figure.Figure(figsize=(7, 5))
    subfigure = fig.subfigures(1,1)
    (
        so.Plot()
        .add(so.Dots(), data=dataframe, x='d1_jitter', y='d2_jitter', color='ringcount')
        .add(so.Line(color='.1', linewidth=1, alpha=0.2), data=morphing_pathway_dataframe, x='d1_jitter', y='d2_jitter', group='pathway_index')

        .add(so.Dots(color='.1', alpha=0.1, fill=None), data=dataframe.query("exist=='PATHWAY'"), x='d1_jitter', y='d2_jitter')
        .add(so.Dots(color='.5'), data=dataframe.query("exist=='2D'"), x='d1_jitter', y='d2_jitter')
        .add(so.Dots(color='.1'), data=dataframe.query("exist=='DJ'"), x='d1_jitter', y='d2_jitter')
        .add(so.Text(fontsize=8, valign='bottom', offset=4),data=dataframe, x='d1_jitter', y='d2_jitter', text='text')
        .limit(x=(zoom_range[0], zoom_range[1]), y=(zoom_range[2], zoom_range[3]))
        .theme(axes_style("ticks"))
        .scale(color='flare')
        .on(subfigure)
        .plot()
    )
    return fig

def plot_chemical_space_pie(dataframe_list):
    dataframe = pd.concat(dataframe_list, axis=1)
    fig, axes = plt.subplots(1, 5, figsize=(20,5))
    for generation in range(5):
        data_generation = dataframe.query('generation=='+str(generation))['ringcount']
        data_generation_counts = data_generation.value_counts().sort_index()
        data_generation_counts.plot(kind='pie', ax= axes[generation],colors = sns.color_palette('flare'))
    return fig

def plot_interactive_chemical_space_by_generation(dataframe_list):
    input_dataframe = pd.concat(dataframe_list, axis=1)
    fig = go.Figure()
    label = ['Generation 0', 'Generation 1', 'Generation 2', 'Generation 3', 'Generation 4']
    for generation in range(4, -1, -1):
        dataframe = input_dataframe.query('generation == '+str(generation)).reset_index()
        trace = go.Scatter(
            x=dataframe['d1_jitter'],
            y=dataframe['d2_jitter'],
            name=label[generation],
            customdata=dataframe, mode='markers',
            marker=dict(
                symbol='circle-open', opacity=0.8, size=4,
                line=dict(width=1), showscale=False,
            ),
            hovertemplate = 'identifier:%{customdata[0]}<extra></extra>',
        )
        fig.add_trace(trace)

    fig.update_layout(
        width=1000, height=800,
        template='plotly_white', xaxis_range=[-5,6],yaxis_range=[-4,7],
    )
    fig.update_yaxes(showline=True, mirror=True)
    fig.update_xaxes(showline=True, mirror=True)
    return fig

def plot_interactive_chemical_space_by_descriptor(dataframe_list):
    input_dataframe = pd.concat(dataframe_list, axis=1).query("generation<=4")
    fig = go.Figure()
    dataframe = input_dataframe.reset_index()

    trace = go.Scatter(
        x = dataframe['d1_jitter'],
        y = dataframe['d2_jitter'],
        customdata = dataframe, mode='markers',
        marker = dict(
            symbol='circle-open', opacity=0.8, size=4,
            line=dict(width=1), showscale=False,
        ),
        hovertemplate = 'identifier:%{customdata[0]}<extra></extra>',
    )

    fig.add_trace(trace)

    def get_button(descriptor):
        button = dict(
            label=descriptor,
            method='restyle',
            args=["marker", 
                  dict(color=dataframe[descriptor],symbol='circle-open', opacity=0.8, size=4,
                       line=dict(width=1), showscale=True,)],
    )
        return button

    descriptor_list = COLUMNS_DICT['organic_descriptors']

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([get_button(descriptor) for descriptor in descriptor_list]),
                pad={"r":10, "t":10}, showactive=True, x=0.1, y=1.09, xanchor='left', yanchor='top'),
        ],
        width=1000, height=800,
        template='plotly_white',
        annotations=[dict(text="Descriptor:", x=0.02, xref="paper", y=1.06, yref="paper", align='left', showarrow=False)]
    )
    fig.update_yaxes(showline=True, mirror=True)
    fig.update_xaxes(showline=True, mirror=True)
    return fig