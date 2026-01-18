import pandas as pd
import matplotlib as mpl
import seaborn.objects as so
from seaborn import axes_style
import seaborn as sns
import plotly.graph_objects as go
import numpy as np

from config import REGRESSOR_DICT, PROJECT_ROOT_DIRECTORY

organic_genome_dataframe = pd.read_csv(
    PROJECT_ROOT_DIRECTORY + '02-metadata/06-csv-files/01-organic-genome.csv', index_col='identifier'
)

def plot_shap_value_count(shap_values_dataframe, order='max'):
    if order == 'max':
        feature_ranking = shap_values_dataframe.max().sort_values().reset_index()
        feature_ranking.columns = ['feature_name', 'max_shap_value']
    elif order == 'min':
        feature_ranking = shap_values_dataframe.min().sort_values(ascending=False).reset_index()
        feature_ranking.columns = ['feature_name', 'min_shap_value']

    dataframe = pd.DataFrame()
    for feature_name in shap_values_dataframe.columns:
        value_count_series = shap_values_dataframe[feature_name].value_counts()
        value_count_dataframe = pd.DataFrame(
            {'shap_value': value_count_series.index,
            'count': value_count_series.values})  
        value_count_dataframe['feature_name'] = str(feature_name)
        value_count_dataframe['feature_ranking'] = feature_ranking.index[feature_ranking['feature_name'] == feature_name][0]
        dataframe = pd.concat([dataframe, value_count_dataframe], ignore_index=True)

    dataframe['y_max'] = dataframe['feature_ranking'] + dataframe['count']/shap_values_dataframe.shape[0]*0.4
    dataframe['y_min'] = dataframe['feature_ranking'] - dataframe['count']/shap_values_dataframe.shape[0]*0.4

    shap_value_count_dataframe = pd.melt(
        dataframe, id_vars=['shap_value', 'feature_name'], 
        value_vars=['y_max', 'y_min'], value_name='y')
    
    fig = mpl.figure.Figure(figsize=(7, 7))
    subfigure = fig.subfigures(1,1)
    (
        so.Plot(data=shap_value_count_dataframe, x='shap_value', y='y')
        .add(so.Range(linewidth=3), group='feature_name')
        .theme(axes_style("ticks"))
        .on(subfigure)
        .plot()
    )
    ax = subfigure.axes[0]
    ax.set_yticks(feature_ranking.index.to_list())
    ax.set_yticklabels(feature_ranking['feature_name'].to_list())
    ax.grid(axis='y', linestyle='--')

    return fig

def plot_model_comparison_by_prediction(dataframe_list, target, score_dataframe):
    dataframe = pd.concat(dataframe_list, axis=1)
    model_name_list = list(REGRESSOR_DICT.keys())
    dataframe_m = pd.melt(
        dataframe, id_vars=[target,'type'], value_vars=model_name_list,
        var_name='model_type',value_name='model_prediction')
    
    axis_range = {
        'HOMO': [-19, -8],
        'LUMO': [-14, -5]
    }

    fig = mpl.figure.Figure(figsize=(10, 10))
    subfigure = fig.subfigures(1, 1)
    (
        so.Plot(
            data=dataframe_m,x='model_prediction', y=target, color='type')
            .facet(col='model_type',wrap=3)
            .add(so.Dots())
            .limit(x=axis_range[target], y=axis_range[target])
            .layout(engine='tight')
            .theme(axes_style("ticks"))
            .on(subfigure)
            .plot()
            )
    
    for axis_index in range(len(subfigure.axes)):
        ax = subfigure.axes[axis_index]
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        sns.lineplot(x=axis_range[target], y=axis_range[target],ax=ax,ls='--',color='grey')
        text_position_x = axis_range[target][0] + 0.15 * (axis_range[target][1] - axis_range[target][0])
        text_position_y = axis_range[target][0] + 0.75 * (axis_range[target][1] - axis_range[target][0])
        r2_score = float(score_dataframe.loc['r2_score',model_name_list[axis_index]])
        rmse = float(score_dataframe.loc['rmse',model_name_list[axis_index]])
        text_to_display = 'R2 score = ' + "{:.2f}".format(r2_score) + '\nRMSE = ' + "{:.2f}".format(rmse)
        ax.text(x=text_position_x, y=text_position_y, s=text_to_display)

    return fig

def plot_model_comparison_by_score(HOMO_score_dataframe, LUMO_score_dataframe):
    HOMO_score_dataframe_m = pd.melt(
        HOMO_score_dataframe.T.reset_index(), 
        id_vars= 'index', value_vars=['train_score', 'test_score'])
    HOMO_score_dataframe_m.columns = ['model','score_type','score']
    HOMO_score_dataframe_m['predictor'] = 'HOMO'

    LUMO_score_dataframe_m = pd.melt(
        LUMO_score_dataframe.T.reset_index(), 
        id_vars= 'index', value_vars=['train_score', 'test_score'])
    LUMO_score_dataframe_m.columns = ['model','score_type','score']
    LUMO_score_dataframe_m['predictor'] = 'LUMO'

    score_dataframe = pd.concat([HOMO_score_dataframe_m, LUMO_score_dataframe_m], ignore_index=True)
    score_dataframe = score_dataframe.astype({'score': float})
    

    fig = mpl.figure.Figure(figsize=(5,5))
    subfigure = fig.subfigures(1, 1)

    (
        so.Plot(data=score_dataframe, x='score', y='model', color='score_type')
        .facet("predictor")
        .add(so.Bar(), so.Dodge(), orient='h')
        .label(y="")
        .theme(axes_style("ticks"))
        .on(subfigure)
        .plot()
    )
    return fig, score_dataframe

def plot_feature_importance(HOMO_feature_importance_dataframe, LUMO_feature_importance_dataframe, feature_order):
    feature_importance_dataframe = pd.concat([HOMO_feature_importance_dataframe, LUMO_feature_importance_dataframe], axis=1)
    feature_importance_dataframe = feature_importance_dataframe.reindex(feature_order)
    feature_importance_dataframe.reset_index(inplace=True)
    feature_importance_dataframe.columns = ['feature_names', 'HOMO_feature_importance', 'LUMO_feature_importance']
    fig = go.Figure()
    fig.update_layout(
        template=None,
    )
    fig.add_trace(go.Scatterpolar(
        r=abs(feature_importance_dataframe['HOMO_feature_importance']),
        theta=feature_importance_dataframe['feature_names'],
        #opacity=0.5,
        name='HOMO',
        mode='lines'
    ))
    fig.add_trace(go.Scatterpolar(
        r=abs(feature_importance_dataframe['LUMO_feature_importance']),
        theta=feature_importance_dataframe['feature_names'],
        #opacity=0.5,
        name='LUMO',
        mode='lines'
    ))

    fig.update_traces(fill='toself')
    return fig

def plot_model_comparison_by_feature_importance(HOMO_dataframe, LUMO_dataframe):
    HOMO_feature_importance_dataframe = HOMO_dataframe.drop(columns=['svr_linear']).reset_index()
    HOMO_feature_importance_dataframe_m = pd.melt(HOMO_feature_importance_dataframe, id_vars='index', var_name='model', value_name='feature_importance')
    HOMO_feature_importance_dataframe_m['predictor'] = 'HOMO'

    LUMO_feature_importance_dataframe = LUMO_dataframe.drop(columns=['svr_linear']).reset_index()
    LUMO_feature_importance_dataframe_m = pd.melt(LUMO_feature_importance_dataframe, id_vars='index', var_name='model', value_name='feature_importance')
    LUMO_feature_importance_dataframe_m['predictor'] = 'LUMO'

    feature_importance_dataframe = pd.concat([HOMO_feature_importance_dataframe_m, LUMO_feature_importance_dataframe_m],ignore_index=True)
    fig = mpl.figure.Figure(figsize=(10, 5))
    subfigure = fig.subfigures(1,1)
    (
        so.Plot(
            data=feature_importance_dataframe,
            x='feature_importance', y='index', color='model')
        .facet(col='predictor')
        .add(so.Bar(), so.Dodge())
        .label(color="", y="")
        .theme(axes_style("ticks"))
        .on(subfigure)
        .plot()
    )
    ax = subfigure.axes[0]
    ax.set_yticklabels(['no. rings', '% ring linkage', '% 6-membered rings','no. primary ammonium', 'linker length', 'ammonium position', 'no. N (pyridine)', 'no. F', 'no. O (furan)', 'no. N (pyrrole)', 'no. side chain (linker)', 'no. side chain (backbone)'])
    return fig

def plot_possible_shap_value(input_dataframe, xlim=[-3,5]):
    dataframe = input_dataframe.sort_values(by=['generation_value'],ascending=False)
    dataframe['y'] = dataframe['feature_index'] * -1
    fig = mpl.figure.Figure(figsize=(5, 5))
    subfigure = fig.subfigures(1,1)
    (
        so.Plot(data=dataframe, 
                x='shap_value', y='y',
                color='generation_value'
                )
        .add(so.Dots())
        .limit(x=xlim)
        .scale(color='Blues_r')
        .label(x="SHAP value (eV)", y="Feature name", color="Generation")
        .theme(axes_style("ticks"))
        .on(subfigure)
        .plot()
    )
    ax = subfigure.axes[0]
    ax.set_yticks(np.arange(0, -12, -1))
    ax.set_yticklabels(['no. rings', '% ring linkage', '% 6-membered rings','no. primary ammonium', 'linker length', 'ammonium position', 'no. N (pyridine)', 'no. F', 'no. O (furan)', 'no. N (pyrrole)', 'no. side chain (linker)', 'no. side chain (backbone)'])
    ax.axvline(x=0, linestyle='--', alpha=0.5, color='black')

    return fig

