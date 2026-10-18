# -*- coding: utf-8 -*-
#%% notes
"""
Created by Francisco Bravo Sanchez Oct 2026

TrillView

"""

#%%


from dash import (Dash, dcc, html, Input, Output, no_update, ctx, callback, 
    clientside_callback, State)
import dash
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import pandas as pd
import librosa
import base64
import soundfile as sf
import matplotlib.pyplot as plt
import os
from pathlib import Path
import base64, io


df_column_to_display = 'sps' ## select the column name to display in legend
data_folder = Path('data')
saved_df = []
temp_wav_file = 'data/temp.wav'
clear_on_unhover_choice = True
# spectrogram dimensions
t = 130  
w = 130  
norm = plt.Normalize(vmin=-80, vmax=0)
text_colour = '#3374de'

#%% markers and colours
markers = ['circle', 'circle-open', 'cross', 'diamond', 'diamond-open', 'square', 'square-open', 'x']
unicode_markers = [u"\u2B24", u"\u25EF", u"\u271A", u"\u25C6", u"\u25C7", u"\u25FC", u"\u25FB", u"\u2716"]
colours = [ '#636efa', '#ab63fa', '#B6E880', '#EF553B', '#FECB52', '#FF6692', '#FF97FF', '#FFA15A', '#0048BA','#C46210','#EFDECD','#9966CC','#CD9575','#00FFFF','#7FFFD4','#D0FF14','#FF9966','#FDEE00','#FAE7B5','#9F8170','#F5F5DC','#2E5894','#9C2542','#FFE4C4','#54626F','#BFAFB2','#FFEBCD','#A57164','#0000FF','#1F75FE','#A2A2D0','#6699CC','#5DADEC','#8A2BE2','#3C69E7','#DE5D83','#CB4154','#D891EF','#FFAA1D','#AF6E4D','#DEB887','#A17A74','#5F9EA0','#FFFF99','#00CC99','#FFA6C9','#C95A49','#6D9BC3','#1DACD6','#D2691E','#CD607E','#DA8A67','#AD6F69','#FF7F50','#6495ED','#2E2D88','#FFBCD9','#DC143C','#58427C','#F56FA1','#654321','#008B8B','#B8860B','#006400','#BDB76B','#8B008B','#556B2F','#FF8C00','#9932CC','#8B0000','#E9967A','#8FBC8F','#483D8B','#2F4F4F','#177245','#00CED1','#9400D3','#DA3287','#B94E48','#FF1493','#00BFFF','#4A646C','#1560BD','#2243B6','#EDC9AF','#696969','#1E90FF','#1B1B1B','#614051','#CCFF00','#BF00FF','#CC474B','#E5AA70','#FF5470','#B22222','#FFFAF0','#228B22','#E936A7','#FF00FF','#C154C1','#87421F','#F0F8FF','#ACE5EE','#F0FFFF','#F5F5F5','#FAEBD7','#FEFEFA','#FFF8DC',
            '#00cc96', '#19d3f3', '#636efa', '#ab63fa', '#B6E880', '#EF553B', '#FECB52', '#FF6692', '#FF97FF', '#FFA15A', '#0048BA','#C46210','#EFDECD','#9966CC','#CD9575','#00FFFF','#7FFFD4','#D0FF14','#FF9966','#FDEE00','#FAE7B5','#9F8170','#F5F5DC','#2E5894','#9C2542','#FFE4C4','#54626F','#BFAFB2','#FFEBCD','#A57164','#0000FF','#1F75FE','#A2A2D0','#6699CC','#5DADEC','#8A2BE2','#3C69E7','#DE5D83','#CB4154','#D891EF','#FFAA1D','#AF6E4D','#DEB887','#A17A74','#5F9EA0','#FFFF99','#00CC99','#FFA6C9','#C95A49','#6D9BC3','#1DACD6','#D2691E','#CD607E','#DA8A67','#AD6F69','#FF7F50','#6495ED','#2E2D88','#FFBCD9','#DC143C','#58427C','#F56FA1','#654321','#008B8B','#B8860B','#006400','#BDB76B','#8B008B','#556B2F','#FF8C00','#9932CC','#8B0000','#E9967A','#8FBC8F','#483D8B','#2F4F4F','#177245','#00CED1','#9400D3','#DA3287','#B94E48','#FF1493','#00BFFF','#4A646C','#1560BD','#2243B6','#EDC9AF','#696969','#1E90FF','#1B1B1B','#614051','#CCFF00','#BF00FF','#CC474B','#E5AA70','#FF5470','#B22222','#FFFAF0','#228B22','#E936A7','#FF00FF','#C154C1','#87421F','#F0F8FF','#ACE5EE','#F0FFFF','#F5F5F5','#FAEBD7','#FEFEFA','#FFF8DC',]
unicode_marker_lookup = dict(zip(markers, unicode_markers))

#%% functions

def get_df(contents, file_name):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if file_name.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(decoded), encoding='cp1252')
    if file_name.endswith('csv.zst'):
        df = pd.read_csv(io.BytesIO(decoded), compression='zstd', encoding='cp1252')
    if 'common' not in df.columns:
        df['common'] = ''
    if 'sps' not in df.columns:
        df['sps'] = ''
    if not (("start" in df.columns) and ("end" in df.columns)):
        df[['start', 'end']] = df['timestamps'].str.split("-", expand = True).astype(float)
    else:
        if not "timestamps" in df.columns:
            df['timestamps'] = df['start'].round(1).astype(str) +"-"+ df['end'].round(1).astype(str)

    df.reset_index(drop=False, inplace=True)
    sps = df.groupby([df_column_to_display], as_index=False).agg({'file': 'count'})
    sps.reset_index(drop=False, inplace=True)
    sps['colour_id'], sps['marker_id'] = divmod(sps['index'], 8)
    sps['colour_id'] = np.where(sps['colour_id'] > len(colours) - 1,
                                sps['colour_id'] - len(colours),
                                sps['colour_id'])
    sps['marker'] = [markers[i] for i in sps['marker_id']]
    sps['colour'] = [colours[i] for i in sps['colour_id']]
    df['marker'] = df[df_column_to_display].map(sps.set_index(df_column_to_display)['marker'])
    df['colour'] = df[df_column_to_display].map(sps.set_index(df_column_to_display)['colour'])
        
    if df_column_to_display == 'pred_sps':
        df.loc[df[df_column_to_display]=='Target' , ['colour', 'marker']] = ['red', 'circle']
        df.loc[df[df_column_to_display]=='Non target' , ['colour', 'marker']] = ['#00cc96', 'circle']

    df.loc[df[df_column_to_display]=='-' , ['colour', 'marker']] = ['#00cc96', 'circle']
    return df

def df_to_text(df_hover, df_column_to_display):
    if df_column_to_display == 'pred_sps':
        if ('sps' in df_hover) and (
                df_hover['sps'] == df_hover['sps']):  
            out_txt = (' ' + str(df_hover[df_column_to_display]) 
                       + ' - act: ' + str(df_hover['sps'])
                       + ' - ' + str(df_hover['file']) 
                       + ' - ' + str(df_hover['timestamps'])
                       + ' score ' + str(round(df_hover['pred_score'], 2)))
        else:
            out_txt = (' ' + str(df_hover[df_column_to_display])  
                       + ' - ' + str(df_hover['file']) 
                       + ' - ' + str(df_hover['timestamps']) 
                       + ' score ' + str(round(df_hover['pred_score'], 2)))

    elif df_column_to_display == 'sps':
        out_txt = (' ' + str(df_hover[df_column_to_display]) + ' - ' 
                   + str(df_hover['pred_sps'])
                   +' ' + str(round(df_hover['pred_score']*100, 1)) +'%'
                   + ' - ' + str(df_hover['file']) 
                   + ' - ' + str(df_hover['timestamps']))
    else:
        out_txt = (' ' + str(df_hover[df_column_to_display]) 
                   + ' - ' + str(df_hover.loc['common'])
                   + ' - ' + str(df_hover['file']) 
                   + ' - ' + str(df_hover['timestamps'])                   )
    return out_txt

def file_to_spec(df_hover, spec_type, n_fft_sel):
    y, sr = librosa.load(df_hover['path'], sr=None,
                         offset=df_hover['start'],
                         duration=df_hover['end'] - df_hover['start'],
                         mono=True, res_type='kaiser_fast')
    if spec_type == 'mel':
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=t,
                                              n_fft=n_fft_sel,
                                              hop_length=len(y) // (w - 1))
        img = librosa.power_to_db(mels, ref=np.max)
    elif spec_type == 'STFT':
        stft = librosa.stft(y, n_fft=n_fft_sel,  
                hop_length=(len(y) - n_fft_sel) // (int(n_fft_sel / 2) - 1))

        img = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    img = np.flip(img)
    img = np.flip(img, axis=1)
    im_cm = plt.cm.viridis(norm(img))
    im_cm = (im_cm * 255).astype(np.uint8)
    png = Image.fromarray(im_cm)  
    return (png)

def file_to_base64(df_hover):
    y, sr = librosa.load(df_hover['path'], sr=None,
                         offset=df_hover['start'],
                         duration=df_hover['end'] - df_hover['start'],
                         mono=True, res_type='kaiser_fast')
    sf.write(temp_wav_file, y, sr)
    encoded_sound = base64.b64encode(open(temp_wav_file, 'rb').read())
    return encoded_sound


#%% dash layout

fig = go.Figure()

dash.register_page(__name__, path="/trill_view")

layout = html.Div(
    style={'width': '100%', 'height': '900px'},
    className="container",
    children=[
        html.Audio(id='audio_player_trill', controls=False, autoPlay=False,
                   style={'width': '1%', 'height': '1px'}),
        html.Div(dcc.Upload(
          id='dataset_selection_trill', 
          children=html.Div(['Drag and drop or ',
                             html.A('select a file to start')]),
          multiple=False,    
          style={'width': '350px', 'height': '33px', 'lineHeight': '33px',
        'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '8px',
        'borderColor': '#888', 'textAlign': 'center', 'cursor': 'pointer',
        'backgroundColor': 'rgba(46,125,50,0.12)'
        },),
        style={'display': 'inline-block', 'paddingLeft': 12},
        ),
        html.Span('Spec. type:', id='text_mel',
          style={'display': 'inline-block', 'white-space': 'pre-wrap',
                  'padding-left': 12, 'color': text_colour,}),
        html.Div(dcc.Dropdown(['mel', 'STFT'], 'mel',
         id='spec_radio', clearable=False, searchable=False,),
          style={'display': 'inline-block', 'white-space': 'pre-wrap',
                  'padding-left': 12,}),
        html.Span('n_fft:', id='text_nfft',
          style={'display': 'inline-block', 'white-space': 'pre-wrap',
                  'padding-left': 12, 'color': text_colour,}),
        html.Div(dcc.Dropdown([2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12], 2 ** 10,
                id='nfft_radio', clearable=False, searchable=False,),
          style={'display': 'inline-block', 'white-space': 'pre-wrap',
                  'padding-left': 12,'padding-right': 12}),
        html.Button(id='button1_trill', children='No selections to save', 
                    style={'display': 'inline-block', 
                           'white-space': 'pre-wrap'}),
        html.Span(id='text1_trill',
                  style={'display': 'inline-block', 'white-space': 'pre-wrap', 
                         'padding-left': 5}),
        html.Div(id='debug-out'),
        
        dcc.Graph(id="graph_name_trill", figure=fig,
                  clear_on_unhover=clear_on_unhover_choice,
                  style={'position': 'absolute'}, 
                  config={'scrollZoom': True,  
                         "displayModeBar": True }),
        dcc.Tooltip(id="graph_tooltip_name_trill", direction='bottom'),
        dcc.Store(id='df_sel_store_trill', storage_type='memory'),
        dcc.Store(id='df_store_trill', storage_type='memory'),
        dcc.Store(id='selection_store_trill', storage_type='memory'), ## Stores target of dataframe to show (ie sps)
        dcc.Store(id="csv_file", storage_type="session"),  # from app page
        dcc.Store(id='graph_selection', storage_type='memory'),
        dcc.Store(id='init', data=True),  # fires on load
        dcc.Store(id='selected_ids', storage_type='memory'),
        dcc.Store(id='selected_to_save', storage_type='memory'),  # lean payload.
        dcc.Store(id="debounced_hover_trill"), 

    ],)

#%% dash callbacks

@callback(
    Output('upload_store', 'data'),
    Input('dataset_selection_trill', 'contents'),
    State('dataset_selection_trill', 'filename'),
    prevent_initial_call=True,
)
def save_upload(contents, filename):
    if contents is None:
        raise PreventUpdate
    return ({'contents': contents, 'filename': filename})#, 

@callback(
    Output("graph_name_trill", "figure"),
    Output("df_sel_store_trill", "data"),
    Output("df_store_trill", "data"),
    Output('graph_name_trill', 'clickData'),  
    Output('init', 'data'),
    Output('dataset_selection_trill', 'children'),
    Input('_pages_location', 'pathname'),
    Input('upload_store', 'data'),
    
)
def change_df_trill(pathname,data):
    if data is None:
        raise PreventUpdate   
    file_name = data['filename'] if data else None
    dataset_choice = data['contents']
    fig = go.Figure()
    df = get_df(dataset_choice, file_name)
    df_sel = df[df_column_to_display].unique()


    for i in df_sel:
        i_df = df[df[df_column_to_display] == i]
        fig.add_trace(go.Scatter(
            x=i_df['x'], y=i_df['y'], 
            customdata = i_df["marker"],
            mode='markers',
            marker=dict(symbol=i_df['marker'], color=i_df['colour'], size=10),
            showlegend=True,
            ### italise sps names
            name='<i>' + str(i_df[df_column_to_display].iloc[0]) 
               + '</i>' if (df_column_to_display in ['sps']) and
               (i_df[df_column_to_display].iloc[0] != 'other') else str(
               i_df[df_column_to_display].iloc[0]),
        )
        )

    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
        selector=dict(mode='markers'),)

    fig.update_layout(width=1010, height=785, margin=dict(l=10, r=10, t=15, b=10))  
    fig.update_layout(dragmode='pan')

    return (fig, df_sel, df.to_dict(), None, False,
            html.Div([html.B(file_name), ' (click or drop to replace)']))



#%%

clientside_callback(
    """
    function(hoverData) {
    if (window._hoverTimeout) {
        clearTimeout(window._hoverTimeout);
        window._hoverTimeout = null;
    }
    if (!hoverData) {
        return Promise.resolve(null);
    }
    return new Promise((resolve) => {
        window._hoverTimeout = setTimeout(() => {
            resolve(hoverData);
        }, 500);
    });
}
    """,
    Output("debounced_hover_trill", "data"),
    Input("graph_name_trill", "hoverData"),
)



## Hover action
@callback(
    Output("graph_tooltip_name_trill", "show"),
    Output("graph_tooltip_name_trill", "bbox"),
    Output("graph_tooltip_name_trill", "children"),
    Input("debounced_hover_trill", "data"),
    State("spec_radio", "value"),
    State("nfft_radio", "value"),
    State("df_store_trill", "data"),
    State("df_sel_store_trill", "data"),
    prevent_initial_call=True
)
def fig_func_trill(hoverData, spec_type, n_fft_sel, df_store_trill, df_sel):

    if hoverData is None:
        return False, no_update, no_update

    df = pd.DataFrame(df_store_trill)  ## Return json to df

    hover_data = hoverData["points"][0]  
    bbox = hover_data["bbox"]  
    curve_num = hover_data["curveNumber"]  
    num = hover_data["pointNumber"]  
    marker_colour = hover_data["marker.color"]
    marker_symbol = unicode_marker_lookup[hover_data["marker.symbol"]]

    select_hover = df_sel[curve_num]
    df_hover = df[df[df_column_to_display] == select_hover].iloc[num]
    if df_hover['path'] == "-":
        im_url = None
    else:
        im_url = file_to_spec(df_hover, spec_type, n_fft_sel)
    children = [
        html.Div([
            html.P(children=[
                html.Div(marker_symbol,
                         style={'font-weight': 'bold', 'display': 'inline-block',
                                'color': marker_colour}),
                html.Div(df_to_text(df_hover, df_column_to_display),
                         style={'display': 'block', 'white-space': 'pre-wrap'}),
            ], style={'display': 'flex'}
        ),html.Img(src=im_url,
            style={"width": "400px", 'display': 'block', 'margin': '0 auto'},),])]
    return True, bbox, children



## Click and Button actions
@callback(
    Output("text1_trill", "children", allow_duplicate=True),
    Output("audio_player_trill", "src"),  ## pass source to audio, it has to be base64 encoded
    Output("audio_player_trill", "autoPlay"),  ## pass autoPlay that was initially disabled
    Input('graph_name_trill', 'clickData'),
    State("df_store_trill", "data"),
    State("df_sel_store_trill", "data"),
    State("selection_store_trill", "data"),
    prevent_initial_call=True
)
def clicks_funct_trill(graph_click, df_store_trill, df_sel, dataset_choice):
    df = pd.DataFrame(df_store_trill)  ## Return json to df
    if graph_click is None:
        return '', None, False

    click_data = graph_click["points"][0]
    click_curve_num = click_data["curveNumber"]
    click_point_num = click_data["pointNumber"]
    click_marker_colour = click_data["marker.color"]
    click_marker_symbol = unicode_marker_lookup[click_data["marker.symbol"]]

    select_hover = df_sel[click_curve_num]
    df_hover = df[df[df_column_to_display] == 
                  select_hover].iloc[click_point_num]

    output_txt = [html.Span('Last clicked: ', style={'color': text_colour}),
            html.Span(' ' + click_marker_symbol, 
              style={'display': 'inline-block', 'font-weight': 'bold',
              'color': click_marker_colour, 'white-space': 'pre-wrap'}),
            html.Span(df_to_text(df_hover, df_column_to_display),
              style={'display': 'inline-block', 'white-space': 'pre-wrap'})]
    
    if ctx.triggered_id == 'graph_name_trill':
        if df_hover['path'] == "-":
            return output_txt, None, False
        else:
            encoded_sound = file_to_base64(df_hover)
            return (output_txt, 
            'data:audio/mpeg;base64,{}'.format(encoded_sound.decode()), True)




clientside_callback(
    """
    function(selectedData) {
    if (!selectedData || !selectedData.points) return [];
    var count = selectedData.points.length;
    return ["Save to csv " + String(count) + " markers selected", 
            {backgroundColor: 'green', color: 'white'}, selectedData.points];
    }
    """,
    Output("button1_trill", "children", allow_duplicate=True), 
    Output("button1_trill", "style", allow_duplicate=True), 
    Output('selected_ids', 'data'),
    Input('graph_name_trill', 'selectedData'),
    prevent_initial_call=True
)
def trill_client(a): ...


@callback(
    Output("button1_trill", "children", allow_duplicate=True), 
    Output("button1_trill", "style", allow_duplicate=True), 
    Output('graph_name_trill', 'selectedData'),
    Output("text1_trill", "children", allow_duplicate=True),
    Output("button1_trill", "n_clicks"),
    Input('button1_trill', 'n_clicks'),
    State('selected_ids', 'data'),
    State("df_store_trill", "data"),
    State("csv_file", "data"),
    prevent_initial_call=True
    )

def save_selected_data_trill(button_click, select_data, df_store_trill, 
                             csv_file,):
    tid = ctx.triggered_id
    if tid != 'button1_trill':
        raise PreventUpdate
    if select_data is None:
        raise PreventUpdate
    selected = pd.DataFrame([[d['curveNumber'], 
                              d['pointNumber']] for d in select_data], 
                            columns=['curveNumber','pointNumber'])
    df = pd.DataFrame(df_store_trill) 
    if df.empty:
        raise PreventUpdate
    else:
        df['curveNumber'] = df[df_column_to_display].factorize()[0]
        df['pointNumber'] = df.groupby(df_column_to_display).cumcount()
        df_select = df.merge(selected, on=['curveNumber','pointNumber'],
                             how='inner')
        df_select = df_select[df_select['path']!='-']

    if  len(df_select) == 0:
        output_txt = [html.Span(
            'No sound frames saved. You may select more to view in Spec View',
            style={'color': text_colour})]
        
        button_clicks_out = 0 if button_click==1 else no_update
        
        return ('No selections to save', 
            {"color": "black", "backgroundColor":"#d3d3d3",
            "border": "1px solid #666666"}, None, output_txt, button_clicks_out)
    if button_click == 1:
        df_select.to_csv(csv_file, index = False, header=True)
        output_txt = [html.Span('Saved ' + 
                str(len(df_select)) + 
                ' sound frames. You may select more to view in Spec View', 
                style={'color': text_colour})]
    else:
        df_select.to_csv(csv_file, mode='a', index = False, header=False)
        output_txt = [html.Span('Saved ' + str(len(df_select)) 
        + ' additional sound frames. You may select more to view in Spec View', 
        style={'color': text_colour})]
    return ('No selections to save', 
            {"color": "black", "backgroundColor":"#d3d3d3",
             "border": "1px solid #666666"}, None, output_txt, no_update)
