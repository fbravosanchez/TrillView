# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:06:43 2026

@author: franc
"""

import dash
from dash import html, dcc

dash.register_page(__name__, path="/help")

layout = html.Div(
    style={
        "display": "flex",
        "flexDirection": "column",
        "alignItems": "left",
        "justifyContent": "left",
        "height": "100vh",
        "fontSize": "20px",
        "marginLeft": "10px"  
    },
    children=[html.Div([
        html.H1("Help", style={"marginTop": "20px", "marginBottom": "10px"}),
        html.H5("TrillView"),
        html.Br(),
        html.P("", style={"marginBottom": "-15px"}),
        html.Span('Start by loading the .csv file on the top left. You may use /example/example_fig.csv'),
        html.Br(),
        html.P("", style={"marginBottom": "10px"}),
        html.Span('Hover over a marker to view the spectrogram. To keep the spectrogram on the display, move the graph using click and drag'),
        html.Br(),
        html.P("", style={"marginBottom": "10px"}),
        html.Span('Use "Spectrogram type" radio buttons to select the type of spectrogram: linear "STFT" (Short Time Fourier Transform) or the logarithmic perceptual scale "mel"'),
        html.Br(),
        html.P("", style={"marginBottom": "10px"}),
        html.Span('"n_fft" selection changes the number of samples per window of the Fourier transform, larger values increase frequency (at the expense of temporal) resolution'),
        html.Br(),
        html.P("", style={"marginBottom": "10px"}),
        html.Span('Click on a marker to listen to the sound. A second click in the same marker does not replay the sound'),
        html.Br(),
        html.P("", style={"marginBottom": "10px"}),
        html.Span('Clicking a legend element hides it, double clicking isolates it (but may rescale the plot)'),
        html.Br(),
        html.P("", style={"marginBottom": "10px"}),
        html.Span('Use select tools (above legend) to select markers. Once selected a button allows saving the selections to view in “Spec View”'),
        html.Br(),
        html.P("", style={"marginBottom": "15px"}),
        html.H5("Spec View"),
        html.Br(),
        html.P("", style={"marginBottom": "-15px"}),
        html.Span('Displays spectrograms of frames saved in TrillView'),
        html.Br(),
        html.P("", style={"marginBottom": "10px"}),
        html.Span('Each spectrogram displays the label and timestamps of the frame and playback tools'),
        html.Br(),
        html.P("", style={"marginBottom": "10px"}),
        html.Span('Edit labels by clicking the label or spectrogram, save with button'),
        html.Br(),
        html.P("", style={"marginBottom": "10px"}),
        html.Span('Dropdown menus control the number of columns per page and number of frames per page'),
        html.Br(),
        html.P("", style={"marginBottom": "10px"}),
        html.Span('Arrow keys and page up/down navigate pages'),
        html.Br(),
        html.P("", style={"marginBottom": "10px"}),
        html.Span('“Spectrogram type” and "n_fft" selections as described above'),
        html.Br(),
        html.P("", style={"marginBottom": "10px"}),        
        ])
        
    ]
)