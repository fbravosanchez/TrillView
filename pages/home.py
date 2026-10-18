# -*- coding: utf-8 -*-
#%% notes
"""
Created by Francisco Bravo Sanchez Oct 2026

TrillView

"""
import dash
from dash import html, dcc

dash.register_page(__name__, path="/")

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
        html.H1("Trill View", style={"marginTop": "20px"}),
        html.P("", style={"marginBottom": "20px"}),
        html.H4("Use links below or in the top bar to navigate between pages"),
        html.P("", style={"marginBottom": "20px"}),
        dcc.Link("Home", href="/"),
        html.Span("  - This page"),
        html.Br(),
        html.P("", style={"marginBottom": "20px"}),

        dcc.Link("TrillView", href="/trill_view"),
        html.Span("  - Interactive scatterplot viewer"),
        html.Br(),
        html.P("", style={"marginBottom": "20px"}),
        html.H5("Scatterplots allow the selection of frames. Once saved you can view the frames in detail and edit labels in Spec View"),
        html.P("", style={"marginBottom": "20px"}),

        dcc.Link("Spec View", href="/spec_view"),
        html.Span("  - Detailed view of selected frames and label editing"),
        html.Br(),

        html.P("", style={"marginBottom": "60px"}),

        html.H4("Citations"),
        html.Br(),
        html.Span('Bravo Sanchez, F. J., Woodbright, M. D., Hossain, M. R., English, N. B., Moore, S. T. Radial Distributions of Bioacoustic Model Predictions: A Visualisation Method that Aids Interpreting Results, Assessing Performance and Explaining Anomalies. Manuscript under review at Ecological Informatics'),

        html.Br(),
        html.P("", style={"marginBottom": "20px"}),
        html.Span('Bravo Sanchez, F. J., English, N. B., Hossain, M. R., & Moore, S. T. (2024). Improved analysis of deep bioacoustic embeddings through dimensionality reduction and interactive visualisation. Ecological Informatics, 81, 102593. DOI 10.1016/j.ecoinf.2024.102593')
        
        ])
        
    ]
)