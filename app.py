# -*- coding: utf-8 -*-
#%% notes
"""
Created by Francisco Bravo Sanchez Oct 2026

TrillView

"""
import dash
from dash import Dash, html, dcc
import os

# import uuid

app = dash.Dash(__name__, use_pages=True, suppress_callback_exceptions=True)
server = app.server

saved_csv = 'data/saved_frames.csv'

app.layout = html.Div([
    dcc.Location(id="url"),
       
    html.Div([
        dcc.Link("Home", href="/"),
        dcc.Link("TrillView", href="/trill_view"),
        dcc.Link("Spec View", href="/spec_view"),
        dcc.Link("Help", href="/help"),
        dcc.Store(id="csv_file", storage_type="session", 
                  data=saved_csv),
        dcc.Store(id="upload_store", storage_type="session"),
        ]
    , style={"display": "flex", "gap": "20px", "padding": "10px",
        "backgroundColor": "#f0f0f0"}),

    dash.page_container
        
])

_MAIN_CSS = """
body { margin: 0; font-family: 'Segoe UI', system-ui, sans-serif; background: #f5f5f5; }

.home-page    { min-height: 100vh; display: flex; align-items: center; justify-content: center; }
.home-content { text-align: center; max-width: 700px; padding: 40px 20px; }
.home-title   { font-size: 2rem; font-weight: 700; color: #1a1a1a; margin-bottom: 8px; }
.home-sub     { font-size: 1rem; color: #666; margin-bottom: 32px; }

.nav-grid  { display: flex; flex-wrap: wrap; gap: 16px; justify-content: center; }
.nav-card  {
  width: 200px; padding: 24px 16px; border-radius: 10px;
  background: #fff; border: 1px solid #d0d0d0;
  cursor: pointer; text-align: center; text-decoration: none;
  transition: box-shadow .2s, border-color .2s;
  display: block;
}
.nav-card:hover   { border-color: #2a6db5; box-shadow: 0 4px 14px rgba(42,109,181,.18); }
.card-icon  { font-size: 2rem; margin-bottom: 10px; }
.card-title { font-size: 1rem; font-weight: 600; color: #1a1a1a; margin-bottom: 6px; }
.card-desc  { font-size: 0.8rem; color: #666; line-height: 1.4; }
.progress-bar { box-shadow: 0 0 6px #2a6db5; }
"""



_PROGRESS_JS = """
(function() {
    var bar = document.createElement('div');
    bar.id = 'progress-bar';
    bar.style.cssText = [
        'position:fixed', 'top:0', 'left:0', 'height:3px',
        'width:0', 'background:#2a6db5', 'z-index:99999',
        'transition:width 0.3s ease, opacity 0.4s ease',
        'pointer-events:none'
    ].join(';');
    document.body.appendChild(bar);

    var timer;
    var pending = false;
    var expectedGraphs = 0;

    function start() {
        clearTimeout(timer);
        pending = true;
        // Count how many graphs are currently in the DOM — 
        // the new page should have at least this many
        expectedGraphs = document.querySelectorAll('.js-plotly-plot').length;
        bar.style.opacity = '1';
        bar.style.width = '30%';
    }

    function finish() {
        if (!pending) return;
        pending = false;
        bar.style.width = '100%';
        timer = setTimeout(function() {
            bar.style.opacity = '0';
            setTimeout(function() { bar.style.width = '0'; }, 400);
        }, 300);
    }

    new MutationObserver(function() {
        if (!pending) return;

        // Advance bar incrementally while waiting
        var current = parseFloat(bar.style.width) || 0;
        if (current < 85) bar.style.width = (current + 5) + '%';

        // Count fully rendered plotly graphs (have a drawn canvas inside)
        var rendered = document.querySelectorAll('.js-plotly-plot .gl-container, .js-plotly-plot .svg-container');
        var spinners = document.querySelectorAll('._dash-loading');

        if (spinners.length === 0 && rendered.length >= expectedGraphs && rendered.length > 0) {
            // Small delay to let the last graph fully paint
            setTimeout(finish, 200);
        }
    }).observe(document.body, { childList: true, subtree: true });

    var origFetch = window.fetch;
    window.fetch = function() {
        var url = arguments[0];
        if (typeof url === 'string' && url.indexOf('_dash-update-component') !== -1) {
            start();
            return origFetch.apply(this, arguments).then(function(response) {
                bar.style.width = '60%';
                return response;
            });
        }
        return origFetch.apply(this, arguments);
    };
})();
"""


# Patch index_string now that _MAIN_CSS is defined
app.index_string = (
    "<!DOCTYPE html>\n<html>\n<head>\n"
    "{%metas%}\n<title>{%title%}</title>\n{%favicon%}\n{%css%}\n"
    "<style>" + _MAIN_CSS + "</style>\n"
    "</head>\n<body>\n{%app_entry%}\n"
    "<footer>{%config%}{%scripts%}{%renderer%}</footer>\n"
    "</body>\n</html>"
    "<script>" + _PROGRESS_JS + "</script>\n"
)


if __name__ == "__main__":
    server_port = os.environ.get('PORT', '8080')
    app.run(port=server_port, host='0.0.0.0', debug=False)

