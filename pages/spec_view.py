# -*- coding: utf-8 -*-
#%% notes
"""
Created by Francisco Bravo Sanchez Oct 2026

TrillView

"""

from __future__ import annotations

import base64
import io
import json
import math
import os
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
from dash import (
    ALL, MATCH, Dash,
    Input, Output, State, callback, clientside_callback,
    ctx, dcc, html, no_update,
)
import io


temp_file = 'data/saved.csv'


DEFAULT_CPP  = 2
DEFAULT_COLS = 2
OUTPUT_CSV   = "data/labelled_frames.csv"
ZMIN, ZMAX = -80, 0


# =============================================================================
# SPECTROGRAM GENERATION
# =============================================================================

def generate_spectrogram(
    filepath: str,
    start: float,
    end: float,
    spec_type: str = "mel",
    n_fft: int = 1024,
    hop_length: Optional[int] = None,
    n_mels: int = 128,
):
    """
    Load a WAV segment and compute a mel or STFT spectrogram in dB.

    Returns
    -------
    S_db  : 2-D ndarray  (freq_bins × n_frames) in dB
    times : 1-D ndarray  time stamps in seconds (offset by `start`)
    freqs : 1-D ndarray  frequency axis in kHz
    """
    if hop_length is None:
        hop_length = n_fft // 4

    y, sr = librosa.load(
        filepath, sr=None,
        offset=float(start),
        duration=max(float(end) - float(start), 0.1),
    )

    if spec_type == "mel":
        S    = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        )
        S_db  = librosa.power_to_db(S, ref=np.max)
        freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr / 2) / 1000
    else:
        D    = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        S_db  = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft) / 1000

    times = (
        librosa.frames_to_time(
            np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length,
        )
        + float(start)
    )
    return S_db, times, freqs


# =============================================================================
# FIGURE BUILDER
# =============================================================================

def build_spectrogram_figure(
    filepath: str,
    start: float,
    end: float,
    spec_type: str = "mel",
    n_fft: int = 1024,
) -> go.Figure:
    """
    Return a Plotly Viridis heatmap for the given WAV clip.
    No per-card colorbar (shared bar shown separately).
    Tight margins — no axis title text, compact ticks only.
    """
    try:
        S_db, times, freqs = generate_spectrogram(
            filepath, start, end, spec_type=spec_type, n_fft=n_fft,
        )
        fig = go.Figure(go.Heatmap(
            z=S_db, x=times, y=freqs,
            colorscale="Viridis",
            zmin=ZMIN, zmax=ZMAX,
            showscale=False,
            hovertemplate=(
                "t=%{x:.3f}s  f=%{y:.2f}kHz  %{z:.1f}dB<extra></extra>"
            ),
        ))
        fig.update_layout(

            uirevision=filepath + str(start),   
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            margin=dict(l=5, r=5, t=5, b=5),

            xaxis=dict(
                tickfont=dict(size=11, color="#555"),
                showgrid=True, gridcolor="#e8e8e8",
                zeroline=False, range=[start, end],
                title=None,
            ),
            yaxis=dict(
                tickfont=dict(size=11, color="#555"),
                showgrid=True, gridcolor="#e8e8e8",
                zeroline=False,
                title=None,
                nticks=5,
            ),
        )
    except Exception as exc:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            annotations=[dict(
                text=f"⚠ Could not load<br><b>{filepath}</b><br>{exc}",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="#cc3333", size=13, family="monospace"),
                align="center",
            )],
            margin=dict(l=4, r=4, t=4, b=4),
        )
    return fig


# =============================================================================
# SHARED COLORBAR FIGURE
# =============================================================================

def build_colorbar_figure() -> go.Figure:
    """
    A thin horizontal colorbar strip (height ~50px rendered via CSS).
    Uses a 1×N heatmap spanning the full dB range.
    """
    n = 200
    z = np.linspace(ZMIN, ZMAX, n).reshape(1, n)
    fig = go.Figure(go.Heatmap(
        z=z,
        x=np.linspace(ZMIN, ZMAX, n),
        y=[0],
        colorscale="Viridis",
        zmin=ZMIN, zmax=ZMAX,
        showscale=False,
        hoverinfo="skip",
    ))

    ticks = list(range(ZMIN, ZMAX + 1, 10))
    annotations = []
    for t in ticks:
        annotations.append(dict(
            x=t, y=0, text=str(t),
            showarrow=False, yanchor="bottom", yshift=4,
            font=dict(size=11, color="#333"),
            xref="x", yref="y",
        ))
    fig.update_layout(
        paper_bgcolor="#f5f5f5",
        plot_bgcolor="#f5f5f5",
        margin=dict(l=8, r=8, t=2, b=16),
        height=26,
        xaxis=dict(
            range=[ZMIN, ZMAX],
            tickvals=ticks,
            ticktext=[str(t)+' dB' for t in ticks],
            tickfont=dict(size=11, color="#333"),
            showgrid=False, zeroline=False,
        ),
        
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    return fig


# =============================================================================
# AUDIO HELPER
# =============================================================================

def clip_to_audio_src(filepath: str, start: float, end: float) -> str:
    """Return a base64 data-URI for the WAV clip, or '' on failure."""
    try:
        y, sr = librosa.load(
            filepath, sr=None,
            offset=float(start),
            duration=max(float(end) - float(start), 0.1),
            mono=True,
        )
        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return "data:audio/wav;base64," + base64.b64encode(buf.read()).decode()
    except Exception:
        return ""


# =============================================================================
# CARD BUILDER
# =============================================================================

def make_card(
    row_idx: int,
    filepath: str,
    file: str,
    timestamps: str,
    start: float,
    end: float,
    label: str,
    spec_type: str = "mel",
    n_fft: int = 1024,
) -> html.Div:
    """One spectrogram card: editable label + spectrogram + audio player."""
    fig       = build_spectrogram_figure(filepath, start, end, spec_type, n_fft)
    audio_src = clip_to_audio_src(filepath, start, end)
    clip_meta = json.dumps({"start": float(start), "end": float(end)})

    return html.Div([

        # Label row 
        html.Div([
            html.Div([
                html.Button(
                    label,
                    id={"type": "label-text", "index": row_idx},
                    className="label-text-btn",
                    title="Click to edit label",
                ),
                html.Button(
                    "✎",
                    id={"type": "edit-btn", "index": row_idx},
                    className="icon-btn edit-btn",
                    title="Edit label",
                ),
            ], id={"type": "label-display", "index": row_idx},
               className="label-display"),

            html.Div([
                dcc.Input(
                    id={"type": "label-input", "index": row_idx},
                    type="text", value=label,
                    className="label-input",
                    debounce=False, n_submit=0,
                ),
                html.Button("✓", id={"type": "label-save",   "index": row_idx},
                            className="icon-btn save-btn",   title="Save"),
                html.Button("✕", id={"type": "label-cancel", "index": row_idx},
                            className="icon-btn cancel-btn", title="Cancel"),
            ], id={"type": "label-edit", "index": row_idx},
               className="label-edit hidden"),
        ], className="card-header"),

        # Spectrogram 
        dcc.Graph(
            figure=fig,
            id={"type": "spec-graph", "index": row_idx},
            config={
                "displayModeBar": False,  
                "scrollZoom": False,
            },
            style={"height": "240px"},
        ),

        # Audio player 
        html.Div([
            (html.Audio(
                id={"type": "audio", "index": row_idx},
                src=audio_src,
                controls=True,
                className="audio-player",
                **{"data-clip": clip_meta, "data-idx": str(row_idx)},
            ) if audio_src else html.Span("⚠ audio unavailable",
                                          className="audio-error")),
            html.Span(f"{file}", className="path-print"),
            html.Span(f"{timestamps}s", className="time-range"),
        ], className="card-footer"),

    ], className="spec-card")


# =============================================================================
# PAGE GRID BUILDER
# =============================================================================

def make_page_grid(
    df: pd.DataFrame,
    page: int,
    cpp: int = DEFAULT_CPP,
    cols: int = DEFAULT_COLS,
    spec_type: str = "mel",
    n_fft: int = 1024,
) -> html.Div:
    start_i = (page - 1) * cpp
    page_df = df.iloc[start_i: start_i + cpp]

    cards = []
    for abs_idx, row in page_df.iterrows():
        label = str(row.get("label", row["filepath"]))
        cards.append(make_card(
            abs_idx, row["filepath"], row["file"], row["timestamps"],
            float(row["start"]), float(row["end"]),
            label, spec_type=spec_type, n_fft=int(n_fft),

        ))

    while len(cards) % cols != 0:
        cards.append(html.Div(className="spec-card spec-card--empty"))

    rows = []
    for j in range(0, len(cards), cols):
        rows.append(html.Div(
            cards[j: j + cols],
            className="spec-row",
            style={"gridTemplateColumns": f"repeat({cols}, 1fr)"},
        ))

    colorbar_strip = dcc.Graph(
        figure=build_colorbar_figure(),
        id="colorbar-strip",
        config={"displayModeBar": False, "staticPlot": True},
        style={"height": "26px"},
    )

    return html.Div([colorbar_strip] + rows, className="main-content")


# =============================================================================
# PAGINATION
# =============================================================================

def make_pagination(current_page: int, n_pages: int,
                    n_total: int, cpp: int) -> html.Div:
    from_n = (current_page - 1) * cpp + 1
    to_n   = min(current_page * cpp, n_total)
    return html.Div([
        html.Button(
            "‹ Prev", id="page-prev", n_clicks=0,
            className="page-btn",
            disabled=(current_page <= 1),
        ),
        html.Span(
            f"Page {current_page} of {n_pages}  ({from_n}–{to_n} of {n_total})",
            className="page-info",
        ),
        html.Button(
            "Next ›", id="page-next", n_clicks=0,
            className="page-btn",
            disabled=(current_page >= n_pages),
        ),
    ], className="pagination-bar")


# =============================================================================
# CSS  +  JS
# =============================================================================

CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:      #f5f5f5; --surface: #ffffff; --border: #d0d0d0;
  --accent:  #2a6db5; --accent2: #5a3e8a;
  --green:   #217a3c; --red: #c0392b;
  --text:    #1a1a1a; --muted:   #666666;
  --radius:  6px;     --gap:     12px;
}
body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; font-size:15px; min-height:100vh; }

.app-header { display:flex; align-items:center; gap:0; padding:0 20px; height:58px; border-bottom:2px solid var(--border); background:var(--surface); overflow-x:auto; white-space:nowrap; box-shadow:0 2px 6px rgba(0,0,0,.08); }
.header-icon { width:32px; height:32px; flex-shrink:0; background:linear-gradient(135deg,var(--accent2),var(--accent)); border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:15px; margin-right:10px; }
.header-title { font-size:17px; font-weight:700; letter-spacing:-.3px; flex-shrink:0; margin-right:6px; color:var(--text); }
.header-sub { font-size:11px; color:var(--muted); font-family:monospace; flex-shrink:0; }
.hdr-sep { width:1px; height:28px; background:var(--border); margin:0 16px; flex-shrink:0; }
.ctrl-group { display:flex; align-items:center; gap:6px; flex-shrink:0; }
.ctrl-label { font-size:11px; color:var(--muted); font-family:monospace; text-transform:uppercase; letter-spacing:.8px; white-space:nowrap; }
.ctrl-sep { width:1px; height:22px; background:var(--border); margin:0 10px; flex-shrink:0; }
.header-badge { margin-left:auto; flex-shrink:0; border:1px solid var(--accent2); color:var(--accent2); font-size:11px; font-family:monospace; padding:3px 10px; border-radius:12px; letter-spacing:1px; background:transparent; }

.ctrl-dd .Select-control { background:var(--bg)!important; border-color:var(--border)!important; min-height:30px!important; height:30px!important; cursor:pointer; }
.ctrl-dd .Select-value-label { color:var(--text)!important; font-family:monospace; font-size:13px; line-height:28px!important; }
.ctrl-dd .Select-menu-outer { background:var(--surface)!important; border-color:var(--border)!important; z-index:9999; }
.ctrl-dd .Select-option { background:var(--surface)!important; color:var(--text)!important; font-family:monospace; font-size:13px; }
.ctrl-dd .Select-option.is-focused { background:#eef4ff!important; }
.ctrl-dd .Select-placeholder { color:var(--muted)!important; line-height:28px!important; font-size:13px; }
.ctrl-dd input { color:var(--text)!important; }

.main-content { padding:10px 16px 4px; }
.spec-row { display:grid; gap:var(--gap); margin-bottom:var(--gap); }
.spec-card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); overflow:hidden; transition:border-color .2s,box-shadow .2s; }
.spec-card:hover { border-color:var(--accent); box-shadow:0 2px 10px rgba(42,109,181,.15); }
.spec-card--empty { background:transparent; border-color:transparent; pointer-events:none; }

.card-header { padding:5px 8px 4px; border-bottom:1px solid var(--border); min-height:32px; background:#f9f9f9; }
.label-display { display:flex; align-items:center; gap:4px; }
.label-text-btn { background:transparent; border:none; color:var(--text); font-size:14px; font-weight:600; font-family:inherit; flex:1; text-align:left; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; cursor:pointer; padding:1px 5px; border-radius:3px; transition:background .15s,color .15s; }
.label-text-btn:hover { background:#eef4ff; color:var(--accent); }
.label-edit { display:flex; align-items:center; gap:5px; }
.label-input { flex:1; background:var(--bg); border:1px solid var(--accent); border-radius:4px; color:var(--text); font-size:13px; padding:3px 8px; outline:none; font-family:inherit; min-width:0; }
.hidden { display:none!important; }

.icon-btn { background:transparent; border:1px solid transparent; border-radius:4px; color:var(--muted); cursor:pointer; font-size:13px; padding:2px 6px; transition:color .15s,border-color .15s,background .15s; flex-shrink:0; }
.edit-btn:hover { color:var(--accent); border-color:var(--accent); }
.save-btn { color:var(--green); border-color:var(--green); }
.save-btn:hover { background:#e8f5ee; }
.cancel-btn { color:var(--red); border-color:var(--red); }
.cancel-btn:hover { background:#fdf0ee; }

.spec-card .js-plotly-plot { cursor:pointer; }

.card-footer { display:flex; align-items:center; padding:4px 8px 5px; gap:8px; border-top:1px solid var(--border); background:#f9f9f9; }
.audio-player { height:26px; flex:1; min-width:0; }
.audio-error { font-size:13px; color:var(--red); font-family:monospace; }
.path-print { font-size:12px; color:var(--muted); font-family:monospace; flex-shrink:0; }
.time-range { font-size:12px; color:var(--muted); font-family:monospace; flex-shrink:0; }

.pagination-bar { display:flex; align-items:center; justify-content:center; gap:14px; padding:10px 20px 16px; }
.page-btn { background:var(--surface); border:1px solid var(--border); color:var(--text); border-radius:5px; padding:6px 18px; font-size:14px; cursor:pointer; font-family:inherit; font-weight:500; transition:border-color .15s,background .15s,color .15s; }
.page-btn:hover:not([disabled]) { border-color:var(--accent); color:var(--accent); background:#eef4ff; }
.page-btn[disabled] { opacity:.4; cursor:default; }
.page-info { font-size:14px; color:var(--muted); font-family:monospace; }

.save-bar { display:flex; align-items:center; justify-content:flex-end; gap:14px; padding:4px 20px 18px; }
.save-btn-main { background:var(--accent2); border:none; color:#fff; border-radius:5px; padding:7px 18px; font-size:14px; font-weight:600; cursor:pointer; font-family:inherit; transition:background .15s; }
.save-btn-main:hover { background:#7a4fb5; }
.save-feedback { font-size:13px; font-family:monospace; color:var(--green); }

::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
"""

EXTRA_JS = r"""
(function () {
  var attached = {};

  function getPlotDiv(idx) {
    var els = document.querySelectorAll('[id]');
    for (var i = 0; i < els.length; i++) {
      var raw = els[i].getAttribute('id');
      if (!raw || raw.indexOf('spec-graph') === -1) continue;
      try {
        var obj = JSON.parse(raw);
        if (obj && obj.type === 'spec-graph' && obj.index === idx)
          return els[i].querySelector('.js-plotly-plot') || null;
      } catch (e) {}
    }
    return null;
  }

  function setPlayhead(idx, xVal) {
    var pd = getPlotDiv(idx);
    if (!pd || !pd._fullLayout) return;
    if (xVal === null) {
      if (pd._fullLayout.shapes && pd._fullLayout.shapes.length > 0)
        Plotly.relayout(pd, { shapes: [] });
    } else {
      Plotly.relayout(pd, { shapes: [{
        type:'line', xref:'x', yref:'paper',
        x0:xVal, x1:xVal, y0:0, y1:1,
        line:{color:'#e03030', width:2.5}
      }]});
    }
  }

  function attachAudio(audio) {
    var idx = parseInt(audio.getAttribute('data-idx'), 10);
    if (attached[idx]) return;
    attached[idx] = true;
    audio.addEventListener('timeupdate', function () {
      if (audio.paused) return;
      var clip = JSON.parse(audio.getAttribute('data-clip') || '{}');
      setPlayhead(idx, (clip.start || 0) + audio.currentTime);
    });
    audio.addEventListener('pause',  function () { setPlayhead(idx, null); });
    audio.addEventListener('ended',  function () { setPlayhead(idx, null); });
    audio.addEventListener('seeked', function () {
      if (audio.paused) setPlayhead(idx, null);
    });
  }

  function scanAudio() {
    document.querySelectorAll('audio[data-idx]').forEach(attachAudio);
  }

  new MutationObserver(function (mutations) {
    mutations.forEach(function (m) {
      m.removedNodes.forEach(function (n) {
        if (n.querySelectorAll)
          n.querySelectorAll('audio[data-idx]').forEach(function (a) {
            delete attached[parseInt(a.getAttribute('data-idx'), 10)];
          });
      });
    });
    scanAudio();
  }).observe(document.body, { childList: true, subtree: true });
  scanAudio();
  
  
  document.addEventListener('keydown', function (e) {
    var tag = (e.target || {}).tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    if (e.key === 'PageUp' || e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      var b = document.getElementById('page-prev');
      if (b && !b.disabled) { b.click(); e.preventDefault(); }
    } else if (e.key === 'PageDown' || e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      var b = document.getElementById('page-next');
      if (b && !b.disabled) { b.click(); e.preventDefault(); }
    }
  });
})();
"""


# =============================================================================
# APP
# =============================================================================


import dash
dash.register_page(__name__, path="/spec_view")
    
def layout():
    
    return html.Div([
    # Stores
    dcc.Store(id="csv_file", storage_type="session"),  
    dcc.Store(id='subpage-data-store', 
              storage_type="session"),  # Holds the fresh df as JSON
    dcc.Store(id="label-store", 
              storage_type="session",),
    dcc.Store(id="page-store",   data=1),
    dcc.Store(id="cpp-store",    data=DEFAULT_CPP, storage_type='session'), # Holds number of frames in page
    dcc.Store(id="cols-store",   data=DEFAULT_COLS, storage_type='session'), # Holds number of columns in page
    dcc.Store(id="params-store", data={"spec_type": "mel", "n_fft": 1024}),
    # Header 
    html.Div([
        html.Div([
            html.Div("Spectrogram Viewer", className="header-title"),
        ]),

        html.Div(className="hdr-sep"),
        html.Div([

            html.Span("Spectrogram type", className="ctrl-label"),
            dcc.Dropdown(
                id="ctrl-spec-type",
                options=[{"label": "Mel",  "value": "mel"},
                         {"label": "STFT", "value": "stft"}],
                value="mel", clearable=False,
                className="ctrl-dd", style={"width": "84px"},
                searchable=False, 
            ),
        ], className="ctrl-group"),

        html.Div(className="ctrl-sep"),

        html.Div([
            html.Span("n_fft", className="ctrl-label"),
            dcc.Dropdown(
                id="ctrl-nfft",
                options=[{"label": str(v), "value": v}
                         for v in [512, 1024, 2048, 4096]],
                value=1024, clearable=False,
                className="ctrl-dd", style={"width": "88px"},
                searchable=False, 
            ),
        ], className="ctrl-group"),

        html.Div(className="ctrl-sep"),

        html.Div([

            html.Span("Page columns", className="ctrl-label"),
            dcc.Dropdown(
                id="ctrl-cols",

                options=[{"label": str(v), "value": v} for v in [1, 2, 3, 4, 5, 6]],
                value=None, clearable=False,
                className="ctrl-dd", style={"width": "62px"},
                searchable=False,
                maxHeight=500,
            ),
        ], className="ctrl-group"),

        html.Div(className="ctrl-sep"),

        html.Div([

            html.Span("Frames per page", className="ctrl-label"),
            dcc.Dropdown(
                id="ctrl-cpp",
                options=[{"label": str(v), "value": v}
                         for v in [1, 2, 3, 4, 6, 8, 10, 12, 15]],
                value=None, clearable=False,
                className="ctrl-dd", style={"width": "68px"},
                searchable=False, 
                maxHeight=500,
            ),
        ], className="ctrl-group"),
        
        html.Div(id="header-badge",
                 className="header-badge"),

    ], className="app-header",),

    # Spectrogram grid
    html.Div(id="spectrogram-grid"),


    html.Div([
        html.Span(id="pagination-container"),
        html.Span("Arrow keys and page up/down navigate pages"),
        # html.Span("May use keys for page navigation"),
        html.Span(id="save-feedback", className="save-feedback"),
        html.Button("💾  Save labels to CSV", id="save-csv-btn",
                    className="save-btn-main", n_clicks=0),
    ], className="save-bar",
    style={"float": "left"},
        
        ),
    
    
    

])


# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
    Output("label-store", "data", allow_duplicate=True),  
    Output("page-store",  "data", allow_duplicate=True),
    Output("subpage-data-store", "data"),
    Output("header-badge", "children"),
    Input("csv_file", "data"),
    prevent_initial_call='initial_duplicate')
def load_csv(csv_path):
    if not csv_path:
        return no_update, no_update
    df = pd.read_csv(csv_path)
    col_aliases = {
        "sps": "label",
        "path": "filepath",
        "remote_path": "filepath",
        "start_time": "start",
        "begin": "start",
        "t_start": "start",
        "end_time": "end",
        "stop": "end",
        "t_end": "end",
        }

    df.rename(columns=col_aliases, inplace=True)
    
    _n_total = len(df)
    
    
    label_store = {str(i): str(row.get("label", row["filepath"])) for i, row in df.iterrows()}
    return label_store, 1, df.to_json(orient='split'), f"Total saved frames: {_n_total}"

@callback(
    Output("params-store", "data"),
    Input("ctrl-spec-type", "value"),
    Input("ctrl-nfft",      "value"),
    prevent_initial_call=False,
)
def sync_params(spec_type, n_fft):
    return {"spec_type": spec_type or "mel", "n_fft": int(n_fft or 1024)}


## Frames per page dropdown controls, including session store
@callback(
    Output("ctrl-cpp", "value"),
    Output("cpp-store", "data"),
    Output("page-store", "data", allow_duplicate=True),
    Input("cpp-store", "data"),
    Input("ctrl-cpp", "value"),
    prevent_initial_call="False",
)
def sync_cpp(stored, selected):
    tid = ctx.triggered_id
    if tid is None or tid == "cpp-store":
        v = stored if stored is not None else DEFAULT_CPP
        return v, no_update, no_update
    else:
        v = int(selected or DEFAULT_CPP)
        return no_update, v, 1


## Columns per page dropdown controls, including session store
@callback(
    Output("ctrl-cols", "value"),
    Output("cols-store", "data"),
    Input("cols-store", "data"),
    Input("ctrl-cols", "value"),
    prevent_initial_call="False",
)
def sync_cols(stored, selected):
    tid = ctx.triggered_id
    if tid is None or tid == "cols-store":
        v = stored if stored is not None else DEFAULT_COLS
        return v, no_update
    else:
        v = int(selected or DEFAULT_COLS)
        return no_update, v



# Pagination 

@callback(
    Output("page-store", "data"),
    Input("page-prev", "n_clicks"),
    Input("page-next", "n_clicks"),
    Input('subpage-data-store', 'data'),
    State("page-store", "data"),
    State("cpp-store",  "data"),
    prevent_initial_call=True,
)
def handle_page_nav(prev_clicks, next_clicks, stored_data, current_page, cpp):
    tid     = ctx.triggered_id
    df = pd.read_json(io.StringIO(stored_data), orient='split')
    n_pages = max(1, math.ceil(len(df) / int(cpp or DEFAULT_CPP)))
    if tid == "page-prev":
        return max(1, current_page - 1)
    if tid == "page-next":
        return min(n_pages, current_page + 1)
    return no_update


# Grid + pagination 

@callback(
    Output("spectrogram-grid",     "children"),
    Output("pagination-container", "children"),
    Input("page-store",   "data"),
    Input("label-store",  "data"),
    Input("cpp-store",    "data"),
    Input("cols-store",   "data"),
    Input("params-store", "data"),
    Input('subpage-data-store', 'data'),
)
def render_page(page, label_store, cpp, cols, params, stored_data):
    cpp     = int(cpp  or DEFAULT_CPP)
    cols    = int(cols or DEFAULT_COLS)
    page    = int(page or 1)
    df = pd.read_json(io.StringIO(stored_data), orient='split').copy()
    n_total = len(df)
    n_pages = max(1, math.ceil(n_total / cpp))
    page    = max(1, min(page, n_pages))
    spec_type = (params or {}).get("spec_type", "mel")
    n_fft     = int((params or {}).get("n_fft", 1024))
    for idx_str, lbl in (label_store or {}).items():
        try:
            df.at[int(idx_str), "label"] = lbl
        except Exception:
            pass

    grid       = make_page_grid(df, page, cpp, cols, spec_type, n_fft)
    pagination = make_pagination(page, n_pages, n_total, cpp)
    return grid, pagination


# Toggle triggered by: label text button, pencil button or graph click

@callback(
    Output({"type": "label-display", "index": MATCH}, "className"),
    Output({"type": "label-edit",    "index": MATCH}, "className"),
    Output({"type": "label-input",   "index": MATCH}, "value"),
    Input({"type": "edit-btn",    "index": MATCH}, "n_clicks"),
    Input({"type": "label-text",  "index": MATCH}, "n_clicks"),
    Input({"type": "label-cancel","index": MATCH}, "n_clicks"),
    Input({"type": "label-save",  "index": MATCH}, "n_clicks"),
    Input({"type": "label-input", "index": MATCH}, "n_submit"),
    Input({"type": "spec-graph",  "index": MATCH}, "clickData"),
    State({"type": "label-text",  "index": MATCH}, "children"),
    prevent_initial_call=True,
)
def toggle_edit_mode(_e, _lt, _c, _s, _ns, _click, current_text):
    tid = ctx.triggered_id
    if tid is None:
        return no_update, no_update, no_update
    t = tid.get("type") if isinstance(tid, dict) else str(tid)
    if t in ("edit-btn", "label-text", "spec-graph"):
        return "label-display hidden", "label-edit", current_text
    return "label-display", "label-edit hidden", no_update


# Persist label edits to store

@callback(
    Output("label-store", "data", allow_duplicate=True),
    Output({"type": "label-text", "index": ALL}, "children"),
    Input({"type": "label-save",  "index": ALL}, "n_clicks"),
    Input({"type": "label-input", "index": ALL}, "n_submit"),
    State({"type": "label-input", "index": ALL}, "value"),
    State({"type": "label-input", "index": ALL}, "id"),
    State("label-store", "data"),
    prevent_initial_call=True,
)
def save_label(_saves, _submits, input_values, input_ids, store):
    tid = ctx.triggered_id
    if tid is None or not isinstance(tid, dict):
        return no_update, no_update
    idx = tid.get("index")
    if idx is None:
        return no_update, no_update

    new_val = next(
        (v for id_obj, v in zip(input_ids, input_values)
         if id_obj.get("index") == idx),
        None,
    )
    if new_val is None:
        return no_update, no_update

    updated = dict(store)
    updated[str(idx)] = new_val
    texts = [updated.get(str(id_obj["index"]), "") for id_obj in input_ids]
    return updated, texts


# Save labels to CSV

@callback(
    Output("save-feedback", "children"),
    Input("save-csv-btn",  "n_clicks"),
    Input('subpage-data-store', 'data'),
    State("label-store",   "data"),
    prevent_initial_call=True,
)
def save_to_csv(n_clicks, stored_data, label_store):
    if not n_clicks:
        return ""
    try:
        df = pd.read_json(io.StringIO(stored_data), orient='split').copy()
        df["edited_label"] = [
            label_store.get(str(i), str(row.get("label", "")))
            for i, row in df.iterrows()
        ]
        df.to_csv(OUTPUT_CSV, index=False)
        return f"✓  Saved → {os.path.abspath(OUTPUT_CSV)}"
    except Exception as exc:
        return f"⚠  {exc}"


