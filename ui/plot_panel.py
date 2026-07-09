"""
plot_panel.py
-------------
Builds the interactive Plotly signal viewer.

Features:
  - Full zoom / pan / hover
  - Annotations rendered as scatter markers on the signal
  - Color-coded by annotation type
  - Time axis in seconds
  - Clean dark theme matching SigLearn UI
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ──────────────────────────────────────────────
# Color palette
# ──────────────────────────────────────────────

SIGNAL_COLORS = {
    "ECG"        : "#00D4FF",
    "EEG"        : "#9B59B6",
    "EMG"        : "#FF6B35",
    "PPG"        : "#FF4B4B",
    "Respiration": "#4BFF91",
    "Unknown"    : "#AAAAAA",
}

DARK_BG     = "#0E1117"
PANEL_BG    = "#1A1D27"
GRID_COLOR  = "#2E3347"
TEXT_COLOR  = "#E0E0E0"


def build_signal_plot(
    sig        : np.ndarray,
    fs         : int,
    signal_type: str,
    annotations: list,
    title      : str = "",
    show_envelope: bool = False,
) -> go.Figure:
    """
    Build and return a Plotly figure for the signal.

    Parameters
    ----------
    sig         : 1-D signal array (normalised)
    fs          : sampling frequency
    signal_type : for color selection
    annotations : list of dicts from peak_detector.detect_peaks()
    title       : optional chart title
    show_envelope: if True (EMG) overlay rectified envelope
    """
    t = np.arange(len(sig)) / fs
    color = SIGNAL_COLORS.get(signal_type, SIGNAL_COLORS["Unknown"])

    fig = go.Figure()

    # ── Main signal trace ─────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=t, y=sig,
        mode="lines",
        name=signal_type,
        line=dict(color=color, width=1.5),
        hovertemplate="<b>Time:</b> %{x:.3f} s<br><b>Amplitude:</b> %{y:.4f}<extra></extra>",
    ))

    # ── Optional envelope (for EMG) ───────────────────────────
    if show_envelope:
        rectified = np.abs(sig)
        win       = max(1, int(0.05 * fs))
        envelope  = np.convolve(rectified, np.ones(win) / win, mode="same")
        fig.add_trace(go.Scatter(
            x=t, y=envelope,
            mode="lines",
            name="EMG Envelope",
            line=dict(color="#FFD700", width=1.2, dash="dot"),
            hovertemplate="<b>Envelope</b>: %{y:.4f}<extra></extra>",
        ))

    # ── Annotation markers ────────────────────────────────────
    if annotations:
        # Group by name so legend isn't flooded
        groups: dict = {}
        for ann in annotations:
            key = ann["name"]
            groups.setdefault(key, []).append(ann)

        for ann_name, anns in groups.items():
            x_vals = [ann["time"]           for ann in anns]
            y_vals = [float(sig[ann["index"]]) for ann in anns
                      if ann["index"] < len(sig)]
            if not y_vals:
                continue

            ann_color  = anns[0]["color"]
            ann_symbol = anns[0].get("symbol", "circle")

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                name=ann_name,
                marker=dict(
                    symbol=ann_symbol,
                    size=10,
                    color=ann_color,
                    line=dict(color="white", width=1),
                ),
                hovertemplate=(
                    f"<b>{ann_name}</b><br>"
                    "Time: %{x:.3f} s<br>"
                    "Amp: %{y:.4f}<extra></extra>"
                ),
            ))

    # ── Layout ────────────────────────────────────────────────
    chart_title = title or f"{signal_type} Signal"
    fig.update_layout(
        title=dict(
            text=chart_title,
            font=dict(size=16, color=TEXT_COLOR),
        ),
        paper_bgcolor=DARK_BG,
        plot_bgcolor =PANEL_BG,
        font         =dict(color=TEXT_COLOR, family="monospace"),
        xaxis=dict(
            title     ="Time (s)",
            gridcolor =GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickformat=".2f",
        ),
        yaxis=dict(
            title    ="Amplitude (normalised)",
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
        ),
        legend=dict(
            bgcolor    ="rgba(30,33,48,0.85)",
            bordercolor=GRID_COLOR,
            borderwidth=1,
            font       =dict(size=11),
        ),
        hovermode="x unified",
        dragmode ="zoom",
        height   =420,
        margin   =dict(l=60, r=20, t=50, b=60),
        modebar_remove=["lasso2d", "select2d"],
    )

    # ── Range selector for quick zoom presets ─────────────────
    fig.update_xaxes(
        rangeslider=dict(visible=True, bgcolor=PANEL_BG, thickness=0.05),
        rangeselector=dict(
            bgcolor=PANEL_BG,
            activecolor="#4B8BFF",
            buttons=[
                dict(count=2,  label="2s",  step="second", stepmode="backward"),
                dict(count=5,  label="5s",  step="second", stepmode="backward"),
                dict(count=10, label="10s", step="second", stepmode="backward"),
                dict(step="all", label="All"),
            ],
        ),
    )

    return fig


def build_psd_plot(sig: np.ndarray, fs: int, signal_type: str) -> go.Figure:
    """
    Power Spectral Density plot using Welch's method.
    Displayed alongside the main signal view.
    """
    from scipy import signal as sp

    freqs, psd = sp.welch(sig, fs=fs, nperseg=min(256, len(sig) // 2))
    color      = SIGNAL_COLORS.get(signal_type, SIGNAL_COLORS["Unknown"])

    # Frequency bands to highlight
    band_regions = {
        "ECG" : [(0.5, 40, "ECG band")],
        "EEG" : [
            (0.5, 4,  "Delta"), (4, 8, "Theta"),
            (8, 13, "Alpha"), (13, 30, "Beta"),
        ],
        "EMG" : [(20, min(fs/2, 150), "EMG band")],
        "PPG" : [(0.5, 8, "PPG band")],
        "Respiration": [(0.05, 1.0, "Resp band")],
    }

    fig = go.Figure()

    # Band shading
    band_colors = ["rgba(75,139,255,0.12)", "rgba(75,255,145,0.12)",
                   "rgba(255,184,75,0.12)",  "rgba(255,75,75,0.12)"]
    for i, (f_lo, f_hi, bname) in enumerate(band_regions.get(signal_type, [])):
        fig.add_vrect(
            x0=f_lo, x1=min(f_hi, freqs[-1]),
            fillcolor=band_colors[i % len(band_colors)],
            line_width=0,
            annotation_text=bname,
            annotation_position="top left",
            annotation_font=dict(size=10, color=TEXT_COLOR),
        )

    fig.add_trace(go.Scatter(
        x=freqs, y=10 * np.log10(psd + 1e-12),
        mode="lines",
        name="PSD",
        line=dict(color=color, width=2),
        hovertemplate="<b>Freq:</b> %{x:.1f} Hz<br><b>PSD:</b> %{y:.1f} dB<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Power Spectral Density", font=dict(size=14, color=TEXT_COLOR)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor =PANEL_BG,
        font         =dict(color=TEXT_COLOR, family="monospace"),
        xaxis=dict(title="Frequency (Hz)", gridcolor=GRID_COLOR),
        yaxis=dict(title="Power (dB)",     gridcolor=GRID_COLOR),
        height =280,
        margin =dict(l=60, r=20, t=40, b=50),
        hovermode="x",
    )

    return fig
