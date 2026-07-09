"""
app.py
------
SigLearn — Biosignal Analyzer & Learning Platform
Phase 1: Upload → Classify → Plot → Parameters → AI Explain

Run locally:
    streamlit run app.py

Deploy: push to HuggingFace Spaces (Streamlit SDK).
"""

import os
import sys
import tempfile
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# Load .env file for local development (before any core imports)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── Path setup so submodules resolve ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.loaders         import (load_ecg, load_emg, load_ppg,
                                   load_respiration, load_eeg,
                                   load_from_upload,
                                   MIT_BIH_RECORDS, BIDMC_RECORDS)
from core.preprocessor    import preprocess
from core.classifier      import classify, LABELS
from core.peak_detector   import detect_peaks
from core.feature_extractor import extract_features
from ui.plot_panel        import build_signal_plot, build_psd_plot
from ai.explainer import AIChat


# Cache EEG loads — MNE re-reads the EDF on every Streamlit rerun without this.
# TTL=3600 means the cache lives for 1 hour; set to None to cache forever per session.
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_load_eeg(subject, run, channel, seconds):
    return load_eeg(subject, run, channel, seconds)


# ══════════════════════════════════════════════════════════════
# Page config & CSS
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SigLearn",
    page_icon ="🫀",
    layout    ="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0E1117;
    color: #E0E0E0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}

/* ── Header ── */
.siglearn-header {
    background: linear-gradient(135deg, #0E1117 0%, #1A1D27 100%);
    border-bottom: 1px solid #2E3347;
    padding: 1.2rem 0 0.8rem 0;
    margin-bottom: 1.5rem;
    text-align: center;
}
.siglearn-title {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    background: linear-gradient(90deg, #00D4FF, #4B8BFF, #9B59B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.siglearn-sub {
    color: #6B7A99;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* ── Cards ── */
.sig-card {
    background: #1A1D27;
    border: 1px solid #2E3347;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.sig-card-title {
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    color: #6B7A99;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

/* ── Classification result ── */
.clf-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.05em;
}
.clf-bar-wrap {
    background: #2E3347;
    border-radius: 6px;
    height: 8px;
    margin: 0.3rem 0 0.15rem 0;
    overflow: hidden;
}
.clf-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.4s ease;
}

/* ── Parameter table ── */
.param-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid #2E3347;
    font-size: 0.88rem;
}
.param-name  { color: #AABBCC; flex: 1.6; }
.param-val   { font-weight: 700; flex: 0.8; text-align: right; color: #E0E0E0; }
.param-unit  { color: #6B7A99; flex: 0.6; text-align: right; font-size: 0.78rem; }
.param-range { color: #4B8BFF; flex: 1.4; text-align: right; font-size: 0.76rem; }

.status-normal { color: #4BFF91; }
.status-high   { color: #FF4B4B; }
.status-low    { color: #FFB84B; }
.status-info   { color: #4B8BFF; }

/* ── Processing steps ── */
.step-item {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.3rem 0;
    font-size: 0.83rem;
    color: #AABBCC;
}
.step-num {
    min-width: 1.6rem;
    height: 1.6rem;
    background: #2E3347;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    color: #4B8BFF;
    font-weight: 700;
}

/* ── AI explanation ── */
.ai-block {
    background: linear-gradient(135deg, #1A1D27, #1E2235);
    border-left: 3px solid #4B8BFF;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #D0D8F0;
}

/* ── Confidence bars ── */
.conf-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0.2rem 0;
    font-size: 0.82rem;
}
.conf-label { flex: 0.8; color: #AABBCC; }
.conf-wrap  { flex: 3; background: #2E3347; border-radius: 4px;
              height: 7px; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 4px; }
.conf-pct   { flex: 0.5; text-align: right; color: #6B7A99; font-size: 0.78rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #12151F !important;
    border-right: 1px solid #2E3347;
}

/* ── Tabs ── */
[data-baseweb="tab"] {
    color: #6B7A99 !important;
    font-size: 0.85rem !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #00D4FF !important;
    border-bottom: 2px solid #00D4FF !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1A1D27, #2E3347);
    border: 1px solid #4B8BFF;
    color: #4B8BFF;
    border-radius: 6px;
    font-family: monospace;
    letter-spacing: 0.05em;
    transition: all 0.2s ease;
}
[data-testid="stButton"] > button:hover {
    background: #4B8BFF;
    color: #0E1117;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════

for key, default in {
    "signal"        : None,
    "fs"            : None,
    "label"         : None,
    "meta"          : {},
    "proc_signal"   : None,
    "proc_steps"    : [],
    "classification": {},
    "annotations"   : [],
    "params"        : [],
    "chat_messages": [],
    "ai_summary": "",
    "ai_questions": [],
    "ai_chat": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="siglearn-header">
    <p class="siglearn-title">🫀 SigLearn</p>
    <p class="siglearn-sub">BIOSIGNAL ANALYZER &amp; LEARNING PLATFORM</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Sidebar — Signal Source
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📡 Signal Source")

    source_mode = st.radio(
        "Load from:",
        ["📂 My Datasets", "⬆️ Upload File"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if source_mode == "📂 My Datasets":
        sig_type = st.selectbox("Signal Type", LABELS)

        sig_type_upper = sig_type.upper()

        # ── ECG ───────────────────────────────────────────────
        if sig_type_upper == "ECG":
            record  = st.selectbox("Record", MIT_BIH_RECORDS, index=0)
            channel = st.selectbox("Channel", [0, 1], index=0,
                                   format_func=lambda x: f"Ch {x}")
            seconds = st.slider("Duration (s)", 5, 30, 10)
            load_fn = lambda: load_ecg(record, channel, seconds)

        # ── EEG ───────────────────────────────────────────────
        elif sig_type_upper == "EEG":
            subject = st.slider("Subject", 1, 109, 1)
            run     = st.selectbox("Run", list(range(1, 15)),
                                   format_func=lambda r: {
                                       1:"Rest (eyes open)", 2:"Rest (eyes closed)"
                                   }.get(r, f"Motor task {r}"))
            channel = st.selectbox("Channel", ["C3", "Cz", "C4", "Fz", "Pz"])
            seconds = st.slider("Duration (s)", 5, 30, 10)
            load_fn = lambda: _cached_load_eeg(subject, run, channel, seconds)

        # ── EMG ───────────────────────────────────────────────
        elif sig_type_upper == "EMG":
            subject  = st.slider("Subject", 1, 27, 1)
            exercise = st.selectbox("Exercise", [1, 2, 3],
                                    format_func=lambda e: {
                                        1: "E1 — Finger movements",
                                        2: "E2 — Isometric/Isotonic hand",
                                        3: "E3 — Grasping",
                                    }[e])
            channel  = st.slider("Electrode", 0, 9, 0)
            seconds  = st.slider("Duration (s)", 5, 30, 10)
            load_fn  = lambda: load_emg(subject, channel, exercise, seconds)

        # ── PPG ───────────────────────────────────────────────
        elif sig_type_upper == "PPG":
            record  = st.selectbox("Record", BIDMC_RECORDS, index=0)
            seconds = st.slider("Duration (s)", 5, 30, 10)
            load_fn = lambda: load_ppg(record, seconds)

        # ── RESPIRATION ───────────────────────────────────────
        elif sig_type_upper == "RESPIRATION":
            record  = st.selectbox("Record", BIDMC_RECORDS, index=0)
            seconds = st.slider("Duration (s)", 10, 60, 30)
            load_fn = lambda: load_respiration(record, seconds)

        if st.button("🔬 Load & Analyze", use_container_width=True):
            # ── Clear ALL previous signal state immediately ──────
            # This ensures stale ECG/EMG/etc never shows while EEG loads
            for k in ("signal","fs","label","meta","proc_signal","proc_steps",
                      "classification","annotations","params","chat_messages", "ai_summary","ai_questions","ai_chat",):
                st.session_state[k] = (
                    None if k in ("signal","fs","proc_signal") else
                    {}   if k in ("meta","classification") else
                    []   if k in ("proc_steps","annotations","chat_messages","ai_questions",) else
                    ""   if k in ("label","ai_summary") else None
                )

            spinner_msg = (
                "📡 Downloading EEG from PhysioNet (first time only, ~2 MB)…"
                if sig_type_upper == "EEG"
                else f"Loading {sig_type} signal…"
            )
            with st.spinner(spinner_msg):
                try:
                    data = load_fn()
                    st.session_state.signal = data["signal"]
                    st.session_state.fs     = data["fs"]
                    st.session_state.label  = data["label"]
                    st.session_state.meta   = data["meta"]
                    st.success(f"✅ {data['label']} loaded  |  {data['fs']} Hz  |  "
                               f"{len(data['signal'])} samples")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Load failed: {e}")

    else:  # Upload
        uploaded = st.file_uploader(
            "Upload signal (.csv / .txt / .npy)",
            type=["csv", "txt", "npy"],
        )
        fs_upload = st.number_input("Sampling Rate (Hz)", 1, 10000, 250)
        if uploaded and st.button("🔬 Load & Analyze", use_container_width=True):
            with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(uploaded.name)[-1]) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                data = load_from_upload(tmp_path, fs=int(fs_upload))
                st.session_state.signal = data["signal"]
                st.session_state.fs     = int(fs_upload)
                st.session_state.label  = "Unknown"
                st.session_state.meta   = data["meta"]
                st.success(f"✅ File loaded  |  {int(fs_upload)} Hz  |  "
                           f"{len(data['signal'])} samples")
            except Exception as e:
                st.error(f"❌ {e}")

    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.72rem;color:#3D4466;'>"
        "SigLearn Phase 1 · ECG/EEG/EMG/PPG/Resp</span>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# Main area — only shown after a signal is loaded
# ══════════════════════════════════════════════════════════════

if st.session_state.signal is None:
    if st.session_state.label == "":
        # Mid-load state — signal cleared, download in progress in sidebar
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem;">
            <p style="font-size:2.5rem;">⏳</p>
            <p style="font-size:1.1rem; color:#6B7A99; margin-top:1rem;">
                Loading signal… please wait.
            </p>
            <p style="font-size:0.82rem; color:#3D4466; margin-top:0.5rem;">
                EEG first-time downloads take 15–60s depending on connection speed.<br>
                Subsequent loads are instant from MNE cache.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Welcome screen ────────────────────────────────────
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; color: #3D4466;">
            <p style="font-size:3rem;">🫀 🧠 💪 💓 🫁</p>
            <p style="font-size:1.1rem; color:#6B7A99; margin-top:1rem;">
                Load a signal from the sidebar to begin analysis.
            </p>
            <p style="font-size:0.85rem; margin-top:0.5rem;">
                Supports ECG · EEG · EMG · PPG · Respiration
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ── Run full pipeline once signal is loaded ───────────────────
sig_raw = st.session_state.signal
fs      = st.session_state.fs
label   = st.session_state.label

# 1. Preprocess
if st.session_state.proc_signal is None:
    with st.spinner("Preprocessing…"):
        proc = preprocess(sig_raw, fs, label)
        st.session_state.proc_signal = proc["signal"]
        st.session_state.proc_steps  = proc["steps"]

sig = st.session_state.proc_signal

# 2. Classify
if not st.session_state.classification:
    with st.spinner("Classifying signal…"):
        clf_result = classify(sig, fs)
        st.session_state.classification = clf_result
        # Override label with classifier output (unless user-loaded from dataset)
        if label == "Unknown":
            st.session_state.label = clf_result["label"]
            label = clf_result["label"]

clf = st.session_state.classification

# 3. Detect peaks
if not st.session_state.annotations:
    with st.spinner("Detecting peaks…"):
        st.session_state.annotations = detect_peaks(sig, fs, label)

annotations = st.session_state.annotations

# 4. Extract parameters
if not st.session_state.params:
    with st.spinner("Computing parameters…"):
        st.session_state.params = extract_features(sig, fs, label)

params = st.session_state.params

# ----------------------------------------
# Initialise AI Tutor
# ----------------------------------------

if st.session_state.ai_chat is None:

    st.session_state.ai_chat = AIChat()

chat = st.session_state.ai_chat

chat.load_analysis(

    signal_type=label,

    classification=clf,

    parameters=params,

    processing_steps=st.session_state.proc_steps,

    annotations=annotations,

    metadata=st.session_state.meta,

)


# ══════════════════════════════════════════════════════════════
# Layout — top info bar
# ══════════════════════════════════════════════════════════════

info_cols = st.columns([1.5, 1, 1, 1])
with info_cols[0]:
    color_map = {
        "ECG":"#00D4FF","EEG":"#9B59B6","EMG":"#FF6B35",
        "PPG":"#FF4B4B","Respiration":"#4BFF91","Unknown":"#AAAAAA"
    }
    c = color_map.get(label, "#AAAAAA")
    st.markdown(
        f'<div class="sig-card">'
        f'<div class="sig-card-title">Signal Type</div>'
        f'<span class="clf-badge" style="background:{c}22;color:{c};border:1px solid {c}44;">'
        f'{label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

with info_cols[1]:
    conf = clf.get("confidence", 0)
    st.markdown(
        f'<div class="sig-card">'
        f'<div class="sig-card-title">Confidence</div>'
        f'<span style="font-size:1.4rem;font-weight:700;color:#4BFF91;">'
        f'{conf*100:.1f}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

with info_cols[2]:
    st.markdown(
        f'<div class="sig-card">'
        f'<div class="sig-card-title">Sample Rate</div>'
        f'<span style="font-size:1.4rem;font-weight:700;">{fs} Hz</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

with info_cols[3]:
    dur = len(sig) / fs
    st.markdown(
        f'<div class="sig-card">'
        f'<div class="sig-card-title">Duration</div>'
        f'<span style="font-size:1.4rem;font-weight:700;">{dur:.1f} s</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════

tab_plot, tab_params, tab_ai, tab_steps = st.tabs([
    "📈 Signal Viewer",
    "📊 Parameters",
    "🤖 AI Explanation",
    "🔧 Processing Log",
])


# ────────────────────────────────────────────
# Tab 1: Signal Viewer
# ────────────────────────────────────────────

with tab_plot:
    col_plot, col_clf = st.columns([3, 1])

    with col_plot:
        show_env = (label.upper() == "EMG")
        fig = build_signal_plot(
            sig, fs, label, annotations,
            title=f"{label} Signal — {st.session_state.meta}",
            show_envelope=show_env,
        )
        st.plotly_chart(fig, use_container_width=True)

        fig_psd = build_psd_plot(sig, fs, label)
        st.plotly_chart(fig_psd, use_container_width=True)

    with col_clf:
        st.markdown(
            '<div class="sig-card"><div class="sig-card-title">Classification Probabilities</div>',
            unsafe_allow_html=True,
        )
        all_probs = clf.get("all_probs", {})
        sorted_probs = sorted(all_probs.items(), key=lambda x: -x[1])
        prob_colors = {
            "ECG":"#00D4FF","EEG":"#9B59B6","EMG":"#FF6B35",
            "PPG":"#FF4B4B","Respiration":"#4BFF91",
        }
        html_bars = ""
        for lbl, prob in sorted_probs:
            pc  = round(prob * 100, 1)
            bc  = prob_colors.get(lbl, "#AAAAAA")
            html_bars += f"""
            <div class="conf-row">
                <span class="conf-label">{lbl}</span>
                <div class="conf-wrap">
                    <div class="conf-fill" style="width:{pc}%;background:{bc};"></div>
                </div>
                <span class="conf-pct">{pc}%</span>
            </div>"""
        st.markdown(html_bars + "</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            f'<div class="sig-card"><div class="sig-card-title">Annotations</div>'
            f'<p style="font-size:0.85rem;color:#4BFF91;">'
            f'{len(annotations)} events detected</p></div>',
            unsafe_allow_html=True,
        )
        if annotations:
            ann_names = list({a["name"] for a in annotations})
            for an in ann_names[:8]:
                count = sum(1 for a in annotations if a["name"] == an)
                st.markdown(
                    f'<div style="font-size:0.8rem;color:#AABBCC;padding:0.2rem 0;">'
                    f'<span style="color:{next(a["color"] for a in annotations if a["name"]==an)}">●</span> '
                    f'{an} &nbsp;<span style="color:#4B8BFF;">×{count}</span></div>',
                    unsafe_allow_html=True,
                )


# ────────────────────────────────────────────
# Tab 2: Parameters
# ────────────────────────────────────────────

with tab_params:
    if params:
        st.markdown('<div class="sig-card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="display:flex;justify-content:space-between;'
            'padding:0.4rem 0;border-bottom:2px solid #4B8BFF;margin-bottom:0.3rem;">'
            '<span style="color:#6B7A99;font-size:0.78rem;">PARAMETER</span>'
            '<span style="color:#6B7A99;font-size:0.78rem;">VALUE</span>'
            '<span style="color:#6B7A99;font-size:0.78rem;">UNIT</span>'
            '<span style="color:#6B7A99;font-size:0.78rem;">NORMAL RANGE</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        for p in params:
            sc = f"status-{p.get('status','info')}"
            st.markdown(
                f'<div class="param-row">'
                f'<span class="param-name">{p["name"]}</span>'
                f'<span class="param-val {sc}">{p["value"]}</span>'
                f'<span class="param-unit">{p["unit"]}</span>'
                f'<span class="param-range">{p["normal"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No parameters available for this signal type yet.")


# ────────────────────────────────────────────
# Tab 3: AI Explanation
# ────────────────────────────────────────────

with tab_ai:

    st.write("✅ AI tab loaded")

    chat = st.session_state.ai_chat

    # ----------------------------------------------------
    # Generate summary once
    # ----------------------------------------------------

    if "ai_summary" not in st.session_state:

        with st.spinner("🧠 AI is analysing the signal..."):

            st.session_state.ai_summary = (
                chat.generate_initial_summary()
            )

            st.session_state.ai_questions = (
                chat.generate_suggested_questions()
            )

    # ----------------------------------------------------
    # Conversation
    # ----------------------------------------------------

    st.subheader("💬 Ask Anything")

    if "chat_messages" not in st.session_state:

        st.session_state.chat_messages = []

    for role, message in st.session_state.chat_messages:

        with st.chat_message(role):

            st.markdown(message)

    question = st.chat_input(
        "Ask anything about this signal..."
    )

    if question:

        st.session_state.chat_messages.append(
            ("user", question)
        )

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            placeholder = st.empty()

            full = ""

            for token in chat.ask_stream(question):

                full += token

                placeholder.markdown(full + "▌")

            placeholder.markdown(full)

        st.session_state.chat_messages.append(
            ("assistant", full)
        )

        st.rerun()

    st.divider()

    # ----------------------------------------------------
    # Reset Conversation
    # ----------------------------------------------------

    if st.button("🗑 Reset Conversation"):

        chat.reset()

        st.session_state.chat_messages = []

        st.session_state.ai_summary = (
            chat.generate_initial_summary()
        )

        st.session_state.ai_questions = (
            chat.generate_suggested_questions()
        )

        st.rerun()

# ────────────────────────────────────────────
# Tab 4: Processing Log
# ────────────────────────────────────────────

with tab_steps:
    st.markdown("#### 🔧 Processing Pipeline")
    st.caption(
        "Every transformation applied to your signal — "
        "full transparency, no black boxes."
    )
    steps = st.session_state.proc_steps
    if steps:
        html_steps = ""
        for i, step in enumerate(steps):
            html_steps += (
                f'<div class="step-item">'
                f'<span class="step-num">{i+1}</span>'
                f'<span>{step}</span>'
                f'</div>'
            )
        st.markdown(
            f'<div class="sig-card">{html_steps}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No processing steps recorded.")
