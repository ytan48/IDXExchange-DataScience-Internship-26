"""Streamlit app for the final residential valuation model."""

from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from src.geocoding import (
    GeocodedAddress,
    GeocodingError,
    geocode_address,
    match_known_category,
)
from src.inference import load_artifacts, predict_price
from src.input_validation import normalize_whole_number


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "xgboost_final"
DEFAULT_ADDRESS_QUERY = "6175 Oneida Drive, San Jose, CA 95123"

WHOLE_NUMBER_LIMITS = {
    "BedroomsTotal": (0, 20),
    "BathroomsTotalInteger": (0, 20),
    "Stories": (1, 4),
    "GarageSpaces": (0, 20),
    "ParkingTotal": (0, 50),
}


st.set_page_config(
    page_title="California Home Valuation | IDX Exchange",
    page_icon="⌂",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&display=swap');

    :root {
        --paper: #f3eee5;
        --paper-deep: #e8dfd1;
        --cream: #fffaf1;
        --ink: #18332c;
        --ink-soft: #29483f;
        --muted: #52635c;
        --line: rgba(24, 51, 44, 0.17);
        --line-strong: rgba(24, 51, 44, 0.32);
        --cinnabar: #d45432;
        --cinnabar-dark: #a83c22;
        --sun: #e9b44c;
        --white: #fffdf8;
        --shadow: 0 20px 60px rgba(47, 50, 37, 0.09);
    }

    html { scroll-behavior: smooth; }
    body, .stApp {
        font-family: 'Manrope', sans-serif;
        font-size: 16px;
        line-height: 1.55;
        color: var(--ink);
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }
    .stApp {
        background:
            radial-gradient(circle at 84% 4%, rgba(233, 180, 76, 0.18), transparent 25rem),
            radial-gradient(circle at 2% 35%, rgba(24, 51, 44, 0.07), transparent 28rem),
            var(--paper);
    }
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.24;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.1'/%3E%3C/svg%3E");
        z-index: 0;
    }
    header[data-testid='stHeader'] { background: transparent; }
    [data-testid='stToolbar'], #MainMenu, footer { visibility: hidden; }
    .block-container {
        max-width: 1320px;
        padding: 1.15rem 2.4rem 4rem;
        position: relative;
        z-index: 1;
    }

    h1, h2, h3, [data-testid='stHeadingWithActionElements'] {
        color: var(--ink);
        font-family: 'Fraunces', Georgia, serif;
        letter-spacing: 0;
    }
    p { color: var(--ink-soft); }

    .masthead {
        border-top: 6px solid var(--ink);
        padding: 1.55rem 0 2.35rem;
        margin-bottom: 0.25rem;
        animation: rise-in 650ms cubic-bezier(.2,.75,.25,1) both;
    }
    .hero-kicker {
        color: var(--cinnabar-dark);
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }
    div[data-testid='stMarkdownContainer'] h1.hero-title {
        color: var(--ink);
        font-family: 'Fraunces', Georgia, serif !important;
        font-size: clamp(3.15rem, 7.2vw, 6.5rem) !important;
        font-weight: 700 !important;
        letter-spacing: 0 !important;
        line-height: 0.92 !important;
        margin: 0 !important;
        max-width: 980px;
    }
    .hero-title em { color: var(--cinnabar); font-style: italic; }
    .hero-footer {
        padding-top: 1.55rem;
        max-width: 650px;
    }
    .hero-scope {
        align-items: baseline;
        border-left: 3px solid var(--cinnabar);
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem 0.8rem;
        margin: 0 0 0.8rem;
        padding: 0.15rem 0 0.15rem 0.95rem;
    }
    .hero-scope span {
        color: var(--cinnabar-dark);
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0;
        text-transform: uppercase;
    }
    .hero-scope strong {
        color: var(--ink);
        font-family: 'Fraunces', Georgia, serif;
        font-size: 1.4rem;
        font-weight: 700;
        line-height: 1.3;
    }
    .hero-copy {
        font-size: 1.02rem;
        line-height: 1.75;
        margin: 0;
        max-width: 590px;
    }
    .workspace-label {
        align-items: center;
        color: var(--muted);
        display: flex;
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        gap: 0.7rem;
        letter-spacing: 0;
        margin: 0.3rem 0 0.8rem;
        text-transform: uppercase;
    }
    .workspace-label::after { background: var(--line-strong); content: ''; height: 1px; flex: 1; }

    div[data-testid='stVerticalBlockBorderWrapper'] {
        background: rgba(255, 250, 241, 0.82);
        border: 1px solid var(--line) !important;
        border-radius: 1.25rem;
        box-shadow: 0 1px 0 rgba(255,255,255,.65);
        padding: 0.2rem;
        transition: border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
    }
    div[data-testid='stVerticalBlockBorderWrapper']:focus-within {
        border-color: rgba(24, 51, 44, 0.38) !important;
        box-shadow: 0 10px 30px rgba(47, 50, 37, 0.07);
    }

    .section-head {
        align-items: flex-start;
        display: grid;
        gap: 1rem;
        grid-template-columns: 3.2rem 1fr;
        padding: 0.15rem 0 0.8rem;
    }
    .section-index {
        align-items: center;
        background: var(--ink);
        border-radius: 50%;
        color: var(--cream);
        display: flex;
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        height: 2.65rem;
        justify-content: center;
        width: 2.65rem;
    }
    .section-title {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: 0;
        line-height: 1.25;
        margin: 0.02rem 0 0.28rem;
    }
    .section-copy { color: var(--ink-soft); font-size: 0.86rem; line-height: 1.65; margin: 0; }
    .field-group-label {
        color: var(--cinnabar-dark);
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0;
        margin: 0.35rem 0 0.2rem;
        text-transform: uppercase;
    }

    [data-testid='stWidgetLabel'] p,
    [data-testid='stMarkdownContainer'] label p {
        color: var(--ink);
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0;
    }
    [data-testid='stNumberInput'], [data-testid='stSelectbox'],
    [data-testid='stDateInput'], [data-testid='stTextInput'] { margin-bottom: 0.2rem; }
    [data-baseweb='input'], [data-baseweb='select'] > div {
        background-color: rgba(255, 253, 248, 0.95) !important;
        border-color: var(--line) !important;
        border-radius: 0.7rem !important;
        min-height: 2.9rem;
        transition: border-color 150ms ease, box-shadow 150ms ease;
    }
    [data-baseweb='input']:focus-within, [data-baseweb='select'] > div:focus-within {
        border-color: var(--cinnabar) !important;
        box-shadow: 0 0 0 3px rgba(212, 84, 50, 0.10) !important;
    }
    input, [data-baseweb='select'] {
        color: var(--ink) !important;
        font-family: 'Manrope', sans-serif !important;
        font-size: 0.9rem !important;
    }
    input::placeholder { color: #9a9d92 !important; opacity: 1 !important; }
    [data-testid='stNumberInputStepDown'], [data-testid='stNumberInputStepUp'] {
        color: var(--muted) !important;
    }
    [data-testid='stNumberInput'] { position: relative; }
    [data-testid='stNumberInput'] > div:has(> [data-testid='InputInstructions']) {
        bottom: auto !important;
        left: auto !important;
        margin: 0.3rem 0 0;
        pointer-events: none;
        position: absolute !important;
        right: 0.15rem !important;
        text-align: right;
        top: 100% !important;
        white-space: nowrap;
        z-index: 2;
    }
    [data-testid='stNumberInput'] [data-testid='InputInstructions'] {
        color: var(--muted) !important;
        font-family: 'Manrope', sans-serif !important;
        font-size: 0.72rem !important;
        line-height: 1.35 !important;
        inset: auto !important;
        position: static !important;
    }

    [data-testid='stSegmentedControl'] [data-baseweb='button-group'] {
        background: var(--paper-deep);
        border-radius: 0.8rem;
        padding: 0.24rem;
    }
    [data-testid='stSegmentedControl'] button {
        border: 0 !important;
        border-radius: 0.62rem !important;
        font-size: 0.86rem;
        min-height: 2.5rem;
    }
    [data-testid='stSegmentedControl'] button[aria-pressed='true'] {
        background: var(--white) !important;
        box-shadow: 0 3px 10px rgba(47,50,37,.08);
        color: var(--ink) !important;
    }

    .stButton > button, [data-testid='stFormSubmitButton'] button {
        border: 1px solid var(--line-strong);
        border-radius: 0.75rem;
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: 0.86rem;
        font-weight: 700;
        min-height: 2.9rem;
        transition: all 160ms ease;
    }
    .stButton > button:hover {
        border-color: var(--ink);
        color: var(--ink);
        transform: translateY(-1px);
    }
    .stButton > button[kind='primary'] {
        background: var(--cinnabar) !important;
        border-color: var(--cinnabar) !important;
        border-radius: 0.9rem;
        box-shadow: 0 9px 0 var(--cinnabar-dark), 0 18px 32px rgba(168, 60, 34, 0.18);
        color: #fffaf1 !important;
        font-size: 0.95rem;
        letter-spacing: 0;
        margin: 0.35rem 0 0.65rem;
        min-height: 3.75rem;
    }
    .stButton > button[kind='primary'] p { color: #fffaf1 !important; }
    .stButton > button[kind='primary']:hover {
        box-shadow: 0 6px 0 var(--cinnabar-dark), 0 14px 24px rgba(168, 60, 34, 0.18);
        transform: translateY(3px);
    }
    .stButton > button[kind='primary']:active {
        box-shadow: 0 2px 0 var(--cinnabar-dark);
        transform: translateY(7px);
    }
    .stButton > button:disabled { box-shadow: none !important; transform: none !important; }

    [data-testid='stCheckbox'] p { color: var(--ink); font-size: 0.82rem; font-weight: 700; }
    [data-testid='stCheckbox'] label:has(input[role='switch']:not(:checked)) > div:first-of-type {
        background-color: rgba(24, 51, 44, 0.22) !important;
    }
    [data-testid='stCheckbox'] label[data-hovered]:has(input[role='switch']:not(:checked)) > div:first-of-type,
    [data-testid='stCheckbox'] label:has(input[role='switch']:not(:checked)):hover > div:first-of-type {
        background-color: rgba(24, 51, 44, 0.34) !important;
    }
    details {
        background: rgba(255, 253, 248, 0.55);
        border: 1px solid var(--line) !important;
        border-radius: 0.8rem !important;
    }
    details summary p { color: var(--ink) !important; font-size: 0.84rem !important; font-weight: 700 !important; }
    [data-testid='stAlert'] { border-radius: 0.8rem; font-size: 0.84rem; }

    .match-card {
        align-items: center;
        background: rgba(80, 116, 84, 0.09);
        border: 1px solid rgba(80, 116, 84, 0.22);
        border-radius: 0.8rem;
        display: flex;
        gap: 0.75rem;
        margin: 0.15rem 0 0.7rem;
        padding: 0.75rem 0.9rem;
    }
    .match-dot {
        background: #4f7454;
        border-radius: 50%;
        box-shadow: 0 0 0 4px rgba(79, 116, 84, 0.12);
        flex: 0 0 auto;
        height: 0.5rem;
        width: 0.5rem;
    }
    .match-address { color: var(--ink); font-size: 0.84rem; font-weight: 700; }
    .match-coordinates { color: var(--ink-soft); font-family: 'DM Mono', monospace; font-size: 0.72rem; margin-top: 0.15rem; }
    .coordinate-pair {
        background: var(--ink);
        border-radius: 0.85rem;
        color: var(--cream);
        display: grid;
        grid-template-columns: 1fr 1fr;
        margin-top: 0.2rem;
        overflow: hidden;
    }
    .coordinate-cell { padding: 0.78rem 0.95rem; }
    .coordinate-cell + .coordinate-cell { border-left: 1px solid rgba(255,255,255,.17); }
    .coordinate-label { color: rgba(255,250,241,.78); font-family: 'DM Mono', monospace; font-size: .68rem; letter-spacing: 0; text-transform: uppercase; }
    .coordinate-value { display: block; font-family: 'DM Mono', monospace; font-size: .88rem; margin-top: .24rem; }

    .inline-estimate {
        align-items: center;
        background: var(--ink);
        border-left: 4px solid var(--cinnabar);
        border-radius: 0.45rem;
        color: var(--cream);
        display: grid;
        gap: 0.75rem 1.2rem;
        grid-template-columns: minmax(0, 1fr) auto;
        margin-top: 0.85rem;
        padding: 1rem 1.15rem;
    }
    .inline-estimate-label {
        color: rgba(255,250,241,.78);
        display: block;
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .inline-estimate-subject {
        color: rgba(255,250,241,.9);
        display: block;
        font-size: 0.82rem;
        line-height: 1.5;
        margin-top: 0.3rem;
    }
    .inline-estimate-note {
        color: rgba(255,250,241,.78);
        display: block;
        font-size: 0.76rem;
        font-weight: 700;
        line-height: 1.45;
        margin-top: 0.35rem;
    }
    .inline-estimate-price {
        color: var(--cream);
        font-family: 'Fraunces', Georgia, serif;
        font-size: clamp(2rem, 3.5vw, 3rem);
        font-weight: 700;
        letter-spacing: 0;
        line-height: 1;
        white-space: nowrap;
    }

    .result-anchor { height: 0; }
    .result-kicker {
        color: var(--cinnabar-dark);
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0;
        text-transform: uppercase;
    }
    .result-title {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: 0;
        line-height: 1.25;
        margin: 0.35rem 0 1rem;
    }
    .estimate-price {
        color: var(--cinnabar);
        font-family: 'Fraunces', Georgia, serif;
        font-size: clamp(2.45rem, 4vw, 4rem);
        font-weight: 700;
        letter-spacing: 0;
        line-height: 1;
        margin: 0.6rem 0 0.5rem;
    }
    .estimate-disclaimer {
        background: rgba(212, 84, 50, 0.10);
        border-left: 3px solid var(--cinnabar);
        border-radius: 0.45rem;
        color: var(--ink);
        font-size: 0.84rem;
        font-weight: 700;
        line-height: 1.55;
        margin: 0.85rem 0 0.35rem;
        padding: 0.68rem 0.78rem;
    }
    .estimate-subject { color: var(--ink-soft); font-size: 0.82rem; line-height: 1.6; }
    .range-card {
        border-bottom: 1px solid var(--line);
        border-top: 1px solid var(--line);
        margin: 1.15rem 0;
        padding: 0.9rem 0;
    }
    .range-label { color: var(--ink-soft); font-family: 'DM Mono', monospace; font-size: .68rem; letter-spacing: 0; text-transform: uppercase; }
    .range-value { color: var(--ink); font-size: .95rem; font-weight: 700; margin-top: .28rem; }
    .empty-figure {
        align-items: center;
        aspect-ratio: 1.25;
        background:
            linear-gradient(135deg, transparent 49.5%, rgba(24,51,44,.12) 50%, transparent 50.5%),
            repeating-linear-gradient(0deg, transparent 0 27px, rgba(24,51,44,.06) 28px),
            rgba(232, 223, 209, .48);
        border-radius: .9rem;
        display: flex;
        justify-content: center;
        margin: .9rem 0 1rem;
        overflow: hidden;
        position: relative;
    }
    .empty-figure::before {
        border: 1px solid rgba(24,51,44,.28);
        border-radius: 50% 50% 50% 10%;
        content: '';
        height: 4.8rem;
        transform: rotate(-18deg);
        width: 4.8rem;
    }
    .empty-figure::after {
        background: var(--cinnabar);
        border-radius: 50%;
        content: '';
        height: .65rem;
        position: absolute;
        width: .65rem;
    }
    .empty-copy { color: var(--ink-soft); font-size: .82rem; line-height: 1.7; margin: 0; }

    .model-strip { display: grid; grid-template-columns: 1fr 1fr; margin-top: .3rem; }
    .model-stat { border-top: 1px solid var(--line); padding: .75rem .25rem .7rem 0; }
    .model-stat:nth-child(even) { border-left: 1px solid var(--line); padding-left: .8rem; }
    .model-value { color: var(--ink); display: block; font-family: 'Fraunces', Georgia, serif; font-size: 1.25rem; font-weight: 700; }
    .model-label { color: var(--ink-soft); display: block; font-family: 'DM Mono', monospace; font-size: .66rem; letter-spacing: 0; line-height: 1.55; margin-top: .18rem; text-transform: uppercase; }
    .fine-print { color: var(--ink-soft); font-size: .72rem; line-height: 1.7; margin: .75rem 0 0; }

    @keyframes rise-in {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }
    [data-testid='stHorizontalBlock'] { animation: rise-in 580ms 90ms cubic-bezier(.2,.75,.25,1) both; }
    @media (min-width: 901px) {
        div[data-testid='stColumn']:has(.result-anchor) > div { position: sticky; top: 1rem; }
    }
    @media (max-width: 900px) {
        .block-container { padding: .75rem 1rem 3rem; }
        .masthead { padding-top: .8rem; }
        div[data-testid='stMarkdownContainer'] h1.hero-title {
            font-size: clamp(3rem, 15vw, 5rem) !important;
            line-height: .96 !important;
        }
        .inline-estimate { grid-template-columns: 1fr; }
        .inline-estimate-price { font-size: 2.35rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation-duration: .01ms !important; scroll-behavior: auto !important; transition-duration: .01ms !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _load_bundle():
    return load_artifacts(ARTIFACT_DIR)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _geocode(address: str) -> GeocodedAddress:
    return geocode_address(address)


try:
    artifacts = _load_bundle()
except Exception as error:
    st.error("Model artifacts are unavailable.")
    st.code(str(error))
    st.stop()


field_schema = {
    field["name"]: field
    for field in artifacts.input_schema["fields"]
}


def _default(name: str) -> Any:
    return field_schema[name]["default"]


def _options(name: str) -> list[str]:
    levels = artifacts.preprocessor.category_levels_
    if levels is None:
        raise RuntimeError("Artifact preprocessor has no category levels.")
    return levels[name]


def _select(
    name: str,
    *,
    accept_new_options: bool = False,
) -> str:
    options = _options(name)
    default = str(_default(name))
    widget_key = f"field_{name}"
    current_value = st.session_state.get(widget_key)
    if current_value is None or (
        not accept_new_options and str(current_value) not in options
    ):
        st.session_state[widget_key] = (
            default if default in options else options[0]
        )
    value = st.selectbox(
        field_schema[name]["label"],
        options=options,
        index=None,
        key=widget_key,
        accept_new_options=accept_new_options,
    )
    if value is None:
        raise RuntimeError(f"No value selected for {name}.")
    return str(value)


def _number(name: str) -> int | float:
    schema = field_schema[name]
    return st.number_input(
        schema["label"],
        value=schema["default"],
        step=schema["step"],
        key=f"field_{name}",
    )


def _normalize_whole_number_state(
    widget_key: str,
    minimum: int,
    maximum: int,
) -> None:
    try:
        normalized = normalize_whole_number(
            st.session_state[widget_key],
            minimum=minimum,
            maximum=maximum,
        )
    except (KeyError, TypeError, ValueError):
        normalized = minimum
    st.session_state[widget_key] = float(normalized)


def _whole_number(
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    schema = field_schema[name]
    widget_key = f"field_{name}"
    initial_value = st.session_state.get(widget_key, schema["default"])
    try:
        normalized = normalize_whole_number(
            initial_value,
            minimum=minimum,
            maximum=maximum,
        )
    except (TypeError, ValueError):
        normalized = minimum
    st.session_state[widget_key] = float(normalized)

    value = st.number_input(
        schema["label"],
        value=None,
        step=1.0,
        format="%.0f",
        key=widget_key,
        on_change=_normalize_whole_number_state,
        args=(widget_key, minimum, maximum),
    )
    if value is None:
        return normalized
    return normalize_whole_number(
        value,
        minimum=minimum,
        maximum=maximum,
    )


def _apply_address_suggestions(result: GeocodedAddress) -> list[str]:
    suggestions = {
        "PostalCode": result.postal_code,
        "City": result.city,
        "CountyOrParish": result.county,
        "UnifiedSchoolDistrict": result.unified_school_district,
    }
    unmatched: list[str] = []
    for name, suggestion in suggestions.items():
        if suggestion is None:
            continue
        matched = match_known_category(suggestion, _options(name))
        if matched is None:
            unmatched.append(field_schema[name]["label"])
            continue
        st.session_state[f"field_{name}"] = matched
    return unmatched


def _section_header(index: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-head">
            <div class="section-index">{escape(index)}</div>
            <div>
                <div class="section-title">{escape(title)}</div>
                <p class="section-copy">{escape(copy)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _field_group(title: str) -> None:
    st.markdown(
        f'<div class="field-group-label">{escape(title)}</div>',
        unsafe_allow_html=True,
    )


metrics = artifacts.metrics
test_metrics = metrics["test"]
row_count = metrics["rows"]["train"]
mdape = float(test_metrics["mdape_percent"])

st.markdown(
    f"""
    <div class="masthead">
        <div class="hero-kicker">Model-assisted residential valuation</div>
        <h1 class="hero-title">Read the home.<br><em>Know the value.</em></h1>
        <div class="hero-footer">
            <div class="hero-scope"><span>Model scope</span><strong>California single-family homes only</strong></div>
            <p class="hero-copy">Condominiums, townhomes, multifamily properties, land, and commercial real estate are outside this model's scope.</p>
        </div>
    </div>
    <div class="workspace-label">Valuation worksheet</div>
    """,
    unsafe_allow_html=True,
)

input_column, result_column = st.columns([1.82, 0.78], gap="large")

with input_column:
    with st.container(border=True):
        _section_header(
            "01",
            "Set the valuation frame",
            "Anchor the estimate to a date so property age and seasonal signals are calculated correctly.",
        )
        date_column, _ = st.columns([0.62, 1.38])
        with date_column:
            valuation_date = st.date_input(
                field_schema["ValuationDate"]["label"],
                value=date.today(),
                key="field_ValuationDate",
            )

    with st.container(border=True):
        _section_header(
            "02",
            "Describe the property",
            "Start with the physical facts. Use whole-property figures rather than per-unit measurements.",
        )
        property_columns = st.columns(3, gap="medium")
        with property_columns[0]:
            _field_group("Scale")
            living_area = _number("LivingArea")
            lot_size = _number("LotSizeSquareFeet")
        with property_columns[1]:
            _field_group("Plan")
            bedrooms = _whole_number(
                "BedroomsTotal",
                minimum=WHOLE_NUMBER_LIMITS["BedroomsTotal"][0],
                maximum=WHOLE_NUMBER_LIMITS["BedroomsTotal"][1],
            )
            bathrooms = _whole_number(
                "BathroomsTotalInteger",
                minimum=WHOLE_NUMBER_LIMITS["BathroomsTotalInteger"][0],
                maximum=WHOLE_NUMBER_LIMITS["BathroomsTotalInteger"][1],
            )
            stories = _whole_number(
                "Stories",
                minimum=WHOLE_NUMBER_LIMITS["Stories"][0],
                maximum=WHOLE_NUMBER_LIMITS["Stories"][1],
            )
        with property_columns[2]:
            _field_group("Vintage & access")
            year_built = _whole_number(
                "YearBuilt",
                minimum=1700,
                maximum=valuation_date.year,
            )
            garage_spaces = _whole_number(
                "GarageSpaces",
                minimum=WHOLE_NUMBER_LIMITS["GarageSpaces"][0],
                maximum=WHOLE_NUMBER_LIMITS["GarageSpaces"][1],
            )
            parking_total = _whole_number(
                "ParkingTotal",
                minimum=WHOLE_NUMBER_LIMITS["ParkingTotal"][0],
                maximum=WHOLE_NUMBER_LIMITS["ParkingTotal"][1],
            )

    with st.container(border=True):
        _section_header(
            "03",
            "Place it in the market",
            "Search a California address for coordinates and category matches, or work directly from a known coordinate pair.",
        )
        location_mode = st.segmented_control(
            "Location input",
            options=["Address", "Coordinates"],
            default="Address",
            required=True,
            key="location_mode",
        )

        resolved_address: GeocodedAddress | None = None
        if location_mode == "Address":
            address_columns = st.columns([4.4, 1], vertical_alignment="bottom")
            with address_columns[0]:
                address_query = st.text_input(
                    "Property address",
                    placeholder=DEFAULT_ADDRESS_QUERY,
                    key="address_query",
                )
            with address_columns[1]:
                locate_address = st.button(
                    "Find address",
                    icon=":material/search:",
                    use_container_width=True,
                )

            normalized_query = " ".join(address_query.split())
            lookup_query = normalized_query or DEFAULT_ADDRESS_QUERY
            if locate_address:
                try:
                    with st.spinner("Resolving market location…"):
                        lookup_result = _geocode(lookup_query)
                except GeocodingError as error:
                    st.session_state.pop("geocoded_address", None)
                    st.error(str(error))
                else:
                    st.session_state["geocoded_address"] = lookup_result
                    unmatched_fields = _apply_address_suggestions(lookup_result)
                    if unmatched_fields:
                        st.warning(
                            "Please confirm these fields manually: "
                            + ", ".join(unmatched_fields)
                            + "."
                        )

            stored_address = st.session_state.get("geocoded_address")
            if (
                isinstance(stored_address, GeocodedAddress)
                and stored_address.query_address == lookup_query
            ):
                resolved_address = stored_address
                st.markdown(
                    f"""
                    <div class="match-card">
                        <span class="match-dot"></span>
                        <div>
                            <div class="match-address">{escape(resolved_address.matched_address)}</div>
                            <div class="match-coordinates">{resolved_address.latitude:.5f} / {resolved_address.longitude:.5f}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with st.expander("Review market-area fields", expanded=True):
            location_columns = st.columns(3, gap="medium")
            with location_columns[0]:
                postal_code = _select("PostalCode")
                city = _select("City")
                county = _select("CountyOrParish")
            with location_columns[1]:
                mls_area = _select("MLSAreaMajor")
                levels = _select("Levels")
                school_district = _select(
                    "UnifiedSchoolDistrict",
                    accept_new_options=True,
                )
            with location_columns[2]:
                if location_mode == "Coordinates":
                    latitude = _number("Latitude")
                    longitude = _number("Longitude")
                elif resolved_address is not None:
                    latitude = resolved_address.latitude
                    longitude = resolved_address.longitude
                    _field_group("Resolved coordinates")
                    st.markdown(
                        f"""
                        <div class="coordinate-pair">
                            <div class="coordinate-cell"><span class="coordinate-label">Latitude</span><span class="coordinate-value">{latitude:.5f}</span></div>
                            <div class="coordinate-cell"><span class="coordinate-label">Longitude</span><span class="coordinate-value">{longitude:.5f}</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    latitude = None
                    longitude = None
                    _field_group("Resolved coordinates")
                    st.caption("Run the address search to resolve latitude and longitude.")

    with st.container(border=True):
        _section_header(
            "04",
            "Add the differentiators",
            "Mark only features that belong to the subject property today.",
        )
        amenity_columns = st.columns(5)
        amenity_values: dict[str, bool] = {}
        for column, name in zip(
            amenity_columns,
            [
                "PoolPrivateYN",
                "ViewYN",
                "AttachedGarageYN",
                "NewConstructionYN",
                "FireplaceYN",
            ],
        ):
            with column:
                amenity_values[name] = st.toggle(
                    field_schema[name]["label"],
                    value=bool(_default(name)),
                    key=f"field_{name}",
                )

    submitted = st.button(
        "Calculate indicative value",
        type="primary",
        icon=":material/arrow_forward:",
        use_container_width=True,
        disabled=(location_mode == "Address" and resolved_address is None),
    )
    estimate_summary_slot = st.empty()

    if location_mode == "Address" and resolved_address is None:
        st.caption("Resolve the property address to unlock the estimate.")

    if submitted:
        validation_error: str | None = None
        if latitude is None or longitude is None:
            validation_error = "Find the address or enter coordinates before estimating value."
        elif int(year_built) > valuation_date.year:
            validation_error = "Year built cannot be later than the valuation date."
        elif float(living_area) <= 0 or float(lot_size) <= 0:
            validation_error = "Living area and lot size must be greater than zero."
        elif min(float(bedrooms), float(bathrooms), float(stories)) < 0:
            validation_error = "Bedrooms, bathrooms, and stories cannot be negative."

        if validation_error:
            st.error(validation_error)
        else:
            property_data = {
                "ValuationDate": valuation_date.isoformat(),
                "LivingArea": living_area,
                "BedroomsTotal": bedrooms,
                "BathroomsTotalInteger": bathrooms,
                "LotSizeSquareFeet": lot_size,
                "YearBuilt": year_built,
                "GarageSpaces": garage_spaces,
                "ParkingTotal": parking_total,
                "Stories": stories,
                "Latitude": latitude,
                "Longitude": longitude,
                "PostalCode": postal_code,
                "CountyOrParish": county,
                "MLSAreaMajor": mls_area,
                "Levels": levels,
                "City": city,
                "UnifiedSchoolDistrict": school_district,
                **amenity_values,
            }
            try:
                with st.spinner("Reading the market signal…"):
                    prediction = predict_price(
                        artifacts.model,
                        property_data,
                        artifacts.preprocessor,
                    )
            except Exception as error:
                st.error("Unable to calculate an estimate for this input.")
                st.code(str(error))
            else:
                st.session_state["last_estimate"] = {
                    "prediction": float(prediction),
                    "city": city,
                    "postal_code": postal_code,
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "living_area": living_area,
                    "valuation_date": valuation_date.isoformat(),
                }
                st.toast(
                    f"Estimated sale price ready: ${float(prediction):,.0f}",
                    icon=":material/price_check:",
                    duration="long",
                )
                if school_district not in _options("UnifiedSchoolDistrict"):
                    st.warning(
                        "This school district was not present in the training data "
                        "and was treated as unknown."
                    )

    inline_estimate = st.session_state.get("last_estimate")
    if isinstance(inline_estimate, dict):
        inline_price = float(inline_estimate["prediction"])
        inline_subject = (
            f"{escape(str(inline_estimate['city']))} &middot; "
            f"{escape(str(inline_estimate['postal_code']))} &middot; "
            f"{float(inline_estimate['bedrooms']):g} bed / "
            f"{float(inline_estimate['bathrooms']):g} bath &middot; "
            f"{float(inline_estimate['living_area']):,.0f} sq ft"
        )
        estimate_summary_slot.markdown(
            f"""
            <div class="inline-estimate" role="status" aria-live="polite">
                <div>
                    <span class="inline-estimate-label">Estimated sale price</span>
                    <span class="inline-estimate-subject">{inline_subject}</span>
                    <span class="inline-estimate-note">Estimated closed sale price, not a list price.</span>
                </div>
                <strong class="inline-estimate-price">${inline_price:,.0f}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

with result_column:
    st.markdown('<div class="result-anchor"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        last_estimate = st.session_state.get("last_estimate")
        if isinstance(last_estimate, dict):
            prediction = float(last_estimate["prediction"])
            reference_low = prediction * (1 - mdape / 100)
            reference_high = prediction * (1 + mdape / 100)
            subject = (
                f"{escape(str(last_estimate['city']))} · "
                f"{escape(str(last_estimate['postal_code']))} · "
                f"{float(last_estimate['bedrooms']):g} bed / "
                f"{float(last_estimate['bathrooms']):g} bath · "
                f"{float(last_estimate['living_area']):,.0f} sq ft"
            )
            st.markdown(
                f"""
                <div class="result-kicker">Estimated sale price</div>
                <div class="estimate-price">${prediction:,.0f}</div>
                <div class="estimate-disclaimer">Estimated closed sale price, not a list price.</div>
                <div class="estimate-subject">{subject}</div>
                <div class="range-card">
                    <div class="range-label">Median-error reference band</div>
                    <div class="range-value">${reference_low:,.0f} — ${reference_high:,.0f}</div>
                </div>
                <p class="fine-print">The band applies the model's {mdape:.2f}% median absolute test error to this estimated sale price. It is context, not a formal prediction interval, list-price recommendation, or appraisal.</p>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-kicker">Estimate desk</div>
                <div class="result-title">Your market signal will land here.</div>
                <div class="empty-figure"></div>
                <p class="empty-copy">Complete the four-part worksheet, resolve the location, and calculate an indicative value. Your latest result will remain here while you refine the property.</p>
                """,
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="result-kicker">Model field notes</div>
            <div class="model-strip">
                <div class="model-stat"><span class="model-value">{test_metrics['r2']:.3f}</span><span class="model-label">Test R²</span></div>
                <div class="model-stat"><span class="model-value">{mdape:.2f}%</span><span class="model-label">Median test error</span></div>
                <div class="model-stat"><span class="model-value">{row_count:,}</span><span class="model-label">Training records</span></div>
                <div class="model-stat"><span class="model-value">v{artifacts.manifest['artifact_version']}</span><span class="model-label">Artifact version</span></div>
            </div>
            <p class="fine-print">Training window: {metrics['train_start_month']} to {metrics['train_end_month']}. Holdout test month: {metrics['test_month']}. The estimate is a model output, not a broker price opinion, lending decision, or licensed appraisal.</p>
            """,
            unsafe_allow_html=True,
        )
