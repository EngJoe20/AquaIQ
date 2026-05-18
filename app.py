"""
╔══════════════════════════════════════════════════════════════════════╗
║   AquaIQ v3 — Dual Smart Assessment & Fuzzy PID Control Simulation    ║
║   100% PDF Reports | Table+Text Rules Editor | All 5 Core Diagrams   ║
╚══════════════════════════════════════════════════════════════════════╝
pip install scikit-fuzzy numpy matplotlib flask reportlab scipy
"""
import hashlib
# ReportLab python 3.8.10 compatibility monkeypatch
try:
    hashlib.md5(usedforsecurity=False)
except TypeError:
    original_md5 = hashlib.md5
    def patched_md5(*args, **kwargs):
        kwargs.pop('usedforsecurity', None)
        return original_md5(*args, **kwargs)
    hashlib.md5 = patched_md5

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # Register 3D projection
import io, base64, json, datetime
from flask import Flask, render_template_string, request, jsonify, send_file

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ══════════════════════════════════════════════════════════════════════
# 1. WATER QUALITY ASSESSMENT DISCOURSE & MFS
# ══════════════════════════════════════════════════════════════════════
x_ph           = np.arange(0,    14.01,  0.1)
x_turbidity    = np.arange(0,   100.01,  1.0)
x_do           = np.arange(0,    20.01,  0.2)
x_temp         = np.arange(0,    50.01,  0.5)
x_conductivity = np.arange(0,  2000.01,  10.0)
x_wqi          = np.arange(0,   100.01,  1.0)

def build_mfs():
    return {
        "acidic":    fuzz.trapmf(x_ph,  [0, 0, 5.5, 6.5]),
        "neutral":   fuzz.trimf( x_ph,  [6.0, 7.0, 8.0]),
        "alkaline":  fuzz.trapmf(x_ph,  [7.5, 8.5, 14, 14]),
    }, {
        "clear":    fuzz.trapmf(x_turbidity, [0, 0, 5, 15]),
        "moderate": fuzz.trimf( x_turbidity, [10, 35, 60]),
        "cloudy":   fuzz.trapmf(x_turbidity, [50, 70, 100, 100]),
    }, {
        "low":    fuzz.trapmf(x_do, [0, 0, 3, 5]),
        "medium": fuzz.trimf( x_do, [4, 7, 10]),
        "high":   fuzz.trapmf(x_do, [8, 11, 20, 20]),
    }, {
        "cold":    fuzz.trapmf(x_temp, [0, 0, 8, 15]),
        "optimal": fuzz.trimf( x_temp, [12, 22, 30]),
        "hot":     fuzz.trapmf(x_temp, [25, 35, 50, 50]),
    }, {
        "low":    fuzz.trapmf(x_conductivity, [0, 0, 200, 400]),
        "medium": fuzz.trimf( x_conductivity, [300, 800, 1200]),
        "high":   fuzz.trapmf(x_conductivity, [1000, 1400, 2000, 2000]),
    }, {
        "very_poor":  fuzz.trapmf(x_wqi, [0, 0, 10, 25]),
        "poor":       fuzz.trimf( x_wqi, [15, 30, 45]),
        "acceptable": fuzz.trimf( x_wqi, [35, 50, 65]),
        "good":       fuzz.trimf( x_wqi, [55, 70, 85]),
        "excellent":  fuzz.trapmf(x_wqi, [75, 90, 100, 100]),
    }

WQI_DEFAULT_RULES = [
    ["neutral",  "clear",    "high",   "optimal", "low",    "excellent", "All ideal -> Excellent"],
    ["neutral",  "clear",    "high",   None,      "medium", "good",      "Near-ideal, slight mineral -> Good"],
    ["neutral",  "moderate", "high",   "optimal", None,     "good",      "Slightly cloudy but oxygenated -> Good"],
    [None,       "clear",    "medium", "optimal", "low",    "good",      "Low pollution all round -> Good"],
    ["neutral",  "clear",    "medium", "cold",    None,     "acceptable","Cold reduces O2 capacity -> Acceptable"],
    ["neutral",  "moderate", "medium", "optimal", "medium", "acceptable","Average all params -> Acceptable"],
    [None,       "clear",    "medium", "hot",     None,     "acceptable","Heat reduces DO naturally -> Acceptable"],
    ["neutral",  "moderate", "medium", "cold",    None,     "acceptable","Cold + moderate turbidity -> Acceptable"],
    ["acidic",   None,       None,     None,      None,     "poor",      "Acid water dangerous -> Poor"],
    ["alkaline", None,       None,     None,      None,     "poor",      "High pH causes scaling -> Poor"],
    [None,       "cloudy",   "low",    None,      None,     "poor",      "Suspended solids + hypoxia -> Poor"],
    [None,       "cloudy",   None,     None,      "high",   "poor",      "High TDS + turbidity -> Poor"],
    ["acidic",   "cloudy",   "low",    None,      None,     "very_poor", "Acidic + polluted + hypoxic -> Very Poor"],
    ["alkaline", None,       "low",    None,      "high",   "very_poor", "Saline alkaline hypoxic -> Very Poor"],
    [None,       "cloudy",   "low",    "hot",     None,     "very_poor", "Algal bloom scenario -> Very Poor"],
]

wqi_rules = [r[:] for r in WQI_DEFAULT_RULES]

WQI_LABELS = {
    "ph": ["acidic", "neutral", "alkaline"],
    "turbidity": ["clear", "moderate", "cloudy"],
    "do": ["low", "medium", "high"],
    "temp": ["cold", "optimal", "hot"],
    "conductivity": ["low", "medium", "high"],
    "output": ["very_poor", "poor", "acceptable", "good", "excellent"],
}

# ══════════════════════════════════════════════════════════════════════
# 2. FUZZY PID DISCOURSE & MFS (4 INPUTS, 1 OUTPUT)
# ══════════════════════════════════════════════════════════════════════
x_pid_e  = np.arange(-10, 10.01, 0.2)
x_pid_de = np.arange(-5,  5.01,  0.1)
x_pid_ie = np.arange(-20, 20.01, 0.4)
x_pid_d  = np.arange(0,   10.01, 0.2)
x_pid_u  = np.arange(0,   100.01, 1.0)

def build_pid_mfs():
    return {
        "negative": fuzz.trapmf(x_pid_e, [-10, -10, -4, 0]),
        "zero":     fuzz.trimf( x_pid_e, [-2, 0, 2]),
        "positive": fuzz.trapmf(x_pid_e, [0, 4, 10, 10]),
    }, {
        "negative": fuzz.trapmf(x_pid_de, [-5, -5, -2, 0]),
        "zero":     fuzz.trimf( x_pid_de, [-1, 0, 1]),
        "positive": fuzz.trapmf(x_pid_de, [0, 2, 5, 5]),
    }, {
        "negative": fuzz.trapmf(x_pid_ie, [-20, -20, -8, 0]),
        "zero":     fuzz.trimf( x_pid_ie, [-4, 0, 4]),
        "positive": fuzz.trapmf(x_pid_ie, [0, 8, 20, 20]),
    }, {
        "low":    fuzz.trapmf(x_pid_d, [0, 0, 2, 4]),
        "medium": fuzz.trimf( x_pid_d, [3, 5, 7]),
        "high":   fuzz.trapmf(x_pid_d, [6, 8, 10, 10]),
    }, {
        "cool_fast": fuzz.trapmf(x_pid_u, [0, 0, 15, 30]),
        "cool_slow": fuzz.trimf( x_pid_u, [20, 40, 60]),
        "maintain":  fuzz.trimf( x_pid_u, [45, 50, 55]),
        "heat_slow": fuzz.trimf( x_pid_u, [40, 60, 80]),
        "heat_fast": fuzz.trapmf(x_pid_u, [70, 85, 100, 100]),
    }

PID_DEFAULT_RULES = [
    ["positive", "zero",     None,       None,     "heat_slow", "Cold system -> Heat slow"],
    ["positive", "positive", None,       None,     "heat_fast", "Cold and falling temp -> Heat fast"],
    ["positive", "negative", None,       None,     "maintain",  "Cold but recovering -> Maintain"],
    ["negative", "zero",     None,       None,     "cool_slow", "Hot system -> Cool slow"],
    ["negative", "negative", None,       None,     "cool_fast", "Hot and rising temp -> Cool fast"],
    ["negative", "positive", None,       None,     "maintain",  "Hot but cooling down -> Maintain"],
    ["zero",     "zero",     "zero",     None,     "maintain",  "Perfect state -> Maintain"],
    ["zero",     None,       "positive", None,     "heat_slow", "Accumulated cold -> Heat slightly"],
    ["zero",     None,       "negative", None,     "cool_slow", "Accumulated heat -> Cool slightly"],
    [None,       None,       None,       "high",   "heat_slow", "High load inflow disturbance -> Heat bias"],
    [None,       None,       None,       "low",    "cool_slow", "Low disturbance -> Cool slightly"],
    ["positive", None,       "positive", None,     "heat_fast", "Both current & integral cold -> Heat fast"],
    ["negative", None,       "negative", None,     "cool_fast", "Both current & integral hot -> Cool fast"],
    ["zero",     "positive", None,       "high",   "heat_slow", "Inflow cold and temp falling -> Heat correction"],
    ["zero",     "negative", None,       "low",    "cool_slow", "Inflow warm and temp rising -> Cool correction"],
]

pid_rules = [r[:] for r in PID_DEFAULT_RULES]

PID_LABELS = {
    "error": ["negative", "zero", "positive"],
    "change_error": ["negative", "zero", "positive"],
    "int_error": ["negative", "zero", "positive"],
    "disturbance": ["low", "medium", "high"],
    "output": ["cool_fast", "cool_slow", "maintain", "heat_slow", "heat_fast"],
}

# Caches for heavy 3D surface plot renderings
_cached_wqi_surface = None
_cached_pid_surface = None

# ══════════════════════════════════════════════════════════════════════
# 3. CONTROL SYSTEMS BUILDERS
# ══════════════════════════════════════════════════════════════════════
def build_ctrl_system():
    mf_ph, mf_turb, mf_do, mf_temp, mf_cond, mf_wqi = build_mfs()
    ph   = ctrl.Antecedent(x_ph,   'ph')
    turb = ctrl.Antecedent(x_turbidity, 'turbidity')
    do   = ctrl.Antecedent(x_do,   'do_level')
    temp = ctrl.Antecedent(x_temp, 'temperature')
    cond = ctrl.Antecedent(x_conductivity, 'conductivity')
    wqi  = ctrl.Consequent(x_wqi,  'wqi', defuzzify_method='centroid')

    for k, v in mf_ph.items(): ph[k] = v
    for k, v in mf_turb.items(): turb[k] = v
    for k, v in mf_do.items(): do[k] = v
    for k, v in mf_temp.items(): temp[k] = v
    for k, v in mf_cond.items(): cond[k] = v
    for k, v in mf_wqi.items(): wqi[k] = v

    ctrl_rules = []
    for r in wqi_rules:
        ants = []
        if r[0]: ants.append(ph[r[0]])
        if r[1]: ants.append(turb[r[1]])
        if r[2]: ants.append(do[r[2]])
        if r[3]: ants.append(temp[r[3]])
        if r[4]: ants.append(cond[r[4]])
        if ants:
            combined = ants[0]
            for a in ants[1:]:
                combined = combined & a
            ctrl_rules.append(ctrl.Rule(combined, wqi[r[5]]))
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(ctrl_rules)), mf_ph, mf_turb, mf_do, mf_temp, mf_cond, mf_wqi

def build_pid_system():
    mf_e, mf_de, mf_ie, mf_d, mf_u = build_pid_mfs()
    e  = ctrl.Antecedent(x_pid_e,  'error')
    de = ctrl.Antecedent(x_pid_de, 'change_error')
    ie = ctrl.Antecedent(x_pid_ie, 'int_error')
    d  = ctrl.Antecedent(x_pid_d,  'disturbance')
    u  = ctrl.Consequent(x_pid_u,  'control_output', defuzzify_method='centroid')

    for k, v in mf_e.items(): e[k] = v
    for k, v in mf_de.items(): de[k] = v
    for k, v in mf_ie.items(): ie[k] = v
    for k, v in mf_d.items(): d[k] = v
    for k, v in mf_u.items(): u[k] = v

    ctrl_rules = []
    for r in pid_rules:
        ants = []
        if r[0]: ants.append(e[r[0]])
        if r[1]: ants.append(de[r[1]])
        if r[2]: ants.append(ie[r[2]])
        if r[3]: ants.append(d[r[3]])
        if ants:
            combined = ants[0]
            for a in ants[1:]:
                combined = combined & a
            ctrl_rules.append(ctrl.Rule(combined, u[r[4]]))
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(ctrl_rules)), mf_e, mf_de, mf_ie, mf_d, mf_u

# ══════════════════════════════════════════════════════════════════════
# 4. CHART UTILITIES & GENERATORS (ALL 5 TYPES)
# ══════════════════════════════════════════════════════════════════════
M3_BLUE     = "#00a3ff"
CHART_BG    = "#f8fafd"
CHART_FACE  = "#ffffff"
GRID_COLOR  = "#00000014"

WQI_COLORS_IN = {"acidic":"#ba1a1a","neutral":"#386a20","alkaline":"#b09fff","clear":"#386a20","moderate":"#d56e00","cloudy":"#ba1a1a","low":"#ba1a1a","medium":"#d56e00","high":"#386a20","cold":"#00a3ff","optimal":"#386a20","hot":"#ba1a1a"}
WQI_COLORS_OUT = {"very_poor":"#ba1a1a","poor":"#d56e00","acceptable":"#e3b000","good":"#386a20","excellent":"#137333"}

PID_COLORS_IN = {"negative":"#ba1a1a","zero":"#386a20","positive":"#00a3ff","low":"#386a20","medium":"#d56e00","high":"#ba1a1a"}
PID_COLORS_OUT = {"cool_fast":"#002147","cool_slow":"#00a3ff","maintain":"#386a20","heat_slow":"#d56e00","heat_fast":"#ba1a1a"}

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Type 1: Membership Functions (Inputs / Outputs)
def chart_mf_input(x, mf_dict, title, xlabel, current_val=None, colors_dict=WQI_COLORS_IN):
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    fig.patch.set_facecolor(CHART_FACE); ax.set_facecolor(CHART_BG)
    for label, arr in mf_dict.items():
        c = colors_dict.get(label, M3_BLUE)
        ax.fill_between(x, 0, arr, alpha=0.15, color=c)
        ax.plot(x, arr, color=c, lw=2, label=label.replace('_',' ').title())
    if current_val is not None:
        ax.axvline(current_val, color=M3_BLUE, lw=1.8, linestyle='--', label=f'Now: {current_val:.1f}')
    ax.set_title(title, fontsize=10, fontweight='bold', color='#1f1f1f')
    ax.legend(fontsize=7, framealpha=.9, facecolor='#edf2fa', edgecolor='#00000014', loc='upper right', labelcolor='#1f1f1f')
    ax.set_ylim(0, 1.1); ax.grid(True, ls=':', color=GRID_COLOR)
    ax.tick_params(colors='#444746', which='both', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_COLOR)
    plt.tight_layout()
    return fig_to_b64(fig)

def chart_output_mf(x, mf_dict, val, title, xlabel, colors_dict):
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    fig.patch.set_facecolor(CHART_FACE); ax.set_facecolor(CHART_BG)
    for label, arr in mf_dict.items():
        c = colors_dict.get(label, M3_BLUE)
        ax.fill_between(x, 0, arr, alpha=0.18, color=c)
        ax.plot(x, arr, color=c, lw=2, label=label.replace('_',' ').title())
    ax.axvline(val, color=M3_BLUE, lw=2.5, linestyle='--', label=f'Centroid: {val:.2f}')
    ax.fill_betweenx([0, 1], val-0.8, val+0.8, color=M3_BLUE, alpha=0.25)
    ax.set_title(title, fontsize=11, fontweight='bold', color='#1f1f1f')
    ax.set_xlabel(xlabel, fontsize=9, color='#444746')
    ax.legend(fontsize=8, facecolor='#edf2fa', edgecolor='#00000014', loc='upper left', labelcolor='#1f1f1f')
    ax.set_ylim(0, 1.1); ax.grid(True, ls=':', color=GRID_COLOR)
    ax.tick_params(colors='#444746', which='both', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_COLOR)
    plt.tight_layout()
    return fig_to_b64(fig)

# Type 2: Rule Firing Strengths
def chart_rule_firing(firing_list, colors_dict):
    n = len(firing_list)
    fig, ax = plt.subplots(figsize=(8.5, max(3, n * 0.35)))
    fig.patch.set_facecolor(CHART_FACE); ax.set_facecolor(CHART_BG)
    labels     = [f"R{i+1}: {f[2][:45]}" for i, f in enumerate(firing_list)]
    values     = [f[0] for f in firing_list]
    bar_colors = [colors_dict.get(f[1], M3_BLUE) for f in firing_list]
    bars = ax.barh(range(n), values, color=bar_colors, alpha=0.82, height=0.6)
    for bar, v in zip(bars, values):
        if v > 0.01:
            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2, f'{v:.2f}', va='center', fontsize=7, fontweight='bold', color='#1f1f1f')
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7.5, color='#1f1f1f')
    ax.set_xlim(0, 1.15)
    ax.set_title('Fuzzy Inference — Rule Firing Strengths', fontsize=11, fontweight='bold', color='#1f1f1f')
    ax.grid(True, ls=':', axis='x', color=GRID_COLOR)
    ax.tick_params(colors='#444746', which='both', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_COLOR)
    plt.tight_layout()
    return fig_to_b64(fig)

# Type 3: Radar Chart
def chart_radar(raw_values, max_ranges, labels, title):
    norm = [v / m for v, m in zip(raw_values, max_ranges)]
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values = norm + norm[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(CHART_FACE); ax.set_facecolor(CHART_BG)
    ax.plot(angles, values, color=M3_BLUE, lw=2)
    ax.fill(angles, values, color=M3_BLUE, alpha=0.15)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=8, color='#1f1f1f')
    ax.set_ylim(0, 1)
    ax.grid(color=GRID_COLOR, ls=':')
    ax.tick_params(colors='#444746', labelsize=7)
    ax.set_title(title, fontsize=11, fontweight='bold', color='#1f1f1f', pad=15)
    for angle, val, rv in zip(angles[:-1], norm, raw_values):
        ax.annotate(f'{rv:.1f}', xy=(angle, val), xytext=(angle, val+0.08), ha='center', fontsize=8, color=M3_BLUE, fontweight='bold')
    plt.tight_layout()
    return fig_to_b64(fig)

# Type 4: Correlation / Membership Breakdowns
def chart_correlation(var_breakdowns, colors_dict, title):
    n_vars = len(var_breakdowns)
    fig, axes = plt.subplots(1, n_vars, figsize=(2.3 * n_vars, 3.8))
    fig.patch.set_facecolor(CHART_FACE)
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02, color='#1f1f1f')
    
    for ax, (name, memberships, cur_val, unit_str) in zip(axes, var_breakdowns):
        ax.set_facecolor(CHART_BG)
        labels = list(memberships.keys())
        vals   = list(memberships.values())
        bar_cs = [colors_dict.get(l, M3_BLUE) for l in labels]
        bars   = ax.bar(labels, vals, color=bar_cs, alpha=0.85, width=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold', color='#1f1f1f')
        ax.set_ylim(0, 1.25)
        ax.set_title(f'{name}\n({cur_val:.1f} {unit_str})', fontsize=8, fontweight='bold', color='#1f1f1f')
        ax.grid(True, ls=':', axis='y', color=GRID_COLOR)
        ax.tick_params(colors='#444746', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID_COLOR)
    plt.tight_layout()
    return fig_to_b64(fig)

# Type 5: 3D Surface Plots
def chart_surface_wqi():
    ph_range = np.linspace(0, 14, 15)
    turb_range = np.linspace(0, 100, 15)
    Z = np.zeros((len(turb_range), len(ph_range)))
    sim, *_ = build_ctrl_system()

    for i, t in enumerate(turb_range):
        for j, p in enumerate(ph_range):
            try:
                sim.input['ph'] = p; sim.input['turbidity'] = t
                sim.input['do_level'] = 9.0; sim.input['temperature'] = 22.0
                sim.input['conductivity'] = 300.0
                sim.compute()
                Z[i, j] = float(np.clip(sim.output['wqi'], 0, 100))
            except Exception:
                Z[i, j] = 50.0

    fig = plt.figure(figsize=(7, 5))
    fig.patch.set_facecolor(CHART_FACE)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(CHART_BG)
    PH, TURB = np.meshgrid(ph_range, turb_range)
    cmap = LinearSegmentedColormap.from_list('wqi_c', ['#ba1a1a','#d56e00','#e3b000','#386a20','#137333'])
    surf = ax.plot_surface(PH, TURB, Z, cmap=cmap, alpha=0.88, linewidth=0)
    
    # Theme 3D panes for modern Dark Mode
    ax.xaxis.set_pane_color((0.95, 0.96, 0.98, 1.0))
    ax.yaxis.set_pane_color((0.95, 0.96, 0.98, 1.0))
    ax.zaxis.set_pane_color((0.95, 0.96, 0.98, 1.0))
    ax.xaxis._axinfo["grid"]['color'] = "#00000014"
    ax.yaxis._axinfo["grid"]['color'] = "#00000014"
    ax.zaxis._axinfo["grid"]['color'] = "#00000014"
    
    cb = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cb.ax.yaxis.set_tick_params(colors='#1f1f1f')
    
    ax.set_xlabel('pH', color='#1f1f1f'); ax.set_ylabel('Turbidity', color='#1f1f1f'); ax.set_zlabel('WQI', color='#1f1f1f')
    ax.tick_params(colors='#444746', labelsize=8)
    ax.set_title('WQI Surface: pH x Turbidity -> WQI\n(DO=9, Temp=22, Cond=300)', fontsize=9, fontweight='bold', color='#1f1f1f')
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    return fig_to_b64(fig)

def chart_surface_pid():
    e_range = np.linspace(-10, 10, 15)
    de_range = np.linspace(-5, 5, 15)
    Z = np.zeros((len(de_range), len(e_range)))
    sim, *_ = build_pid_system()

    for i, de_val in enumerate(de_range):
        for j, e_val in enumerate(e_range):
            try:
                sim.input['error'] = e_val
                sim.input['change_error'] = de_val
                sim.input['int_error'] = 0.0
                sim.input['disturbance'] = 2.0
                sim.compute()
                Z[i, j] = float(np.clip(sim.output['control_output'], 0, 100))
            except Exception:
                Z[i, j] = 50.0

    fig = plt.figure(figsize=(7, 5))
    fig.patch.set_facecolor(CHART_FACE)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(CHART_BG)
    E_m, DE_m = np.meshgrid(e_range, de_range)
    cmap = LinearSegmentedColormap.from_list('pid_c', ['#002147','#0062a3','#386a20','#d56e00','#ba1a1a'])
    surf = ax.plot_surface(E_m, DE_m, Z, cmap=cmap, alpha=0.88, linewidth=0)
    
    # Theme 3D panes for modern Dark Mode
    ax.xaxis.set_pane_color((0.95, 0.96, 0.98, 1.0))
    ax.yaxis.set_pane_color((0.95, 0.96, 0.98, 1.0))
    ax.zaxis.set_pane_color((0.95, 0.96, 0.98, 1.0))
    ax.xaxis._axinfo["grid"]['color'] = "#00000014"
    ax.yaxis._axinfo["grid"]['color'] = "#00000014"
    ax.zaxis._axinfo["grid"]['color'] = "#00000014"
    
    cb = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cb.ax.yaxis.set_tick_params(colors='#1f1f1f')
    
    ax.set_xlabel('Error (e)', color='#1f1f1f'); ax.set_ylabel('dError (de)', color='#1f1f1f'); ax.set_zlabel('Output (u)', color='#1f1f1f')
    ax.tick_params(colors='#444746', labelsize=8)
    ax.set_title('PID Surface: Error x dError -> Output\n(ie=0, d=2)', fontsize=9, fontweight='bold', color='#1f1f1f')
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    return fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════
# 5. MAMDANI EVALUATE ENGINE
# ══════════════════════════════════════════════════════════════════════
def get_category(v):
    if v >= 80:   return "Excellent",  "No treatment required. Safe for all uses.",                       "#137333"
    elif v >= 60: return "Good",       "Minor filtration recommended. Suitable for most uses.",           "#386a20"
    elif v >= 40: return "Acceptable", "Standard treatment (filtration + chlorination) required.",        "#e3b000"
    elif v >= 20: return "Poor",       "Advanced treatment: multi-stage filtration + UV disinfection.",   "#d56e00"
    else:         return "Very Poor",  "Severe: reverse osmosis + chemical correction + monitoring.",    "#ba1a1a"

def evaluate_wqi(ph_val, turb_val, do_val, temp_val, cond_val):
    sim, mf_ph, mf_turb, mf_do, mf_temp, mf_cond, mf_wqi = build_ctrl_system()
    sim.input['ph']           = float(ph_val)
    sim.input['turbidity']    = float(turb_val)
    sim.input['do_level']     = float(do_val)
    sim.input['temperature']  = float(temp_val)
    sim.input['conductivity'] = float(cond_val)
    
    try:
        sim.compute()
        wqi_val = float(np.clip(sim.output['wqi'], 0, 100))
    except Exception:
        wqi_val = 50.0  # Safe fallback
        
    category, rec, color = get_category(wqi_val)

    # Compute rule firing strengths
    def get_mu(univ, mf_dict, label, val):
        if not label: return 1.0
        return float(fuzz.interp_membership(univ, mf_dict[label], val))

    firing = []
    for r in wqi_rules:
        s = min(
            get_mu(x_ph,           mf_ph,   r[0], ph_val),
            get_mu(x_turbidity,    mf_turb, r[1], turb_val),
            get_mu(x_do,           mf_do,   r[2], do_val),
            get_mu(x_temp,         mf_temp, r[3], temp_val),
            get_mu(x_conductivity, mf_cond, r[4], cond_val)
        )
        firing.append((round(s, 3), r[5], r[6]))

    # Input breakdowns
    breakdowns = [
        ("pH",           {k: get_mu(x_ph,           mf_ph,   k, ph_val) for k in mf_ph},     ph_val,   ""),
        ("Turbidity",    {k: get_mu(x_turbidity,    mf_turb, k, turb_val) for k in mf_turb}, turb_val, "NTU"),
        ("Dissolved O2", {k: get_mu(x_do,           mf_do,   k, do_val) for k in mf_do},     do_val,   "mg/L"),
        ("Temperature",  {k: get_mu(x_temp,         mf_temp, k, temp_val) for k in mf_temp}, temp_val, "°C"),
        ("Conductivity", {k: get_mu(x_conductivity, mf_cond, k, cond_val) for k in mf_cond}, cond_val, "µS"),
    ]

    global _cached_wqi_surface
    if _cached_wqi_surface is None:
        _cached_wqi_surface = chart_surface_wqi()

    charts = {
        "output_mf":   chart_output_mf(x_wqi, mf_wqi, wqi_val, 'Output MF — Defuzzified Centroid', 'Water Quality Index', WQI_COLORS_OUT),
        "rule_firing": chart_rule_firing(firing, WQI_COLORS_OUT),
        "radar":       chart_radar([ph_val, turb_val, do_val, temp_val, cond_val], [14, 100, 20, 50, 2000], ["pH", "Turbidity", "DO", "Temp", "Cond"], "Water Quality Parameters (Normalized)"),
        "correlation": chart_correlation(breakdowns, WQI_COLORS_IN, "Input Membership Breakdowns"),
        "surface":     _cached_wqi_surface,
        "mf_ph":       chart_mf_input(x_ph,           mf_ph,   "pH MFs", "pH", ph_val, WQI_COLORS_IN),
        "mf_turbidity":chart_mf_input(x_turbidity,    mf_turb, "Turbidity MFs", "NTU", turb_val, WQI_COLORS_IN),
        "mf_do":       chart_mf_input(x_do,           mf_do,   "DO MFs", "mg/L", do_val, WQI_COLORS_IN),
        "mf_temp":     chart_mf_input(x_temp,         mf_temp, "Temperature MFs", "°C", temp_val, WQI_COLORS_IN),
        "mf_conductivity": chart_mf_input(x_conductivity, mf_cond, "Conductivity MFs", "µS/cm", cond_val, WQI_COLORS_IN),
    }

    return {
        "wqi": round(wqi_val, 2), "category": category, "recommendation": rec, "color": color,
        "firing": firing, "inputs": {"pH": ph_val, "Turbidity": turb_val, "Dissolved O2": do_val, "Temperature": temp_val, "Conductivity": cond_val},
        "charts": charts
    }

# ══════════════════════════════════════════════════════════════════════
# 6. FUZZY PID SIMULATOR LOOP
# ══════════════════════════════════════════════════════════════════════
def run_pid_simulation(setpoint, init_temp, disturbance, duration=60):
    sim, mf_e, mf_de, mf_ie, mf_d, mf_u = build_pid_system()
    
    alpha, beta, gamma = 0.03, 0.40, 0.04
    T_ambient, T_inlet = 20.0, 15.0
    
    t_h, sp_h, pv_h, u_h, d_h = [], [], [], [], []
    e_h, de_h, ie_h = [], [], []
    
    cur_temp = float(init_temp)
    prev_err = float(setpoint - cur_temp)
    accum_err = 0.0
    
    for t in range(duration):
        err = float(setpoint - cur_temp)
        derr = float(err - prev_err)
        accum_err = float(np.clip(accum_err + err, -20.0, 20.0))
        
        # Feed FLC
        try:
            sim.input['error'] = float(np.clip(err, -10, 10))
            sim.input['change_error'] = float(np.clip(derr, -5, 5))
            sim.input['int_error'] = float(np.clip(accum_err, -20, 20))
            sim.input['disturbance'] = float(np.clip(disturbance, 0, 10))
            sim.compute()
            u_val = float(np.clip(sim.output['control_output'], 0.0, 100.0))
        except Exception:
            u_val = 50.0 if err > 0 else 10.0  # Fallback bang-bang
            
        # Physics update
        next_temp = cur_temp + alpha * (T_ambient - cur_temp) + beta * (u_val / 10.0) - gamma * disturbance * (cur_temp - T_inlet)
        
        # Save
        t_h.append(t); sp_h.append(round(setpoint, 2)); pv_h.append(round(cur_temp, 2)); u_h.append(round(u_val, 2))
        d_h.append(round(disturbance, 2)); e_h.append(round(err, 2)); de_h.append(round(derr, 2)); ie_h.append(round(accum_err, 2))
        
        cur_temp = next_temp
        prev_err = err
        
    return {"t": t_h, "sp": sp_h, "pv": pv_h, "u": u_h, "d": d_h, "e": e_h, "de": de_h, "ie": ie_h}

def generate_pid_timestep_charts(e_val, de_val, ie_val, d_val, u_val):
    _, mf_e, mf_de, mf_ie, mf_d, mf_u = build_pid_system()
    
    def get_mu(univ, mf_dict, label, val):
        if not label: return 1.0
        return float(fuzz.interp_membership(univ, mf_dict[label], val))

    firing = []
    for r in pid_rules:
        s = min(
            get_mu(x_pid_e,  mf_e,  r[0], e_val),
            get_mu(x_pid_de, mf_de, r[1], de_val),
            get_mu(x_pid_ie, mf_ie, r[2], ie_val),
            get_mu(x_pid_d,  mf_d,  r[3], d_val)
        )
        firing.append((round(s, 3), r[4], r[5]))

    breakdowns = [
        ("Error",          {k: get_mu(x_pid_e,  mf_e,  k, e_val) for k in mf_e},   e_val,  ""),
        ("Change Error",   {k: get_mu(x_pid_de, mf_de, k, de_val) for k in mf_de}, de_val, "/s"),
        ("Integral Error", {k: get_mu(x_pid_ie, mf_ie, k, ie_val) for k in mf_ie}, ie_val, "*s"),
        ("Disturbance",    {k: get_mu(x_pid_d,  mf_d,  k, d_val) for k in mf_d},   d_val,  "L/m"),
    ]

    global _cached_pid_surface
    if _cached_pid_surface is None:
        _cached_pid_surface = chart_surface_pid()

    return {
        "output_mf":   chart_output_mf(x_pid_u, mf_u, u_val, 'Output MF — Defuzzified Centroid', 'Heater Control Power (%)', PID_COLORS_OUT),
        "rule_firing": chart_rule_firing(firing, PID_COLORS_OUT),
        "radar":       chart_radar([e_val, de_val, ie_val, d_val], [10, 5, 20, 10], ["Error", "dError", "intError", "Disturb"], "PID Inputs (Normalized)"),
        "correlation": chart_correlation(breakdowns, PID_COLORS_IN, "PID Input Breakdowns"),
        "surface":     _cached_pid_surface,
        "mf_e":        chart_mf_input(x_pid_e,  mf_e,  "Error MFs", "Error", e_val, PID_COLORS_IN),
        "mf_de":       chart_mf_input(x_pid_de, mf_de, "Change of Error MFs", "dError", de_val, PID_COLORS_IN),
        "mf_ie":       chart_mf_input(x_pid_ie, mf_ie, "Integral Error MFs", "intError", ie_val, PID_COLORS_IN),
        "mf_d":        chart_mf_input(x_pid_d,  mf_d,  "Disturbance MFs", "Disturbance", d_val, PID_COLORS_IN),
    }

# ══════════════════════════════════════════════════════════════════════
# 7. PROFESSIONAL PDF REPORT BUILDERS (PDF ONLY!)
# ══════════════════════════════════════════════════════════════════════
def embed_rl_chart(b64, w=14*cm, h=6*cm):
    return RLImage(io.BytesIO(base64.b64decode(b64)), width=w, height=h)

def make_wqi_pdf(res):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5*cm, rightMargin=1.5*cm, topMargin=2*cm, bottomMargin=2*cm, title="AquaIQ Water Assessment")
    sty = getSampleStyleSheet()
    
    t_sty = ParagraphStyle('T', parent=sty['Title'], fontSize=20, textColor=colors.HexColor('#0078d4'), spaceAfter=5)
    s_sty = ParagraphStyle('S', parent=sty['Heading3'], fontSize=11, textColor=colors.HexColor('#605e5c'), spaceAfter=15, alignment=TA_CENTER)
    h_sty = ParagraphStyle('H', parent=sty['Heading2'], fontSize=12, textColor=colors.HexColor('#0078d4'), spaceBefore=12, spaceAfter=6)
    b_sty = ParagraphStyle('B', parent=sty['Normal'], fontSize=9.5, leading=14, spaceAfter=5)
    
    story = [
        Paragraph("AquaIQ Smart Water Assessment", t_sty),
        Paragraph("Advanced Water Quality Diagnosis & Inference Report", s_sty),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0078d4'), spaceAfter=10),
        Paragraph(f"<b>Assessment Timestamp:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", b_sty),
        Spacer(1, 0.2*cm)
    ]

    # Metrics
    summary = [
        ["Diagnostics Parameter", "Assessment Rating / Recommendations"],
        ["Water Quality Index (WQI)", f"{res['wqi']:.2f} / 100"],
        ["Inference Category", res['category']],
        ["Treatment Guidelines", res['recommendation']]
    ]
    t = Table(summary, colWidths=[6*cm, 12*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0078d4')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e1dfdd')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('PADDING', (0,0), (-1,-1), 6),
        ('TEXTCOLOR', (1,2), (1,2), colors.HexColor(res['color'])),
        ('FONTNAME', (1,1), (1,-1), 'Helvetica-Bold'),
    ]))
    story.append(t); story.append(Spacer(1, 0.4*cm))

    # Inputs
    inputs_t = [
        ["Sensor Parameter", "Value", "Metric Unit", "Valid Operational Bounds"],
        ["pH Acidity Level", f"{res['inputs']['pH']:.2f}", "pH scale", "0.0 - 14.0"],
        ["Turbidity (Clarity)", f"{res['inputs']['Turbidity']:.2f}", "NTU", "0.0 - 100.0"],
        ["Dissolved Oxygen", f"{res['inputs']['Dissolved O2']:.2f}", "mg/L", "0.0 - 20.0"],
        ["Temperature", f"{res['inputs']['Temperature']:.2f}", "°C", "0.0 - 50.0"],
        ["Conductivity (Salinity)", f"{res['inputs']['Conductivity']:.2f}", "µS/cm", "0.0 - 2000.0"],
    ]
    ti = Table(inputs_t, colWidths=[5.5*cm, 3*cm, 3.5*cm, 6*cm])
    ti.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#005a9e')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e1dfdd')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(ti); story.append(PageBreak())

    # Visualizations
    story.append(Paragraph("Diagnostics Charts Breakdown", h_sty))
    story.append(Paragraph("<b>Output Membership Function (Defuzzified Result):</b>", b_sty))
    story.append(embed_rl_chart(res['charts']['output_mf'], 16*cm, 5.5*cm))
    story.append(Spacer(1, 0.3*cm))
    
    story.append(Paragraph("<b>Correlation & Input breakdowns:</b>", b_sty))
    story.append(embed_rl_chart(res['charts']['correlation'], 16*cm, 5.2*cm))
    story.append(PageBreak())

    story.append(Paragraph("<b>3D Control Surface Mapping:</b>", b_sty))
    story.append(embed_rl_chart(res['charts']['surface'], 12*cm, 8.5*cm))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("<b>Active Inputs Radar Representation:</b>", b_sty))
    story.append(embed_rl_chart(res['charts']['radar'], 8*cm, 8*cm))
    story.append(PageBreak())

    story.append(Paragraph("Fuzzy Rule Inference Map", h_sty))
    story.append(embed_rl_chart(res['charts']['rule_firing'], 16*cm, 8.5*cm))
    story.append(Spacer(1, 0.5*cm))

    # Methodology
    story.append(Paragraph("Mamdani Inference Methodology", h_sty))
    story.append(Paragraph(
        "Fuzzy assessment evaluates water quality across highly non-linear sensor states using a Mamdani control system. "
        "Each active rule combines antecedents with a MIN-AND operator, yielding scaled output distributions. "
        "Centroid defuzzification solves the balance of gravity mathematically to pinpoint a crisp index score from 0 to 100:", b_sty
    ))
    story.append(Paragraph("<i>z* = ∫(μ_C(z) * z) dz / ∫(μ_C(z)) dz</i>", ParagraphStyle('code', parent=sty['Code'], backColor=colors.HexColor('#f3f2f1'), borderPad=6)))
    
    doc.build(story)
    buf.seek(0)
    return buf.read()

def make_pid_pdf(res):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5*cm, rightMargin=1.5*cm, topMargin=2*cm, bottomMargin=2*cm, title="AquaIQ PID Controller Simulation Report")
    sty = getSampleStyleSheet()
    
    t_sty = ParagraphStyle('T', parent=sty['Title'], fontSize=20, textColor=colors.HexColor('#4f46e5'), spaceAfter=5)
    s_sty = ParagraphStyle('S', parent=sty['Heading3'], fontSize=11, textColor=colors.HexColor('#6b7280'), spaceAfter=15, alignment=TA_CENTER)
    h_sty = ParagraphStyle('H', parent=sty['Heading2'], fontSize=12, textColor=colors.HexColor('#4f46e5'), spaceBefore=12, spaceAfter=6)
    b_sty = ParagraphStyle('B', parent=sty['Normal'], fontSize=9.5, leading=14, spaceAfter=5)
    
    story = [
        Paragraph("AquaIQ Fuzzy PID Control Simulation", t_sty),
        Paragraph("Interactive 4-Input Thermal Process Controller Diagnostics", s_sty),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#4f46e5'), spaceAfter=10),
        Paragraph(f"<b>Report Timestamp:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", b_sty),
        Spacer(1, 0.2*cm)
    ]

    # Metrics computation
    pv = res['sim']['pv']
    sp = res['sim']['sp'][0]
    final_val = pv[-1]
    overshoot = max(0.0, max(pv) - sp)
    steady_err = abs(sp - final_val)
    
    summary = [
        ["Controller Simulation Metric", "Value"],
        ["Target Setpoint Temperature", f"{sp:.1f} °C"],
        ["Initial System Temperature", f"{res['initial_temp']:.1f} °C"],
        ["Disturbance Flow Rate", f"{res['disturbance']:.1f} L/min"],
        ["Final Settled Temperature (t=60)", f"{final_val:.2f} °C"],
        ["Transient Peak Overshoot", f"{overshoot:.2f} °C"],
        ["Steady State Tracking Error", f"{steady_err:.2f} °C"],
    ]
    t = Table(summary, colWidths=[8*cm, 10*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4f46e5')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t); story.append(Spacer(1, 0.4*cm))

    # Dynamic Simulation response plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5), sharex=True)
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#f9fafb'); ax2.set_facecolor('#f9fafb')
    ax1.plot(res['sim']['t'], res['sim']['sp'], color='#ef4444', lw=1.5, ls='--', label='Setpoint')
    ax1.plot(res['sim']['t'], res['sim']['pv'], color='#4f46e5', lw=2, label='Water Temperature (PV)')
    ax1.set_ylabel('Temp (°C)', fontsize=8); ax1.legend(loc='lower right', fontsize=8); ax1.grid(True, ls=':')
    ax2.plot(res['sim']['t'], res['sim']['u'], color='#10b981', lw=1.8, label='Heater Signal (u)')
    ax2.set_ylabel('Heater Power (%)', fontsize=8); ax2.set_xlabel('Time (Seconds)', fontsize=8); ax2.legend(loc='upper right', fontsize=8); ax2.grid(True, ls=':')
    plt.tight_layout()
    b64_response = fig_to_b64(fig)
    
    story.append(Paragraph("System Closed-Loop Response Curve", h_sty))
    story.append(embed_rl_chart(b64_response, 16*cm, 8*cm))
    story.append(PageBreak())

    # Visualizations
    story.append(Paragraph("Fuzzy Controller Step Diagnostics (t=15)", h_sty))
    story.append(Paragraph("<b>Output Control Power Distribution:</b>", b_sty))
    story.append(embed_rl_chart(res['charts']['output_mf'], 16*cm, 5.5*cm))
    story.append(Spacer(1, 0.3*cm))
    
    story.append(Paragraph("<b>PID Membership Correlation Breakdown:</b>", b_sty))
    story.append(embed_rl_chart(res['charts']['correlation'], 16*cm, 5.2*cm))
    story.append(PageBreak())

    story.append(Paragraph("<b>3D Control Surface Mapping (Error x dError -> Output):</b>", b_sty))
    story.append(embed_rl_chart(res['charts']['surface'], 12*cm, 8.5*cm))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("<b>Active Normalized Input Radar representation:</b>", b_sty))
    story.append(embed_rl_chart(res['charts']['radar'], 8*cm, 8*cm))
    story.append(PageBreak())

    story.append(Paragraph("Fuzzy Rule Inference Map (t=15)", h_sty))
    story.append(embed_rl_chart(res['charts']['rule_firing'], 16*cm, 8.5*cm))
    story.append(Spacer(1, 0.5*cm))

    # Methodology
    story.append(Paragraph("Fuzzy PID Control & Thermal Dynamics System Model", h_sty))
    story.append(Paragraph(
        "Fuzzy PID controller regulates thermal levels using a robust 4-input Mamdani FLC scheme. "
        "Discrete error states, derivatives, and integral actions are fuzzified, evaluated dynamically, "
        "and defuzzified. The system response follows standard thermodynamic energy balance physics equations:", b_sty
    ))
    story.append(Paragraph("<i>T(k+1) = T(k) + α*(T_ambient - T(k)) + β*(u(k)/10) - γ*d(k)*(T(k) - T_inlet)</i>", ParagraphStyle('code', parent=sty['Code'], backColor=colors.HexColor('#eef2ff'), borderPad=6)))
    
    doc.build(story)
    buf.seek(0)
    return buf.read()

# ══════════════════════════════════════════════════════════════════════
# 8. FLASK SERVER & INTEGRATED HTML INTERFACE
# ══════════════════════════════════════════════════════════════════════
app = Flask(__name__)
_last_wqi_assessment = {}
_last_pid_simulation = {}

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>AquaIQ v3 — Fuzzy Assessment & PID Control System</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Roboto:wght@300;400;500;700&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap" rel="stylesheet" />
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
<style>
:root{
  --accent:#0b57d0; /* M3 Primary Light */
  --accent-dark:#0842a0;
  --accent-deep:#001d35;
  --accent-light:rgba(11, 87, 208, 0.12);
  --accent-lighter:rgba(11, 87, 208, 0.05);
  --accent-gradient:linear-gradient(135deg,#0b57d0,#4285f4 45%,#001d35);
  
  --s00:#ffffff; /* Deep surface */
  --s01:#f8fafd; /* Darker body */
  --s02:#edf2fa; /* Elevated components */
  --s03:#e1eafd; /* High elevated */
  --border:rgba(0,0,0,0.08);
  --border-s:rgba(0,0,0,0.12);
  --text:#1f1f1f;
  --text2:#444746;
  --text3:#747775;
  
  --sh2: 0 4px 12px -2px rgba(0,0,0,0.4), inset 0 1px 0 rgba(0,0,0,0.05);
  --sh8: 0 20px 40px -8px rgba(0,0,0,0.5), inset 0 1px 0 rgba(0,0,0,0.1);
  --r4: 6px; --r8: 16px; --r12: 24px;
  --font:'Outfit', 'Google Sans', 'Roboto', system-ui, sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{
  font-family:var(--font);
  background:var(--s01);
  color:var(--text);
  min-height:100vh;
  overflow-x:hidden;
  position:relative;
}

.material-symbols-outlined {
  font-family: 'Material Symbols Outlined', sans-serif;
  font-weight: normal;
  font-style: normal;
  font-size: 24px;
  line-height: 1;
  letter-spacing: normal;
  text-transform: none;
  display: inline-block;
  white-space: nowrap;
  word-wrap: normal;
  direction: ltr;
  -webkit-font-feature-settings: 'liga';
  -webkit-font-smoothing: antialiased;
}

/* Animated Background Mesh */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background: 
    radial-gradient(circle at 10% 0%, var(--accent-light) 0%, transparent 40%),
    radial-gradient(circle at 90% 100%, var(--accent-light) 0%, transparent 40%);
  z-index: -1;
  pointer-events: none;
  animation: bgPulse 10s ease-in-out infinite alternate;
}
@keyframes bgPulse {
  0% { opacity: 0.5; transform: scale(1); }
  100% { opacity: 1; transform: scale(1.1); }
}

/* Custom Scrollbars */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--s01); }
::-webkit-scrollbar-thumb { background: var(--s03); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* Hero Section */
.hero{
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  padding: 30px 40px;
  position: relative;
  z-index: 10;
}
.hero-inner{max-width:1400px;margin:0 auto;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:20px}
.hero-text{flex:1;min-width:300px; animation: slideDown 0.6s cubic-bezier(0.2, 0.8, 0.2, 1);}
@keyframes slideDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }

.hero-badge{
  display:inline-flex;align-items:center;gap:6px;
  background:var(--accent-lighter);
  border:1px solid var(--accent-light);
  border-radius:100px;padding:6px 16px;
  font-size:12px;font-weight:600;letter-spacing:.05em;margin-bottom:12px;
  color:var(--accent);
  box-shadow: 0 0 15px var(--accent-light);
}
.hero h1{font-size:32px;font-weight:700;letter-spacing:-.5px;margin-bottom:6px; background: linear-gradient(to right, #1f1f1f, var(--text2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.hero p{font-size:14px;color:var(--text2);max-width:560px;line-height:1.6}

/* Global Mode Switcher */
.mode-panel{background:var(--s02);padding:6px;border-radius:100px;display:flex;gap:6px;border:1px solid var(--border);box-shadow: var(--sh2); animation: fadeIn 0.8s ease;}
.mode-btn{background:none;border:none;color:var(--text2);padding:10px 20px;font-size:13px;font-weight:600;border-radius:100px;cursor:pointer;transition:all .3s cubic-bezier(0.4, 0, 0.2, 1);display:inline-flex;align-items:center;gap:8px}
.mode-btn.active{background:var(--accent);color:#ffffff;box-shadow:0 4px 15px var(--accent-light);font-weight:700; transform: scale(1.02);}
.mode-btn:hover:not(.active){color:var(--text);background:var(--s03);}

.nav-wrap{padding:20px 24px 0; z-index: 5; position: relative;}
.tab-bar{display:flex;gap:8px;max-width:1400px;margin:0 auto;background:var(--s02);padding:8px;border-radius:100px;border:1px solid var(--border);width:fit-content; box-shadow: var(--sh2);}
.tab-btn{padding:10px 24px;font-size:13px;font-weight:600;color:var(--text2);border:none;border-radius:100px;background:none;cursor:pointer;transition:all .3s cubic-bezier(0.4, 0, 0.2, 1);display:inline-flex;align-items:center;gap:8px}
.tab-btn:hover{color:var(--text);background:var(--s03);}
.tab-btn.active{color:var(--accent);background:var(--accent-light); border: 1px solid var(--accent); box-shadow:0 0 20px var(--accent-lighter);font-weight:700;}
.tab-panel{display:none;animation:fadeUp 0.5s cubic-bezier(0.2, 0.8, 0.2, 1) forwards; opacity: 0;}
.tab-panel.active{display:block}
@keyframes fadeUp{from{opacity:0;transform:translateY(15px)}to{opacity:1;transform:translateY(0)}}

.main{max-width:1400px;margin:0 auto;padding:24px 24px 80px; position: relative; z-index: 2;}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:24px}
@media(max-width:960px){.grid-2{grid-template-columns:1fr}}

/* M3 Elevated Cards */
.card{
  background:rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border:1px solid var(--border);
  border-radius:var(--r12);
  box-shadow:var(--sh2);
  overflow:hidden;
  transition:all .4s cubic-bezier(0.4, 0, 0.2, 1);
  margin-bottom:24px;
  position: relative;
}
.card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--border-s), transparent);
  opacity: 0.5;
}
.card:hover{
  box-shadow:var(--sh8);
  border-color:var(--border-s);
  transform: translateY(-2px);
}
.card-hd{padding:20px 24px;border-bottom:1px solid rgba(0,0,0,0.03);display:flex;align-items:center;justify-content:space-between;gap:12px;background:rgba(0,0,0,0.01)}
.card-hd-title{display:flex;align-items:center;gap:14px}
.card-ic{
  width:40px;height:40px;border-radius:12px;
  background:var(--accent-gradient);
  color:#ffffff;display:flex;align-items:center;justify-content:center;
  font-size:20px;flex-shrink:0;
  box-shadow: 0 4px 12px var(--accent-light);
}
.card-hd h2{font-size:15px;font-weight:700;color:#1f1f1f; letter-spacing: 0.3px;}
.card-hd p{font-size:12px;color:var(--text2);margin-top:2px}
.card-body{padding:24px}

.ctrl{margin-bottom:22px}
.ctrl:last-of-type{margin-bottom:0}
.ctrl-top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px}
.ctrl-name{font-size:13px;font-weight:600;color:var(--text)}
.ctrl-unit{font-size:11px;color:var(--text3); font-weight: 400;}
.ctrl-val{font-size:16px;font-weight:700;color:var(--accent);transition:color .3s; background: var(--accent-light); padding: 2px 8px; border-radius: 6px; border: 1px solid var(--accent);}

/* Material Design 3 Slider Glow */
input[type=range]{
  -webkit-appearance:none;
  width:100%;
  height:6px;
  background:var(--s03);
  border-radius:100px;
  outline:none;
  cursor:pointer;
  border:none;
  position: relative;
  overflow: hidden; /* For track fill effect */
}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;
  width:18px;
  height:18px;
  background:var(--accent);
  border:3px solid var(--s00);
  border-radius:50%;
  box-shadow: -100vw 0 0 100vw var(--accent-light), 0 0 10px var(--accent);
  transition:transform .2s cubic-bezier(0.4, 0, 0.2, 1), background .3s;
}
input[type=range]::-webkit-slider-thumb:hover{
  transform:scale(1.3);
}
.rl{display:flex;justify-content:space-between;font-size:10.5px;color:var(--text3);margin-top:6px; font-weight: 500;}

/* M3 Buttons */
.btn{
  padding:12px 24px;border:none;border-radius:100px;
  font-size:13px;font-weight:600;cursor:pointer;
  transition:all .3s cubic-bezier(0.4, 0, 0.2, 1);
  display:inline-flex;align-items:center;gap:8px;
  position: relative; overflow: hidden;
}
.btn::after {
  content: "";
  position: absolute; inset: 0; background: rgba(0,0,0,0.1);
  opacity: 0; transition: opacity 0.3s;
}
.btn:hover::after { opacity: 1; }
.btn-p{background:var(--accent);color:#ffffff;box-shadow:0 4px 14px var(--accent-light)}
.btn-p:hover{box-shadow:0 6px 20px var(--accent); transform: translateY(-1px);}
.btn-p:active{transform:scale(.98) translateY(0);}
.btn-s{background:transparent;color:var(--accent);border:1px solid var(--accent); box-shadow: 0 0 10px var(--accent-lighter);}
.btn-s:hover{background:var(--accent-light);color:var(--accent)}
.btn-g{background:#2e5c1a;color:#ffffff;box-shadow:0 4px 14px rgba(46, 92, 26, 0.3)}
.btn-g:hover{background:#386a20;box-shadow:0 6px 20px rgba(56, 106, 32, 0.4); transform: translateY(-1px);}
.btn-full{width:100%;justify-content:center;margin-top:14px; padding: 14px;}
.btn.loading{opacity:.7;pointer-events:none}

.spinner{width:14px;height:14px;border:2.5px solid rgba(0,0,0,.2);border-top-color:currentColor;border-radius:50%;animation:spin .8s cubic-bezier(0.4, 0, 0.2, 1) infinite;display:none}
.btn.loading .spinner{display:block}
@keyframes spin{to{transform:rotate(360deg)}}

/* Info Banner */
.info{background:var(--accent-lighter);border:1px solid var(--accent-light);border-radius:var(--r8);padding:14px 18px;display:flex;gap:14px;align-items:center;margin-bottom:20px;transition:all .3s;}
.info:hover { background: var(--accent-light); }
.info p{font-size:12.5px;color:var(--text);line-height:1.6;}
.info .material-symbols-outlined{color:var(--accent); font-size: 24px;}

.ph{text-align:center;padding:60px 20px;color:var(--text3); animation: fadeUp 0.6s ease;}
.ph .ico{font-size:48px;margin-bottom:16px;display:block;color:var(--text3); opacity: 0.5;}
.ph p{font-size:13px;line-height:1.6}

.wqi-g{display:flex;flex-direction:column;align-items:center;padding:24px;border-bottom:1px solid rgba(0,0,0,0.03)}
.wqi-ring{position:relative;width:140px;height:140px;margin-bottom:12px; filter: drop-shadow(0 0 10px var(--accent-light));}
.wqi-ring svg{width:140px;height:140px;transform:rotate(-90deg)}
.wqi-ring circle.bg{fill:none;stroke:var(--s03);stroke-width:8}
.wqi-ring circle.fg{fill:none;stroke-width:8;stroke-linecap:round;stroke-dasharray:377;transition:stroke-dashoffset 1.5s cubic-bezier(0.34, 1.56, 0.64, 1),stroke .5s}
.wqi-ct{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.wqi-n{font-size:32px;font-weight:700;line-height:1; letter-spacing: -1px;}
.wqi-l{font-size:11px;color:var(--text2);margin-top:2px; font-weight: 500;}

.badge{display:inline-block;padding:4px 14px;border-radius:100px;font-size:11.5px;font-weight:700;color:#ffffff;margin-bottom:8px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);}
.rec{margin:16px 24px;padding:16px;background:rgba(0,0,0,0.02);border-radius:var(--r8);border-left:4px solid var(--accent); transition:all .3s;}
.rec:hover { background: rgba(0,0,0,0.04); }
.rec h4{font-size:11.5px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text);margin-bottom:6px}
.rec p{font-size:13px;line-height:1.6;color:var(--text2)}

.ch-wrap{padding:16px 24px}
.ch-wrap h4{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text2);margin-bottom:10px}
.ch-wrap img{width:100%;border-radius:var(--r8);display:block;background:var(--s02); box-shadow: var(--sh2);}

/* Rules Split Panel Side-by-Side */
.rules-split{display:grid;grid-template-columns:1.2fr 0.8fr;gap:24px}
@media(max-width:1100px){.rules-split{grid-template-columns:1fr}}
.textarea-editor{
  width:100%;height:400px;
  font-family:'Courier New',Courier,monospace;font-size:13px;
  padding:16px;border:1px solid var(--border);border-radius:var(--r8);
  background:rgba(0,0,0,0.03);color:#0842a0;outline:none;resize:vertical;line-height:1.6;
  box-shadow: inset 0 2px 10px rgba(0,0,0,0.5);
  transition: border-color 0.3s, box-shadow 0.3s;
}
.textarea-editor:focus{border-color:var(--accent); box-shadow: inset 0 2px 10px rgba(0,0,0,0.5), 0 0 0 3px var(--accent-light);}

.rules-tb{display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap;align-items:center}
.rules-cnt{background:var(--accent-light);color:var(--accent);border:1px solid var(--accent);border-radius:100px;padding:2px 12px;font-size:11px;font-weight:700;margin-left:8px;transition:all .3s}
table.rt{width:100%;border-collapse:separate; border-spacing: 0; font-size:12.5px;}
table.rt th{background:rgba(0,0,0,0.03);color:var(--text);padding:12px;text-align:center;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--border);}
table.rt td{padding:10px 8px;border-bottom:1px solid rgba(0,0,0,0.03);text-align:center;vertical-align:middle;color:var(--text2); transition: background 0.2s;}
table.rt tr:hover td{background:var(--accent-lighter);color:var(--text);}
table.rt select, table.rt input.di{
  font-size:12px;padding:6px 10px;
  border:1px solid var(--border);border-radius:6px;
  background:rgba(0,0,0,0.02);color:var(--text);
  outline:none; transition: all 0.2s;
  font-family: var(--font);
}
table.rt select:focus, table.rt input.di:focus{border-color:var(--accent); box-shadow: 0 0 0 2px var(--accent-light);}
table.rt select{cursor:pointer;width:100%;max-width:95px;}
table.rt input.di{width:100%;min-width:120px;}
.del-b{
  background:rgba(186, 26, 26, 0.1);color:#ff5449;border:1px solid rgba(255, 84, 73, 0.3);
  border-radius:6px;padding:6px;cursor:pointer;
  display:inline-flex;align-items:center;justify-content:center;
  transition:all .2s;
}
.del-b:hover{background:#ff5449;color:#ffffff; box-shadow: 0 4px 10px rgba(255, 84, 73, 0.3);}

/* Diagrams View */
.diag-container{display:grid;grid-template-columns:260px 1fr;gap:24px}
@media(max-width:800px){.diag-container{grid-template-columns:1fr}}
.diag-list{display:flex;flex-direction:column;gap:8px}
.diag-btn{
  padding:12px 20px;font-size:13px;font-weight:600;
  background:rgba(0,0,0,0.02);border:1px solid transparent;border-radius:12px;
  cursor:pointer;color:var(--text2);text-align:left;
  transition:all .3s cubic-bezier(0.4, 0, 0.2, 1);
  display:flex;align-items:center;justify-content:space-between;
}
.diag-btn:hover{background:var(--s03);color:var(--text); transform: translateX(4px);}
.diag-btn.active{
  background:var(--accent-light);color:var(--accent);
  border-color:var(--accent);
  box-shadow:0 4px 12px var(--accent-lighter);
  font-weight:700; transform: translateX(8px);
}
.diag-btn .material-symbols-outlined{font-size:18px}

.timeline-card{
  background:var(--accent-lighter);border:1px solid var(--accent-light);
  border-radius:var(--r12);padding:16px 24px;margin-bottom:20px;
  display:flex;align-items:center;gap:20px;flex-wrap:wrap;
  box-shadow: var(--sh2); transition:all .3s;
}
.timeline-slider{flex:1;min-width:250px}

footer{
  text-align:center;padding:30px;font-size:12px;color:var(--text3);
  border-top:1px solid var(--border);margin-top:40px;
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(10px);
}
</style>
</head>
<body>

<div class="hero">
  <div class="hero-inner">
    <div class="hero-text">
      <div class="hero-badge" id="app-badge"><span class="material-symbols-outlined" style="font-size:14px;vertical-align:middle;margin-right:4px">psychology</span> Mamdani Fuzzy Logic AI</div>
      <h1 id="app-title">AquaIQ Smart Water Assessment</h1>
      <p id="app-desc">5 sensors · 15 editable fuzzy rules · centroid defuzzification · full diagnostics · PDF export</p>
    </div>
    <div class="mode-panel">
      <button class="mode-btn active" id="btn-mode-wqi" onclick="toggleMode('wqi')"><span class="material-symbols-outlined">water_drop</span> Water Assessment</button>
      <button class="mode-btn" id="btn-mode-pid" onclick="toggleMode('pid')"><span class="material-symbols-outlined">settings</span> Fuzzy PID Simulator</button>
    </div>
  </div>
</div>

<!-- ================= 1. WATER QUALITY ASSESSMENT APP CONTAINER ================= -->
<div id="wqi-container">
  <div class="nav-wrap">
    <div class="tab-bar">
      <button class="tab-btn active" id="wqi-tab-btn-assess" onclick="switchWqiTab('assess')"><span class="material-symbols-outlined">science</span> Assessment Dashboard</button>
      <button class="tab-btn" id="wqi-tab-btn-diagrams" onclick="switchWqiTab('diagrams')"><span class="material-symbols-outlined">analytics</span> Diagnostic Mappings</button>
      <button class="tab-btn" id="wqi-tab-btn-rules" onclick="switchWqiTab('rules')"><span class="material-symbols-outlined">rule</span> Rules Engine Editor</button>
    </div>
  </div>

  <div class="main">
    <!-- WQI TAB 1: ASSESSMENT -->
    <div id="wqi-tab-assess" class="tab-panel active">
      <div class="info">
        <span class="material-symbols-outlined">info</span>
        <p><strong>Fuzzy Mamdani Pipeline:</strong> 5 input sensor states are evaluated simultaneously through the rule base to calculate WQI (0-100) via centroid defuzzification.</p>
      </div>
      <div class="grid-2">
        <div class="card">
          <div class="card-hd"><div class="card-hd-title"><div class="card-ic"><span class="material-symbols-outlined">sensors</span></div><div><h2>Water Sensor Telemetry</h2><p>Adjust current water values</p></div></div></div>
          <div class="card-body">
            <div class="ctrl">
              <div class="ctrl-top"><span class="ctrl-name">pH Level</span><span class="ctrl-val" id="val-wqi-ph">7.0</span></div>
              <input type="range" id="sl-wqi-ph" min="0" max="14" step="0.1" value="7.0" oninput="document.getElementById('val-wqi-ph').textContent=parseFloat(this.value).toFixed(1)"/>
              <div class="rl"><span>0.0 Acidic</span><span>7.0 Neutral</span><span>14.0 Alkaline</span></div>
            </div>
            <div class="ctrl">
              <div class="ctrl-top"><span class="ctrl-name">Turbidity <span class="ctrl-unit">(Clarity)</span></span><span class="ctrl-val" id="val-wqi-turb">10</span></div>
              <input type="range" id="sl-wqi-turb" min="0" max="100" step="1" value="10" oninput="document.getElementById('val-wqi-turb').textContent=this.value"/>
              <div class="rl"><span>0 NTU Clear</span><span>50</span><span>100 NTU Cloudy</span></div>
            </div>
            <div class="ctrl">
              <div class="ctrl-top"><span class="ctrl-name">Dissolved Oxygen</span><span class="ctrl-val" id="val-wqi-do">9.0</span></div>
              <input type="range" id="sl-wqi-do" min="0" max="20" step="0.1" value="9.0" oninput="document.getElementById('val-wqi-do').textContent=parseFloat(this.value).toFixed(1)"/>
              <div class="rl"><span>0.0 mg/L Hypoxic</span><span>10.0</span><span>20.0 mg/L Saturated</span></div>
            </div>
            <div class="ctrl">
              <div class="ctrl-top"><span class="ctrl-name">Temperature</span><span class="ctrl-val" id="val-wqi-temp">22</span></div>
              <input type="range" id="sl-wqi-temp" min="0" max="50" step="1" value="22" oninput="document.getElementById('val-wqi-temp').textContent=this.value"/>
              <div class="rl"><span>0 °C Freezing</span><span>25</span><span>50 °C Boiling</span></div>
            </div>
            <div class="ctrl">
              <div class="ctrl-top"><span class="ctrl-name">Conductivity</span><span class="ctrl-val" id="val-wqi-cond">250</span></div>
              <input type="range" id="sl-wqi-cond" min="0" max="2000" step="10" value="250" oninput="document.getElementById('val-wqi-cond').textContent=this.value"/>
              <div class="rl"><span>0 µS Purified</span><span>1000 Mineral</span><span>2000 µS Saline</span></div>
            </div>
            <button class="btn btn-p btn-full" id="btn-wqi-eval" onclick="runWqiAssessment()">
              <div class="spinner"></div>
              <span class="material-symbols-outlined" style="font-size:16px;vertical-align:middle">bolt</span>
              <span>Compute Assessment Diagnostics</span>
            </button>
          </div>
        </div>

        <div class="card">
          <div class="card-hd"><div class="card-hd-title"><div class="card-ic"><span class="material-symbols-outlined">insights</span></div><div><h2>Diagnostics Outcome</h2><p>Inference rating and treatment recommendation</p></div></div></div>
          <div id="wqi-result-area">
            <div class="ph"><span class="material-symbols-outlined ico">query_stats</span><p>Set sensor states and execute <strong>Diagnostics</strong><br/>to trigger the fuzzy inference engine.</p></div>
          </div>
        </div>
      </div>
    </div>

    <!-- WQI TAB 2: DIAGRAMS -->
    <div id="wqi-tab-diagrams" class="tab-panel">
      <div id="wqi-diag-ph" class="ph" style="padding:60px"><span class="material-symbols-outlined ico">legend_toggle</span><p>Run diagnostic assessment first to generate 2D/3D fuzzy mapping graphs.</p></div>
      <div id="wqi-diag-area" style="display:none">
        <div class="diag-container">
          <div class="diag-list">
            <button class="diag-btn active" id="wqi-btn-diag-output_mf" onclick="switchWqiDiag('output_mf')"><span>Output Centroid</span> <span class="material-symbols-outlined">chevron_right</span></button>
            <button class="diag-btn" id="wqi-btn-diag-rule_firing" onclick="switchWqiDiag('rule_firing')"><span>Rule Firing</span> <span class="material-symbols-outlined">chevron_right</span></button>
            <button class="diag-btn" id="wqi-btn-diag-correlation" onclick="switchWqiDiag('correlation')"><span>Breakdown Bars</span> <span class="material-symbols-outlined">chevron_right</span></button>
            <button class="diag-btn" id="wqi-btn-diag-surface" onclick="switchWqiDiag('surface')"><span>3D Control Surface</span> <span class="material-symbols-outlined">chevron_right</span></button>
            <button class="diag-btn" id="wqi-btn-diag-radar" onclick="switchWqiDiag('radar')"><span>Input Radar</span> <span class="material-symbols-outlined">chevron_right</span></button>
            <button class="diag-btn" id="wqi-btn-diag-mf_ph" onclick="switchWqiDiag('mf_ph')"><span>pH MFs</span> <span class="material-symbols-outlined">chevron_right</span></button>
            <button class="diag-btn" id="wqi-btn-diag-mf_turbidity" onclick="switchWqiDiag('mf_turbidity')"><span>Turbidity MFs</span> <span class="material-symbols-outlined">chevron_right</span></button>
            <button class="diag-btn" id="wqi-btn-diag-mf_do" onclick="switchWqiDiag('mf_do')"><span>DO MFs</span> <span class="material-symbols-outlined">chevron_right</span></button>
            <button class="diag-btn" id="wqi-btn-diag-mf_temp" onclick="switchWqiDiag('mf_temp')"><span>Temp MFs</span> <span class="material-symbols-outlined">chevron_right</span></button>
            <button class="diag-btn" id="wqi-btn-diag-mf_conductivity" onclick="switchWqiDiag('mf_conductivity')"><span>Conductivity MFs</span> <span class="material-symbols-outlined">chevron_right</span></button>
          </div>
          <div class="card" style="margin-bottom:0">
            <div class="card-body" style="text-align:center"><img id="wqi-diag-img" src="" style="max-width:100%;border-radius:4px"/></div>
            <div class="card-hd" style="border-top:1px solid var(--border)">
              <span>Download Assessment Artifacts:</span>
              <button class="btn btn-g" id="btn-wqi-pdf" onclick="downloadWqiPDF()">
                <div class="spinner"></div>
                <span class="material-symbols-outlined" style="font-size:16px;vertical-align:middle">picture_as_pdf</span>
                <span>Export Complete PDF Diagnostic Report</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- WQI TAB 3: RULES -->
    <div id="wqi-tab-rules" class="tab-panel">
      <div class="rules-split">
        <div class="card">
          <div class="card-hd">
            <div class="card-hd-title"><div class="card-ic"><span class="material-symbols-outlined">table_chart</span></div><div><h2>Dynamic Visual Table Editor</h2><p>Rules count: <span class="rules-cnt" id="wqi-rules-cnt">0</span></p></div></div>
            <div class="rules-tb">
              <button class="btn btn-p" onclick="addWqiRule()"><span class="material-symbols-outlined" style="font-size:15px;vertical-align:middle">add</span> Add Rule</button>
              <button class="btn btn-g" onclick="applyWqiRules()"><span class="material-symbols-outlined" style="font-size:15px;vertical-align:middle">save</span> Save & Recheck</button>
              <button class="btn btn-s" onclick="resetWqiRules()"><span class="material-symbols-outlined" style="font-size:15px;vertical-align:middle">restart_alt</span> Reset Default</button>
            </div>
          </div>
          <div class="card-body" style="overflow-x:auto;padding:0">
            <table class="rt">
              <thead>
                <tr><th>#</th><th>pH</th><th>Turbidity</th><th>DO</th><th>Temp</th><th>Cond</th><th>⇒ Output Rating</th><th>Annotation Note</th><th></th></tr>
              </thead>
              <tbody id="wqi-rules-tbody"></tbody>
            </table>
          </div>
        </div>

        <div class="card">
          <div class="card-hd"><div class="card-hd-title"><div class="card-ic"><span class="material-symbols-outlined">edit_note</span></div><div><h2>Natural Language Rule Writer</h2><p>Rules sync instantly side-by-side</p></div></div></div>
          <div class="card-body">
            <div class="info" style="margin-bottom:12px">
              <span class="material-symbols-outlined">info</span>
              <p>Write rules in conversational English. <strong>Format:</strong><br/><i>IF ph IS neutral AND turbidity IS clear THEN output IS excellent // Annotation</i></p>
            </div>
            <textarea class="textarea-editor" id="wqi-textarea" oninput="syncWqiTextareaToTable()"></textarea>
            <div style="margin-top:10px;text-align:right">
              <span id="wqi-text-status" style="font-size:11px;font-weight:700;color:#386a20;margin-right:10px"><span class="material-symbols-outlined" style="font-size:14px;vertical-align:middle">check_circle</span> Synced</span>
              <button class="btn btn-p" onclick="applyWqiRules()"><span class="material-symbols-outlined" style="font-size:15px;vertical-align:middle">flash_on</span> Parse & Compile Rules</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ================= 2. FUZZY PID SIMULATION APP CONTAINER ================= -->
<div id="pid-container" style="display:none">
  <div class="nav-wrap">
    <div class="tab-bar">
      <button class="tab-btn active" id="pid-tab-btn-sim" onclick="switchPidTab('sim')"><span class="material-symbols-outlined">thermostat</span> Closed-Loop Simulation</button>
      <button class="tab-btn" id="pid-tab-btn-diagrams" onclick="switchPidTab('diagrams')"><span class="material-symbols-outlined">analytics</span> Controller Mappings</button>
      <button class="tab-btn" id="pid-tab-btn-rules" onclick="switchPidTab('rules')"><span class="material-symbols-outlined">rule</span> PID Decision Matrices</button>
    </div>
  </div>

  <div class="main">
    <!-- PID TAB 1: SIMULATION -->
    <div id="pid-tab-sim" class="tab-panel active">
      <div class="info">
        <span class="material-symbols-outlined">info</span>
        <p><strong>4-Input thermodynamic Loop:</strong> Continuous PID system regulates a water tank. System processes <strong>Error (e)</strong>, <strong>dError (de)</strong>, <strong>intError (ie)</strong>, and <strong>Disturbance Flow (d)</strong> to compute output control power.</p>
      </div>
      <div class="grid-2">
        <div class="card">
          <div class="card-hd"><div class="card-hd-title"><div class="card-ic"><span class="material-symbols-outlined">tune</span></div><div><h2>Simulation Parameters</h2><p>Set physical loop and flow conditions</p></div></div></div>
          <div class="card-body">
            <div class="ctrl">
              <div class="ctrl-top"><span class="ctrl-name">Target Setpoint Temperature</span><span class="ctrl-val" id="val-pid-sp">50 °C</span></div>
              <input type="range" id="sl-pid-sp" min="20" max="90" step="1" value="50" oninput="document.getElementById('val-pid-sp').textContent=this.value+' °C'"/>
              <div class="rl"><span>20 °C Room Temp</span><span>55</span><span>90 °C Near Boiling</span></div>
            </div>
            <div class="ctrl">
              <div class="ctrl-top"><span class="ctrl-name">Initial Fluid Temperature</span><span class="ctrl-val" id="val-pid-init">20 °C</span></div>
              <input type="range" id="sl-pid-init" min="10" max="40" step="1" value="20" oninput="document.getElementById('val-pid-init').textContent=this.value+' °C'"/>
              <div class="rl"><span>10 °C Cold Chill</span><span>25</span><span>40 °C Warm Water</span></div>
            </div>
            <div class="ctrl">
              <div class="ctrl-top"><span class="ctrl-name">Cold Water Inflow Disturbance</span><span class="ctrl-val" id="val-pid-dist">2.0 L/m</span></div>
              <input type="range" id="sl-pid-dist" min="0" max="10" step="0.5" value="2.0" oninput="document.getElementById('val-pid-dist').textContent=parseFloat(this.value).toFixed(1)+' L/m'"/>
              <div class="rl"><span>0.0 L/m Zero Inflow</span><span>5.0</span><span>10.0 L/m Heavy Flush</span></div>
            </div>
            <div class="ctrl">
              <div class="ctrl-top"><span class="ctrl-name">Simulation Time (Duration)</span><span class="ctrl-val" id="val-pid-time">300 Seconds</span></div>
              <input type="range" id="sl-pid-time" min="30" max="600" step="10" value="300" oninput="document.getElementById('val-pid-time').textContent=this.value+' Seconds'"/>
              <div class="rl"><span>30s Fast Response</span><span>315s</span><span>600s Steady State</span></div>
            </div>
            <button class="btn btn-p btn-full" id="btn-pid-sim" onclick="runPidSimulation()">
              <div class="spinner"></div>
              <span class="material-symbols-outlined" style="font-size:16px;vertical-align:middle">play_arrow</span>
              <span>Execute Physics Loop Simulation</span>
            </button>
          </div>
        </div>

        <div class="card">
          <div class="card-hd"><div class="card-hd-title"><div class="card-ic"><span class="material-symbols-outlined">show_chart</span></div><div><h2>Real-time Closed-Loop Response</h2><p>Target tracking & disturbance recovery curve</p></div></div></div>
          <div class="card-body" id="pid-chart-area">
            <div class="ph"><span class="material-symbols-outlined ico">refresh</span><p>Configure parameters and run the simulation loop<br/>to generate the real-time step response chart.</p></div>
          </div>
        </div>
      </div>
    </div>

    <!-- PID TAB 2: DIAGRAMS -->
    <div id="pid-tab-diagrams" class="tab-panel">
      <div id="pid-diag-ph" class="ph" style="padding:60px"><span class="material-symbols-outlined ico">analytics</span><p>Run a simulation loop first, then scroll through time-series step responses.</p></div>
      <div id="pid-diag-area" style="display:none">
        
        <!-- Timeline Scrubbing -->
        <div class="timeline-card">
          <span class="material-symbols-outlined" style="color:var(--accent)">history</span>
          <span><strong>Timeline Diagnostics scrubbing:</strong></span>
          <input type="range" class="timeline-slider" id="sl-pid-scrub" min="0" max="59" step="1" value="15" oninput="scrubSimulationTime(this.value)"/>
          <span style="font-size:13px;font-weight:700">Timestep k = <span id="val-scrub-step" style="color:var(--accent)">15</span> Seconds</span>
          <span style="font-size:12px;color:var(--text2)">(e: <span id="val-scrub-e">0.0</span> | de: <span id="val-scrub-de">0.0</span> | ie: <span id="val-scrub-ie">0.0</span> | Output u: <span id="val-scrub-u">0.0</span>%)</span>
        </div>

        <div class="diag-container">
          <div class="diag-list">
            <button class="diag-btn active" id="pid-btn-diag-output_mf" onclick="switchPidDiag('output_mf')"><span>PID Output u MFs</span> <span class="material-symbols-outlined">chevron_right</span></button>
            <button class="diag-btn" id="pid-btn-diag-rule_firing" onclick="switchPidDiag('rule_firing')"><span>Active Firing</span> <span class="material-symbols-outlined">chevron_right</span></button>
            <button class="diag-btn" id="pid-btn-diag-correlation" onclick="switchPidDiag('correlation')"><span>Correlation Bars</span> <span class="material-symbols-outlined">chevron_right</span></button>
            <button class="diag-btn" id="pid-btn-diag-surface" onclick="switchPidDiag('surface')"><span>3D Surface (e x de)</span> <span class="material-symbols-outlined">chevron_right</span></button>
            <button class="diag-btn" id="pid-btn-diag-radar" onclick="switchPidDiag('radar')"><span>Normalized Radar</span> <span class="material-symbols-outlined">chevron_right</span></button>
            <button class="diag-btn" id="pid-btn-diag-mf_e" onclick="switchPidDiag('mf_e')"><span>Error e MFs</span> <span class="material-symbols-outlined">chevron_right</span></button>
            <button class="diag-btn" id="pid-btn-diag-mf_de" onclick="switchPidDiag('mf_de')"><span>dError de MFs</span> <span class="material-symbols-outlined">chevron_right</span></button>
            <button class="diag-btn" id="pid-btn-diag-mf_ie" onclick="switchPidDiag('mf_ie')"><span>intError ie MFs</span> <span class="material-symbols-outlined">chevron_right</span></button>
            <button class="diag-btn" id="pid-btn-diag-mf_d" onclick="switchPidDiag('mf_d')"><span>Disturbance d MFs</span> <span class="material-symbols-outlined">chevron_right</span></button>
          </div>
          <div class="card" style="margin-bottom:0">
            <div class="card-body" style="text-align:center"><img id="pid-diag-img" src="" style="max-width:100%;border-radius:4px"/></div>
            <div class="card-hd" style="border-top:1px solid var(--border)">
              <span>Export Control Reports:</span>
              <button class="btn btn-g" id="btn-pid-pdf" onclick="downloadPidPDF()">
                <div class="spinner"></div>
                <span class="material-symbols-outlined" style="font-size:16px;vertical-align:middle">picture_as_pdf</span>
                <span>Export Complete PDF Simulation Report</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- PID TAB 3: RULES -->
    <div id="pid-tab-rules" class="tab-panel">
      <div class="rules-split">
        <div class="card">
          <div class="card-hd">
            <div class="card-hd-title"><div class="card-ic"><span class="material-symbols-outlined">table_chart</span></div><div><h2>Fuzzy PID Decisions Grid</h2><p>Rules count: <span class="rules-cnt" id="pid-rules-cnt">0</span></p></div></div>
            <div class="rules-tb">
              <button class="btn btn-p" onclick="addPidRule()"><span class="material-symbols-outlined" style="font-size:15px;vertical-align:middle">add</span> Add Rule</button>
              <button class="btn btn-g" onclick="applyPidRules()"><span class="material-symbols-outlined" style="font-size:15px;vertical-align:middle">save</span> Save Decisions</button>
              <button class="btn btn-s" onclick="resetPidRules()"><span class="material-symbols-outlined" style="font-size:15px;vertical-align:middle">restart_alt</span> Reset Default</button>
            </div>
          </div>
          <div class="card-body" style="overflow-x:auto;padding:0">
            <table class="rt">
              <thead>
                <tr><th>#</th><th>Error e</th><th>dError de</th><th>intError ie</th><th>Disturb d</th><th>⇒ Control Action u</th><th>Annotation Description</th><th></th></tr>
              </thead>
              <tbody id="pid-rules-tbody"></tbody>
            </table>
          </div>
        </div>

        <div class="card">
          <div class="card-hd"><div class="card-hd-title"><div class="card-ic"><span class="material-symbols-outlined">edit_note</span></div><div><h2>PID Decision Script Writer</h2><p>Rules sync instantly side-by-side</p></div></div></div>
          <div class="card-body">
            <div class="info" style="margin-bottom:12px">
              <span class="material-symbols-outlined">info</span>
              <p>Write rules in conversational English. <strong>Format:</strong><br/><i>IF error IS positive AND change_error IS positive THEN output IS heat_fast // Annotation</i></p>
            </div>
            <textarea class="textarea-editor" id="pid-textarea" oninput="syncPidTextareaToTable()"></textarea>
            <div style="margin-top:10px;text-align:right">
              <span id="pid-text-status" style="font-size:11px;font-weight:700;color:#386a20;margin-right:10px"><span class="material-symbols-outlined" style="font-size:14px;vertical-align:middle">check_circle</span> Synced</span>
              <button class="btn btn-p" onclick="applyPidRules()"><span class="material-symbols-outlined" style="font-size:15px;vertical-align:middle">flash_on</span> Compile PID Matrix</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<footer>AquaIQ v3 · Full Multi-Variable Mamdani Inference & Closed-Loop Simulation System · PDF Only Reports</footer>

<script>
// Options configuration lists
const WQI_OPTS = {
  ph: ['acidic','neutral','alkaline','ANY'],
  turbidity: ['clear','moderate','cloudy','ANY'],
  do: ['low','medium','high','ANY'],
  temp: ['cold','optimal','hot','ANY'],
  conductivity: ['low','medium','high','ANY'],
  output: ['very_poor','poor','acceptable','good','excellent']
};
const PID_OPTS = {
  error: ['negative','zero','positive','ANY'],
  change_error: ['negative','zero','positive','ANY'],
  int_error: ['negative','zero','positive','ANY'],
  disturbance: ['low','medium','high','ANY'],
  output: ['cool_fast','cool_slow','maintain','heat_slow','heat_fast']
};

let currentMode = 'wqi';
let wqiTab = 'assess';
let pidTab = 'sim';

let wqiRulesList = [];
let pidRulesList = [];

let lastWqiResult = null;
let lastPidResult = null;

let currentWqiDiag = 'output_mf';
let currentPidDiag = 'output_mf';
let chartInstance = null; // ChartJS reference

// ══════════════════════════════════════════════════════════════════════
// MODE SWITCHER
// ══════════════════════════════════════════════════════════════════════
function toggleMode(mode) {
  currentMode = mode;
  document.getElementById('btn-mode-wqi').classList.toggle('active', mode==='wqi');
  document.getElementById('btn-mode-pid').classList.toggle('active', mode==='pid');
  document.getElementById('wqi-container').style.display = mode==='wqi' ? 'block' : 'none';
  document.getElementById('pid-container').style.display = mode==='pid' ? 'block' : 'none';
  
  const root = document.documentElement;
  if(mode==='wqi'){
    root.style.setProperty('--accent', '#00a3ff');
    root.style.setProperty('--accent-dark', '#008ae0');
    root.style.setProperty('--accent-deep', '#002f4d');
    root.style.setProperty('--accent-light', 'rgba(0, 163, 255, 0.12)');
    root.style.setProperty('--accent-lighter', 'rgba(0, 163, 255, 0.05)');
    root.style.setProperty('--accent-gradient', 'linear-gradient(135deg, #00c6ff, #0072ff 45%, #0052a3)');
    
    document.getElementById('app-badge').innerHTML = '<span class="material-symbols-outlined" style="font-size:14px;vertical-align:middle;margin-right:4px">psychology</span> Mamdani Fuzzy Logic AI';
    document.getElementById('app-title').textContent = 'AquaIQ Smart Water Assessment';
    document.getElementById('app-desc').textContent = '5 sensors · 15 editable fuzzy rules · centroid defuzzification · full diagnostics · PDF export';
  } else {
    root.style.setProperty('--accent', '#b09fff');
    root.style.setProperty('--accent-dark', '#8a75e5');
    root.style.setProperty('--accent-deep', '#2c1654');
    root.style.setProperty('--accent-light', 'rgba(176, 159, 255, 0.15)');
    root.style.setProperty('--accent-lighter', 'rgba(176, 159, 255, 0.05)');
    root.style.setProperty('--accent-gradient', 'linear-gradient(135deg, #d0bcff, #b09fff 45%, #6750a4)');
    
    document.getElementById('app-badge').innerHTML = '<span class="material-symbols-outlined" style="font-size:14px;vertical-align:middle;margin-right:4px">thermostat</span> Fuzzy PID Control Loop';
    document.getElementById('app-title').textContent = 'AquaIQ Fuzzy PID Control Simulator';
    document.getElementById('app-desc').textContent = '4 inputs · thermal water process model · 15 rules · real-time response curve · interactive timeline scrubbing';
  }
  
  if (mode === 'pid' && !lastPidResult) {
    runPidSimulation();
  }
}

// ══════════════════════════════════════════════════════════════════════
// WATER QUALITY FLOWS
// ══════════════════════════════════════════════════════════════════════
function switchWqiTab(tab){
  wqiTab = tab;
  document.querySelectorAll('#wqi-container .tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('#wqi-container .tab-panel').forEach(p => p.classList.remove('active'));
  
  document.getElementById(`wqi-tab-btn-${tab}`).classList.add('active');
  document.getElementById(`wqi-tab-${tab}`).classList.add('active');
  if(tab==='rules') loadWqiRules();
}

async function runWqiAssessment(){
  const btn = document.getElementById('btn-wqi-eval');
  btn.classList.add('loading');
  const payload = {
    ph: parseFloat(document.getElementById('sl-wqi-ph').value),
    turbidity: parseFloat(document.getElementById('sl-wqi-turb').value),
    do_level: parseFloat(document.getElementById('sl-wqi-do').value),
    temperature: parseFloat(document.getElementById('sl-wqi-temp').value),
    conductivity: parseFloat(document.getElementById('sl-wqi-cond').value),
  };
  try{
    const res = await fetch('/wqi/evaluate', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const data = await res.json();
    lastWqiResult = data;
    renderWqiResult(data);
    showWqiCharts(data);
  }catch(e){alert('Assessment Failure: '+e.message);}
  finally{btn.classList.remove('loading');}
}

function renderWqiResult(d){
  const offset = 345 - (d.wqi / 100) * 345;
  document.getElementById('wqi-result-area').innerHTML = `
    <div style="animation:fadeIn 0.4s ease">
      <div class="wqi-g">
        <div class="wqi-ring">
          <svg viewBox="0 0 120 120">
            <circle class="bg" cx="60" cy="60" r="55"/>
            <circle class="fg" cx="60" cy="60" r="55" stroke="${d.color}" stroke-dashoffset="${offset}"/>
          </svg>
          <div class="wqi-ct">
            <span class="wqi-n" style="color:${d.color}">${d.wqi}</span>
            <span class="wqi-l">/ 100</span>
          </div>
        </div>
        <span class="badge" style="background:${d.color}">${d.category}</span>
        <span style="font-size:12px;color:var(--text2)">Computed Water Quality Index</span>
      </div>
      <div class="rec">
        <h4>🔧 Treatment Protocol Guideline</h4>
        <p>${d.recommendation}</p>
      </div>
      <div class="ch-wrap">
        <h4>Centroid Balance Distribution</h4>
        <img src="data:image/png;base64,${d.charts.output_mf}"/>
      </div>
    </div>`;
}

function showWqiCharts(d){
  document.getElementById('wqi-diag-ph').style.display='none';
  document.getElementById('wqi-diag-area').style.display='block';
  switchWqiDiag(currentWqiDiag);
}

function switchWqiDiag(key){
  currentWqiDiag = key;
  document.querySelectorAll('#wqi-tab-diagrams .diag-btn').forEach(btn => btn.classList.remove('active'));
  document.getElementById(`wqi-btn-diag-${key}`).classList.add('active');
  if(lastWqiResult) {
    document.getElementById('wqi-diag-img').src = 'data:image/png;base64,' + lastWqiResult.charts[key];
  }
}

async function downloadWqiPDF(){
  const btn = document.getElementById('btn-wqi-pdf');
  btn.classList.add('loading');
  try{
    const res = await fetch('/wqi/report', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({})});
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `AquaIQ_WQI_Report_${new Date().toISOString().slice(0,10)}.pdf`; a.click();
    URL.revokeObjectURL(url);
  }catch(e){alert('PDF Generation Failure: '+e.message);}
  finally{btn.classList.remove('loading');}
}

// Rules Sync Engine WQI
async function loadWqiRules(){
  const res = await fetch('/wqi/rules');
  wqiRulesList = await res.json();
  renderWqiRulesTable();
  serializeWqiRulesToText();
}

function renderWqiRulesTable(){
  document.getElementById('wqi-rules-cnt').textContent = wqiRulesList.length;
  const fields = ['ph','turbidity','do','temp','conductivity'];
  document.getElementById('wqi-rules-tbody').innerHTML = wqiRulesList.map((r, i) => {
    const cells = fields.map((f, fi) => {
      const val = r[fi] || 'ANY';
      const opts = WQI_OPTS[f].map(o => `<option value="${o}"${o===val?' selected':''}>${o}</option>`).join('');
      return `<td><select onchange="updateWqiTableRuleValue(${i}, ${fi}, this.value)">${opts}</select></td>`;
    }).join('');
    
    const outOpts = WQI_OPTS.output.map(o => `<option value="${o}"${o===r[5]?' selected':''}>${o.replace('_',' ')}</option>`).join('');
    return `<tr>
      <td style="font-weight:700;color:var(--text2)">${i+1}</td>
      ${cells}
      <td><select style="font-weight:700;color:var(--accent)" onchange="updateWqiTableRuleValue(${i}, 5, this.value)">${outOpts}</select></td>
      <td><input class="di" value="${r[6]}" oninput="updateWqiTableRuleValue(${i}, 6, this.value)"/></td>
      <td><button class="del-b" onclick="deleteWqiRule(${i})"><span class="material-symbols-outlined" style="font-size:16px;vertical-align:middle">delete</span></button></td>
    </tr>`;
  }).join('');
}

function updateWqiTableRuleValue(idx, col, val){
  wqiRulesList[idx][col] = val === 'ANY' ? null : val;
  serializeWqiRulesToText();
}

function deleteWqiRule(idx){
  wqiRulesList.splice(idx, 1);
  renderWqiRulesTable();
  serializeWqiRulesToText();
}

function addWqiRule(){
  wqiRulesList.push([null, null, null, null, null, 'acceptable', 'Custom wqi rule']);
  renderWqiRulesTable();
  serializeWqiRulesToText();
}

async function resetWqiRules(){
  if(!confirm('Reset rules to default baseline?')) return;
  const res = await fetch('/wqi/rules/reset', {method:'POST'});
  wqiRulesList = await res.json();
  renderWqiRulesTable();
  serializeWqiRulesToText();
}

function serializeWqiRulesToText(){
  const fields = ['ph','turbidity','do','temp','conductivity'];
  const text = wqiRulesList.map((r, i) => {
    let ants = [];
    fields.forEach((f, fi) => {
      if(r[fi]) ants.push(`${f} IS ${r[fi]}`);
    });
    let s = `IF ${ants.join(' AND ')} THEN output IS ${r[5]}`;
    if(r[6]) s += ` // ${r[6]}`;
    return s;
  }).join('\\n');
  document.getElementById('wqi-textarea').value = text;
}

function syncWqiTextareaToTable(){
  const text = document.getElementById('wqi-textarea').value;
  const lines = text.split('\\n');
  const fields = ['ph','turbidity','do','temp','conductivity'];
  let newRules = [];
  
  try{
    lines.forEach(line => {
      line = line.trim();
      if(!line || line.startsWith('//') || line.startsWith('#')) return;
      
      let desc = '';
      if(line.includes('//')){
        const idx = line.indexOf('//');
        desc = line.substring(idx+2).trim();
        line = line.substring(0, idx).trim();
      }
      
      const match = line.match(/^IF\\s+(.+?)\\s+THEN\\s+output\\s+IS\\s+(\\w+)/i);
      if(!match) return;
      
      const antsPart = match[1];
      const outVal = match[2].trim().toLowerCase();
      let rule = Array(7).fill(null);
      rule[5] = outVal;
      rule[6] = desc || 'Parsed wqi rule';
      
      const ants = antsPart.split(/\\s+AND\\s+/i);
      ants.forEach(ant => {
        const parts = ant.split(/\\s+IS\\s+/i);
        if(parts.length === 2){
          const fName = parts[0].trim().toLowerCase();
          const fVal = parts[1].trim().toLowerCase();
          const fIdx = fields.indexOf(fName);
          if(fIdx !== -1) rule[fIdx] = fVal === 'any' ? null : fVal;
        }
      });
      newRules.push(rule);
    });
    
    wqiRulesList = newRules;
    renderWqiRulesTable();
    document.getElementById('wqi-text-status').textContent = '✓ Synced';
    document.getElementById('wqi-text-status').style.color = '#107c10';
  }catch(err){
    document.getElementById('wqi-text-status').textContent = '⚠️ Syntax Error';
    document.getElementById('wqi-text-status').style.color = '#c50f1f';
  }
}

async function applyWqiRules(){
  syncWqiTextareaToTable();
  const res = await fetch('/wqi/rules', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(wqiRulesList)});
  alert('WQI Rules Matrix Compiled & Saved!');
}


// ══════════════════════════════════════════════════════════════════════
// DYNAMIC FUZZY PID FLOWS
// ══════════════════════════════════════════════════════════════════════
function switchPidTab(tab){
  pidTab = tab;
  document.querySelectorAll('#pid-container .tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('#pid-container .tab-panel').forEach(p => p.classList.remove('active'));
  
  document.getElementById(`pid-tab-btn-${tab}`).classList.add('active');
  document.getElementById(`pid-tab-${tab}`).classList.add('active');
  if(tab==='rules') loadPidRules();
}

async function runPidSimulation(){
  const btn = document.getElementById('btn-pid-sim');
  btn.classList.add('loading');
  
  const payload = {
    setpoint: parseFloat(document.getElementById('sl-pid-sp').value),
    initial_temp: parseFloat(document.getElementById('sl-pid-init').value),
    disturbance: parseFloat(document.getElementById('sl-pid-dist').value),
    duration: parseInt(document.getElementById('sl-pid-time').value)
  };
  
  try{
    const res = await fetch('/pid/simulate', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const data = await res.json();
    lastPidResult = data;
    
    renderPidChart(data.simulation);
    setupTimelineScrubber(data.simulation);
    showPidCharts();
  }catch(e){alert('Simulation Physics Failure: '+e.message);}
  finally{btn.classList.remove('loading');}
}

function renderPidChart(sim){
  if (typeof Chart === 'undefined') {
    document.getElementById('pid-chart-area').innerHTML = `
      <div class="ph" style="padding:40px;color:#ba1a1a">
        <span class="material-symbols-outlined ico">warning</span>
        <p><strong>Chart.js CDN could not be loaded.</strong><br/>Please check your internet connection to render response curves dynamically.</p>
      </div>`;
    return;
  }
  document.getElementById('pid-chart-area').innerHTML = `<canvas id="pidCanvas" style="width:100%;height:300px"></canvas>`;
  const ctx = document.getElementById('pidCanvas').getContext('2d');
  
  if (chartInstance) chartInstance.destroy();
  
  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: sim.t,
      datasets: [
        { 
          label: 'Target Setpoint (°C)', 
          data: sim.sp, 
          borderColor: '#ff6b6b', 
          borderDash: [6, 4], 
          borderWidth: 2, 
          fill: false, 
          pointRadius: 0,
          tension: 0.3
        },
        { 
          label: 'Water Temperature PV (°C)', 
          data: sim.pv, 
          borderColor: '#b09fff', 
          borderWidth: 3.5, 
          fill: true,
          backgroundColor: 'rgba(176, 159, 255, 0.04)',
          tension: 0.35, 
          pointRadius: 1.5,
          pointHoverRadius: 5
        },
        { 
          label: 'Heater Output u (%)', 
          data: sim.u, 
          borderColor: '#386a20', 
          borderWidth: 1.5, 
          fill: false, 
          pointRadius: 0,
          tension: 0.25
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: '#1f1f1f',
            font: { family: "'Google Sans', Roboto, sans-serif", size: 11, weight: '500' }
          }
        },
        zoom: {
          pan: {
            enabled: true,
            mode: 'x',
          },
          zoom: {
            wheel: {
              enabled: true,
            },
            pinch: {
              enabled: true
            },
            mode: 'x',
          }
        }
      },
      scales: {
        x: { 
          grid: { color: 'rgba(255, 255, 255, 0.06)' }, 
          ticks: { color: '#444746', font: { family: "'Google Sans', Roboto, sans-serif" } },
          title: { display: true, text: 'Time (Seconds)', color: '#1f1f1f', font: { family: "'Google Sans', Roboto, sans-serif", weight: 'bold' } } 
        },
        y: { 
          grid: { color: 'rgba(255, 255, 255, 0.06)' }, 
          ticks: { color: '#444746', font: { family: "'Google Sans', Roboto, sans-serif" } },
          title: { display: true, text: 'Value', color: '#1f1f1f', font: { family: "'Google Sans', Roboto, sans-serif", weight: 'bold' } } 
        }
      }
    }
  });
}

function setupTimelineScrubber(sim){
  document.getElementById('pid-diag-ph').style.display = 'none';
  document.getElementById('pid-diag-area').style.display = 'block';
  
  const slider = document.getElementById('sl-pid-scrub');
  slider.max = sim.t.length - 1;
  slider.value = 15;
  scrubSimulationTime(15);
}

async function scrubSimulationTime(k){
  if(!lastPidResult) return;
  k = parseInt(k);
  const sim = lastPidResult.simulation;
  
  document.getElementById('val-scrub-step').textContent = k;
  document.getElementById('val-scrub-e').textContent = sim.e[k].toFixed(2);
  document.getElementById('val-scrub-de').textContent = sim.de[k].toFixed(2);
  document.getElementById('val-scrub-ie').textContent = sim.ie[k].toFixed(2);
  document.getElementById('val-scrub-u').textContent = sim.u[k].toFixed(1);
  
  // Hit fast local timestep endpoint to scrub charts
  const payload = {
    e: sim.e[k], de: sim.de[k], ie: sim.ie[k], d: sim.d[k], u: sim.u[k]
  };
  
  try{
    const res = await fetch('/pid/timestep', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const charts = await res.json();
    lastPidResult.charts = charts; // update
    switchPidDiag(currentPidDiag);
  }catch(err){console.error('Time Scrub Error:', err);}
}

function showPidCharts(){
  switchPidDiag(currentPidDiag);
}

function switchPidDiag(key){
  currentPidDiag = key;
  document.querySelectorAll('#pid-tab-diagrams .diag-btn').forEach(btn => btn.classList.remove('active'));
  document.getElementById(`pid-btn-diag-${key}`).classList.add('active');
  if(lastPidResult && lastPidResult.charts) {
    document.getElementById('pid-diag-img').src = 'data:image/png;base64,' + lastPidResult.charts[key];
  }
}

async function downloadPidPDF(){
  const btn = document.getElementById('btn-pid-pdf');
  btn.classList.add('loading');
  try{
    const res = await fetch('/pid/report', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({})});
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `AquaIQ_PID_Simulation_Report_${new Date().toISOString().slice(0,10)}.pdf`; a.click();
    URL.revokeObjectURL(url);
  }catch(e){alert('PDF Report Failure: '+e.message);}
  finally{btn.classList.remove('loading');}
}

// Rules Sync Engine PID
async function loadPidRules(){
  const res = await fetch('/pid/rules');
  pidRulesList = await res.json();
  renderPidRulesTable();
  serializePidRulesToText();
}

function renderPidRulesTable(){
  document.getElementById('pid-rules-cnt').textContent = pidRulesList.length;
  const fields = ['error','change_error','int_error','disturbance'];
  document.getElementById('pid-rules-tbody').innerHTML = pidRulesList.map((r, i) => {
    const cells = fields.map((f, fi) => {
      const val = r[fi] || 'ANY';
      const opts = PID_OPTS[f].map(o => `<option value="${o}"${o===val?' selected':''}>${o}</option>`).join('');
      return `<td><select onchange="updatePidTableRuleValue(${i}, ${fi}, this.value)">${opts}</select></td>`;
    }).join('');
    
    const outOpts = PID_OPTS.output.map(o => `<option value="${o}"${o===r[4]?' selected':''}>${o.replace('_',' ')}</option>`).join('');
    return `<tr>
      <td style="font-weight:700;color:var(--text2)">${i+1}</td>
      ${cells}
      <td><select style="font-weight:700;color:var(--accent)" onchange="updatePidTableRuleValue(${i}, 4, this.value)">${outOpts}</select></td>
      <td><input class="di" value="${r[5]}" oninput="updatePidTableRuleValue(${i}, 5, this.value)"/></td>
      <td><button class="del-b" onclick="deletePidRule(${i})"><span class="material-symbols-outlined" style="font-size:16px;vertical-align:middle">delete</span></button></td>
    </tr>`;
  }).join('');
}

function updatePidTableRuleValue(idx, col, val){
  pidRulesList[idx][col] = val === 'ANY' ? null : val;
  serializePidRulesToText();
}

function deletePidRule(idx){
  pidRulesList.splice(idx, 1);
  renderPidRulesTable();
  serializePidRulesToText();
}

function addPidRule(){
  pidRulesList.push([null, null, null, null, 'maintain', 'Custom pid rule']);
  renderPidRulesTable();
  serializePidRulesToText();
}

async function resetPidRules(){
  if(!confirm('Reset rules to default PID controller baseline?')) return;
  const res = await fetch('/pid/rules/reset', {method:'POST'});
  pidRulesList = await res.json();
  renderPidRulesTable();
  serializePidRulesToText();
}

function serializePidRulesToText(){
  const fields = ['error','change_error','int_error','disturbance'];
  const text = pidRulesList.map((r, i) => {
    let ants = [];
    fields.forEach((f, fi) => {
      if(r[fi]) ants.push(`${f} IS ${r[fi]}`);
    });
    let s = `IF ${ants.join(' AND ')} THEN output IS ${r[4]}`;
    if(r[5]) s += ` // ${r[5]}`;
    return s;
  }).join('\\n');
  document.getElementById('pid-textarea').value = text;
}

function syncPidTextareaToTable(){
  const text = document.getElementById('pid-textarea').value;
  const lines = text.split('\\n');
  const fields = ['error','change_error','int_error','disturbance'];
  let newRules = [];
  
  try{
    lines.forEach(line => {
      line = line.trim();
      if(!line || line.startsWith('//') || line.startsWith('#')) return;
      
      let desc = '';
      if(line.includes('//')){
        const idx = line.indexOf('//');
        desc = line.substring(idx+2).trim();
        line = line.substring(0, idx).trim();
      }
      
      const match = line.match(/^IF\\s+(.+?)\\s+THEN\\s+output\\s+IS\\s+(\\w+)/i);
      if(!match) return;
      
      const antsPart = match[1];
      const outVal = match[2].trim().toLowerCase();
      let rule = Array(6).fill(null);
      rule[4] = outVal;
      rule[5] = desc || 'Parsed pid rule';
      
      const ants = antsPart.split(/\\s+AND\\s+/i);
      ants.forEach(ant => {
        const parts = ant.split(/\\s+IS\\s+/i);
        if(parts.length === 2){
          const fName = parts[0].trim().toLowerCase();
          const fVal = parts[1].trim().toLowerCase();
          const fIdx = fields.indexOf(fName);
          if(fIdx !== -1) rule[fIdx] = fVal === 'any' ? null : fVal;
        }
      });
      newRules.push(rule);
    });
    
    pidRulesList = newRules;
    renderPidRulesTable();
    document.getElementById('pid-text-status').textContent = '✓ Synced';
    document.getElementById('pid-text-status').style.color = '#107c10';
  }catch(err){
    document.getElementById('pid-text-status').textContent = '⚠️ Syntax Error';
    document.getElementById('pid-text-status').style.color = '#c50f1f';
  }
}

async function applyPidRules(){
  syncPidTextareaToTable();
  const res = await fetch('/pid/rules', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(pidRulesList)});
  alert('PID Controller Rules Saved!');
}

// Auto-initialize on page load
toggleMode('wqi');
runWqiAssessment();
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════════
# 9. FLASK ENDPOINTS
# ══════════════════════════════════════════════════════════════════════
@app.after_request
def disable_caching(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/')
def index():
    return render_template_string(INDEX_TEMPLATE)

@app.route('/evaluate', methods=['POST'])
def legacy_evaluate():
    # Helper to support persistent browser caches calling the legacy endpoint
    return wqi_evaluate()

@app.route('/wqi/evaluate', methods=['POST'])
def wqi_evaluate():
    d = request.get_json(force=True)
    res = evaluate_wqi(
        ph_val=d.get('ph', 7.0),
        turb_val=d.get('turbidity', 10.0),
        do_val=d.get('do_level', 9.0),
        temp_val=d.get('temperature', 22.0),
        cond_val=d.get('conductivity', 250.0)
    )
    _last_wqi_assessment.clear()
    _last_wqi_assessment.update(res)
    return jsonify(res)

@app.route('/wqi/report', methods=['POST'])
def wqi_report():
    if not _last_wqi_assessment:
        return jsonify({"error": "Run an assessment first."}), 400
    pdf_bytes = make_wqi_pdf(_last_wqi_assessment)
    fn = f"AquaIQ_WaterQuality_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(io.BytesIO(pdf_bytes), mimetype='application/pdf', as_attachment=True, download_name=fn)

@app.route('/wqi/rules', methods=['GET', 'POST'])
def wqi_rules_handler():
    global wqi_rules, _cached_wqi_surface
    if request.method == 'POST':
        wqi_rules = request.get_json(force=True)
        _cached_wqi_surface = None  # Invalidate cache
        return jsonify({"status": "ok"})
    return jsonify(wqi_rules)

@app.route('/wqi/rules/reset', methods=['POST'])
def wqi_rules_reset():
    global wqi_rules, _cached_wqi_surface
    wqi_rules = [r[:] for r in WQI_DEFAULT_RULES]
    _cached_wqi_surface = None
    return jsonify(wqi_rules)

@app.route('/pid/rules', methods=['GET', 'POST'])
def pid_rules_handler():
    global pid_rules, _cached_pid_surface
    if request.method == 'POST':
        pid_rules = request.get_json(force=True)
        _cached_pid_surface = None  # Invalidate cache
        return jsonify({"status": "ok"})
    return jsonify(pid_rules)

@app.route('/pid/rules/reset', methods=['POST'])
def pid_rules_reset():
    global pid_rules, _cached_pid_surface
    pid_rules = [r[:] for r in PID_DEFAULT_RULES]
    _cached_pid_surface = None
    return jsonify(pid_rules)

@app.route('/pid/simulate', methods=['POST'])
def pid_simulate():
    d = request.get_json(force=True)
    sp = float(d.get('setpoint', 50.0))
    init = float(d.get('initial_temp', 20.0))
    dist = float(d.get('disturbance', 2.0))
    dur = int(d.get('duration', 300))
    
    sim = run_pid_simulation(sp, init, dist, duration=dur)
    
    # Generate default charts at k=15
    k = 15
    charts = generate_pid_timestep_charts(sim['e'][k], sim['de'][k], sim['ie'][k], sim['d'][k], sim['u'][k])
    
    _last_pid_simulation.clear()
    _last_pid_simulation.update({
        "sim": sim, "charts": charts,
        "setpoint": sp, "initial_temp": init, "disturbance": dist
    })
    
    return jsonify({
        "simulation": sim,
        "charts": charts
    })

@app.route('/pid/timestep', methods=['POST'])
def pid_timestep():
    d = request.get_json(force=True)
    charts = generate_pid_timestep_charts(
        d.get('e', 0.0), d.get('de', 0.0), d.get('ie', 0.0), d.get('d', 2.0), d.get('u', 50.0)
    )
    return jsonify(charts)

@app.route('/pid/report', methods=['POST'])
def pid_report():
    if not _last_pid_simulation:
        return jsonify({"error": "Execute simulation first."}), 400
    pdf_bytes = make_pid_pdf(_last_pid_simulation)
    fn = f"AquaIQ_PID_Simulation_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(io.BytesIO(pdf_bytes), mimetype='application/pdf', as_attachment=True, download_name=fn)

# ══════════════════════════════════════════════════════════════════════
# 10. MAIN PROCESS START
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("+------------------------------------------------+")
    print("|   AquaIQ v3 - Fuzzy Dual Control & Simulation  |")
    print("+------------------------------------------------+")
    print("|   Server Running: http://127.0.0.1:5000        |")
    print("+------------------------------------------------+")
    app.run(debug=False, port=5000)