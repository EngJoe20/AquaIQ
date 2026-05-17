"""
╔══════════════════════════════════════════════════════════════════════╗
║       Smart Water Quality Assessment — Fuzzy Logic System           ║
║       Built with scikit-fuzzy + Flask + Microsoft Fluent Design     ║
╚══════════════════════════════════════════════════════════════════════╝

pip install scikit-fuzzy numpy matplotlib flask
"""

# ══════════════════════════════════════════════════════════════════════
# 1. IMPORTS
# ══════════════════════════════════════════════════════════════════════
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (no GUI window)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import base64
from flask import Flask, render_template_string, request, jsonify

# ══════════════════════════════════════════════════════════════════════
# 2. UNIVERSE OF DISCOURSE
#    Each variable needs a fine-grained NumPy array as its "x-axis"
# ══════════════════════════════════════════════════════════════════════
x_ph          = np.arange(0,   14.01, 0.01)   # pH            0–14
x_turbidity   = np.arange(0,  100.01, 0.1)    # Turbidity     0–100 NTU
x_do          = np.arange(0,   20.01, 0.01)   # Dissolved O₂  0–20 mg/L
x_temp        = np.arange(0,   50.01, 0.1)    # Temperature   0–50 °C
x_conductivity= np.arange(0, 2000.01, 1.0)    # Conductivity  0–2000 µS/cm
x_wqi         = np.arange(0,  100.01, 0.1)    # WQI output    0–100

# ══════════════════════════════════════════════════════════════════════
# 3. MEMBERSHIP FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

# ── 3A. pH  ─────────────────────────────────────────────────────────
# pH 7 is neutral; below 6.5 is acidic, above 8.5 is alkaline.
# Trapezoidal edges ensure full membership at extremes.
mf_ph = {
    "acidic":    fuzz.trapmf(x_ph, [0,   0,   5.5, 6.5]),   # very low pH
    "neutral":   fuzz.trimf( x_ph, [6.0, 7.0, 8.0]),         # ideal range
    "alkaline":  fuzz.trapmf(x_ph, [7.5, 8.5, 14,  14]),     # high pH
}

# ── 3B. Turbidity  ──────────────────────────────────────────────────
# WHO drinking-water guideline: <1 NTU; rivers may be 0–100 NTU.
mf_turb = {
    "clear":    fuzz.trapmf(x_turbidity, [0,  0,  5,  15]),
    "moderate": fuzz.trimf( x_turbidity, [10, 35, 60]),
    "cloudy":   fuzz.trapmf(x_turbidity, [50, 70, 100, 100]),
}

# ── 3C. Dissolved Oxygen  ───────────────────────────────────────────
# >8 mg/L = excellent; 5–8 = good; <4 = stressed / hypoxic.
mf_do = {
    "low":      fuzz.trapmf(x_do, [0,  0,  3,  5]),
    "medium":   fuzz.trimf( x_do, [4,  7,  10]),
    "high":     fuzz.trapmf(x_do, [8,  11, 20, 20]),
}

# ── 3D. Temperature  ────────────────────────────────────────────────
# Ideal aquatic life range: 10–25 °C. Very hot or cold is stressful.
mf_temp = {
    "cold":     fuzz.trapmf(x_temp, [0,  0,  8,  15]),
    "optimal":  fuzz.trimf( x_temp, [12, 22, 30]),
    "hot":      fuzz.trapmf(x_temp, [25, 35, 50, 50]),
}

# ── 3E. Conductivity  ───────────────────────────────────────────────
# Good drinking water: <400 µS/cm. High = dissolved salts / pollution.
mf_cond = {
    "low":      fuzz.trapmf(x_conductivity, [0,    0,   200,  400]),
    "medium":   fuzz.trimf( x_conductivity, [300,  800, 1200]),
    "high":     fuzz.trapmf(x_conductivity, [1000, 1400, 2000, 2000]),
}

# ── 3F. Water Quality Index (OUTPUT)  ───────────────────────────────
# Five linguistic categories across 0–100.
mf_wqi = {
    "very_poor":   fuzz.trapmf(x_wqi, [0,  0,  10, 25]),
    "poor":        fuzz.trimf( x_wqi, [15, 30, 45]),
    "acceptable":  fuzz.trimf( x_wqi, [35, 50, 65]),
    "good":        fuzz.trimf( x_wqi, [55, 70, 85]),
    "excellent":   fuzz.trapmf(x_wqi, [75, 90, 100, 100]),
}

# ══════════════════════════════════════════════════════════════════════
# 4. FUZZY RULES  (15 rules — Mamdani inference)
#    Each rule is a tuple:
#    (ph_label, turb_label, do_label, temp_label, cond_label, wqi_label)
#    None = "don't care" for that antecedent
# ══════════════════════════════════════════════════════════════════════
RULES = [
    # ── EXCELLENT CONDITIONS ─────────────────────────────────────────
    # R1: Neutral pH + clear + high DO + optimal temp + low cond → Excellent
    ("neutral",   "clear",    "high",   "optimal", "low",    "excellent"),

    # R2: Neutral pH + clear + high DO + any temp + medium cond → Good
    ("neutral",   "clear",    "high",   None,      "medium", "good"),

    # ── GOOD CONDITIONS ──────────────────────────────────────────────
    # R3: Neutral pH + moderate turbidity + high DO + optimal → Good
    ("neutral",   "moderate", "high",   "optimal", None,     "good"),

    # R4: Slightly off pH + clear water + medium DO + optimal → Good
    (None,        "clear",    "medium", "optimal", "low",    "good"),

    # R5: Neutral pH + clear + medium DO + cold temp → Acceptable
    ("neutral",   "clear",    "medium", "cold",    None,     "acceptable"),

    # ── ACCEPTABLE CONDITIONS ─────────────────────────────────────────
    # R6: Neutral pH + moderate + medium DO + optimal → Acceptable
    ("neutral",   "moderate", "medium", "optimal", "medium", "acceptable"),

    # R7: Any pH + clear + medium DO + hot temp → Acceptable
    (None,        "clear",    "medium", "hot",     None,     "acceptable"),

    # R8: Neutral + moderate + medium + cold → Acceptable
    ("neutral",   "moderate", "medium", "cold",    None,     "acceptable"),

    # ── POOR CONDITIONS ──────────────────────────────────────────────
    # R9: Acidic pH (low) → immediately Poor regardless of rest
    ("acidic",    None,       None,     None,      None,     "poor"),

    # R10: Alkaline pH → Poor (too basic)
    ("alkaline",  None,       None,     None,      None,     "poor"),

    # R11: High turbidity + low DO → Poor water, heavy pollution likely
    (None,        "cloudy",   "low",    None,      None,     "poor"),

    # R12: High conductivity (salty/polluted) + cloudy → Poor
    (None,        "cloudy",   None,     None,      "high",   "poor"),

    # ── VERY POOR CONDITIONS ─────────────────────────────────────────
    # R13: Acidic + cloudy + low DO → Very Poor (severe contamination)
    ("acidic",    "cloudy",   "low",    None,      None,     "very_poor"),

    # R14: Alkaline + high conductivity + low DO → Very Poor
    ("alkaline",  None,       "low",    None,      "high",   "very_poor"),

    # R15: Very hot + cloudy + low DO (algal bloom scenario) → Very Poor
    (None,        "cloudy",   "low",    "hot",     None,     "very_poor"),
]

# ══════════════════════════════════════════════════════════════════════
# 5. CONTROL SYSTEM SETUP  (using skfuzzy ctrl API)
# ══════════════════════════════════════════════════════════════════════
def build_ctrl_system():
    """
    Build the scikit-fuzzy Antecedent/Consequent objects and
    assemble the ControlSystem with all 15 rules.
    Returns a ControlSystemSimulation ready for input assignment.
    """
    # Antecedents (inputs)
    ph          = ctrl.Antecedent(x_ph,          'ph')
    turbidity   = ctrl.Antecedent(x_turbidity,   'turbidity')
    do_level    = ctrl.Antecedent(x_do,          'do_level')
    temperature = ctrl.Antecedent(x_temp,        'temperature')
    conductivity= ctrl.Antecedent(x_conductivity,'conductivity')

    # Consequent (output)
    wqi = ctrl.Consequent(x_wqi, 'wqi', defuzzify_method='centroid')

    # Assign membership functions
    ph['acidic']    = mf_ph['acidic']
    ph['neutral']   = mf_ph['neutral']
    ph['alkaline']  = mf_ph['alkaline']

    turbidity['clear']    = mf_turb['clear']
    turbidity['moderate'] = mf_turb['moderate']
    turbidity['cloudy']   = mf_turb['cloudy']

    do_level['low']    = mf_do['low']
    do_level['medium'] = mf_do['medium']
    do_level['high']   = mf_do['high']

    temperature['cold']    = mf_temp['cold']
    temperature['optimal'] = mf_temp['optimal']
    temperature['hot']     = mf_temp['hot']

    conductivity['low']    = mf_cond['low']
    conductivity['medium'] = mf_cond['medium']
    conductivity['high']   = mf_cond['high']

    wqi['very_poor']  = mf_wqi['very_poor']
    wqi['poor']       = mf_wqi['poor']
    wqi['acceptable'] = mf_wqi['acceptable']
    wqi['good']       = mf_wqi['good']
    wqi['excellent']  = mf_wqi['excellent']

    # Map label strings to antecedent objects
    antecedent_map = {
        0: ph, 1: turbidity, 2: do_level, 3: temperature, 4: conductivity
    }

    # Build ctrl.Rule objects from the RULES table
    ctrl_rules = []
    for rule in RULES:
        ph_lbl, turb_lbl, do_lbl, temp_lbl, cond_lbl, out_lbl = rule
        antecedents = []
        if ph_lbl:   antecedents.append(ph[ph_lbl])
        if turb_lbl: antecedents.append(turbidity[turb_lbl])
        if do_lbl:   antecedents.append(do_level[do_lbl])
        if temp_lbl: antecedents.append(temperature[temp_lbl])
        if cond_lbl: antecedents.append(conductivity[cond_lbl])

        # AND all antecedents together
        combined = antecedents[0]
        for ant in antecedents[1:]:
            combined = combined & ant

        ctrl_rules.append(ctrl.Rule(combined, wqi[out_lbl]))

    system   = ctrl.ControlSystem(ctrl_rules)
    simulation = ctrl.ControlSystemSimulation(system)
    return simulation

# ══════════════════════════════════════════════════════════════════════
# 6. SIMULATION FUNCTION
# ══════════════════════════════════════════════════════════════════════
def get_category(wqi_value: float) -> tuple[str, str, str]:
    """
    Map a numeric WQI to:
    - category label
    - treatment recommendation
    - hex color for UI badge
    """
    if wqi_value >= 80:
        return "Excellent", "No treatment required. Water is safe for all uses.", "#107C10"
    elif wqi_value >= 60:
        return "Good", "Minor filtration recommended. Suitable for most uses.", "#498205"
    elif wqi_value >= 40:
        return "Acceptable", "Standard treatment (coagulation + filtration + chlorination) needed.", "#FF8C00"
    elif wqi_value >= 20:
        return "Poor", "Advanced treatment required: multi-stage filtration + UV disinfection.", "#D83B01"
    else:
        return "Very Poor", "Severe treatment needed: reverse osmosis + chemical neutralisation + monitoring.", "#A80000"


def evaluate(ph_val, turb_val, do_val, temp_val, cond_val):
    """
    Run the complete Mamdani fuzzy inference pipeline and return results.

    Parameters
    ----------
    ph_val   : float  — pH reading (0–14)
    turb_val : float  — Turbidity NTU (0–100)
    do_val   : float  — Dissolved Oxygen mg/L (0–20)
    temp_val : float  — Temperature °C (0–50)
    cond_val : float  — Conductivity µS/cm (0–2000)

    Returns
    -------
    dict with wqi_value, category, recommendation, chart_b64
    """
    sim = build_ctrl_system()

    sim.input['ph']           = float(ph_val)
    sim.input['turbidity']    = float(turb_val)
    sim.input['do_level']     = float(do_val)
    sim.input['temperature']  = float(temp_val)
    sim.input['conductivity'] = float(cond_val)

    sim.compute()
    wqi_value = float(sim.output['wqi'])
    wqi_value = max(0.0, min(100.0, wqi_value))   # clamp

    category, recommendation, color = get_category(wqi_value)
    chart_b64 = generate_chart(wqi_value)

    return {
        "wqi":            round(wqi_value, 2),
        "category":       category,
        "recommendation": recommendation,
        "color":          color,
        "chart":          chart_b64,
    }


# ══════════════════════════════════════════════════════════════════════
# 7A. CHART GENERATION
# ══════════════════════════════════════════════════════════════════════
def generate_chart(wqi_value: float) -> str:
    """
    Render all five output membership functions and mark the
    defuzzified centroid with a vertical line.
    Returns a base-64 PNG string for embedding in HTML.

    Centroid formula implemented by skfuzzy:
        z* = ∫ μ(z)·z dz  /  ∫ μ(z) dz
    """
    colors = {
        "very_poor":  "#c50f1f",
        "poor":       "#d83b01",
        "acceptable": "#ff8c00",
        "good":       "#498205",
        "excellent":  "#107c10",
    }

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#f3f2f1')
    ax.set_facecolor('#faf9f8')

    for label, mf in mf_wqi.items():
        ax.fill_between(x_wqi, 0, mf, alpha=0.25, color=colors[label])
        ax.plot(x_wqi, mf, color=colors[label], lw=2,
                label=label.replace('_', ' ').title())

    ax.axvline(wqi_value, color='#0078d4', lw=2.5, linestyle='--',
               label=f'WQI = {wqi_value:.1f} (Centroid)')
    ax.fill_betweenx([0, 1], wqi_value - 0.5, wqi_value + 0.5,
                     color='#0078d4', alpha=0.35)

    ax.set_xlabel('Water Quality Index', fontsize=11, color='#323130')
    ax.set_ylabel('Membership Degree μ(z)', fontsize=11, color='#323130')
    ax.set_title('Output Membership Functions — Defuzzified Result (Centroid)',
                 fontsize=12, color='#201f1e', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=':', alpha=0.4, color='#c8c6c4')
    ax.tick_params(colors='#605e5c')
    for spine in ax.spines.values():
        spine.set_edgecolor('#e1dfdd')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ══════════════════════════════════════════════════════════════════════
# 7B. WEB INTERFACE — Flask + Fluent Design HTML
# ══════════════════════════════════════════════════════════════════════
HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>AquaIQ — Smart Water Quality Assessment</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&family=Segoe+UI+Variable:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
<style>
/* ── Fluent Design Tokens ── */
:root {
  --brand-primary:      #0078d4;
  --brand-dark:         #005a9e;
  --brand-light:        #deecf9;
  --surface-00:         #ffffff;
  --surface-01:         #faf9f8;
  --surface-02:         #f3f2f1;
  --surface-03:         #edebe9;
  --border:             #e1dfdd;
  --border-strong:      #c8c6c4;
  --text-primary:       #201f1e;
  --text-secondary:     #605e5c;
  --text-disabled:      #a19f9d;
  --shadow-2:  0 1.6px 3.6px 0 rgba(0,0,0,.132), 0 .3px .9px 0 rgba(0,0,0,.108);
  --shadow-8:  0 6.4px 14.4px 0 rgba(0,0,0,.132), 0 1.2px 3.6px 0 rgba(0,0,0,.108);
  --shadow-16: 0 12.8px 28.8px 0 rgba(0,0,0,.132), 0 2.4px 7.2px 0 rgba(0,0,0,.108);
  --radius-s:  4px;
  --radius-m:  8px;
  --radius-l:  12px;
  --radius-xl: 20px;
  --font: 'Segoe UI', 'Segoe UI Variable', system-ui, -apple-system, sans-serif;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }

body {
  font-family: var(--font);
  background: var(--surface-01);
  color: var(--text-primary);
  min-height: 100vh;
}

/* ── Hero Header ── */
.hero {
  background: linear-gradient(135deg, #0078d4 0%, #005a9e 40%, #003966 100%);
  color: #fff;
  padding: 40px 48px 56px;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(circle at 20% 50%, rgba(255,255,255,.08) 0%, transparent 60%),
    radial-gradient(circle at 80% 20%, rgba(0,120,212,.4) 0%, transparent 50%);
}
.hero-inner { position: relative; max-width: 1140px; margin: 0 auto; }
.hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(255,255,255,.15);
  border: 1px solid rgba(255,255,255,.25);
  border-radius: 20px;
  padding: 4px 14px;
  font-size: 12px; font-weight: 600; letter-spacing: .05em;
  margin-bottom: 16px;
  backdrop-filter: blur(8px);
}
.hero h1 { font-size: 36px; font-weight: 700; letter-spacing: -.5px; margin-bottom: 8px; }
.hero p  { font-size: 16px; opacity: .85; max-width: 520px; line-height: 1.6; }
.hero-dots {
  position: absolute; right: 0; top: -20px;
  display: grid; grid-template-columns: repeat(8,1fr); gap: 18px;
  opacity: .12;
}
.hero-dots span {
  width: 6px; height: 6px;
  background: #fff; border-radius: 50%;
  display: block;
}

/* ── Main layout ── */
.main { max-width: 1140px; margin: 0 auto; padding: 40px 24px 80px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
@media(max-width:800px){ .grid { grid-template-columns: 1fr; } }

/* ── Card ── */
.card {
  background: var(--surface-00);
  border: 1px solid var(--border);
  border-radius: var(--radius-l);
  box-shadow: var(--shadow-2);
  overflow: hidden;
  transition: box-shadow .2s;
}
.card:hover { box-shadow: var(--shadow-8); }
.card-header {
  padding: 20px 24px 16px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 12px;
}
.card-icon {
  width: 36px; height: 36px; border-radius: 8px;
  background: var(--brand-light);
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; flex-shrink: 0;
}
.card-header h2 { font-size: 16px; font-weight: 600; color: var(--text-primary); }
.card-header p  { font-size: 12px; color: var(--text-secondary); margin-top: 2px; }
.card-body { padding: 24px; }

/* ── Slider Control ── */
.control { margin-bottom: 28px; }
.control:last-child { margin-bottom: 0; }
.ctrl-label {
  display: flex; justify-content: space-between; align-items: baseline;
  margin-bottom: 8px;
}
.ctrl-name { font-size: 14px; font-weight: 600; color: var(--text-primary); }
.ctrl-unit { font-size: 12px; color: var(--text-secondary); }
.ctrl-value {
  font-size: 18px; font-weight: 700; color: var(--brand-primary);
  min-width: 60px; text-align: right;
}
.slider-track { position: relative; }
input[type=range] {
  -webkit-appearance: none;
  width: 100%; height: 4px;
  background: var(--surface-03);
  border-radius: 2px; outline: none; cursor: pointer;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 20px; height: 20px;
  background: var(--brand-primary);
  border: 2px solid #fff;
  border-radius: 50%;
  box-shadow: var(--shadow-2);
  transition: transform .15s, box-shadow .15s;
}
input[type=range]::-webkit-slider-thumb:hover {
  transform: scale(1.2);
  box-shadow: var(--shadow-8);
}
input[type=range]:focus::-webkit-slider-thumb {
  box-shadow: 0 0 0 3px rgba(0,120,212,.3);
}
.range-labels {
  display: flex; justify-content: space-between;
  font-size: 11px; color: var(--text-disabled);
  margin-top: 4px;
}

/* ── Button ── */
.btn-compute {
  width: 100%;
  padding: 14px;
  background: var(--brand-primary);
  color: #fff;
  border: none; border-radius: var(--radius-m);
  font-size: 15px; font-weight: 600;
  cursor: pointer;
  display: flex; align-items: center; justify-content: center; gap: 8px;
  box-shadow: var(--shadow-2);
  transition: background .15s, transform .1s, box-shadow .15s;
  margin-top: 8px;
}
.btn-compute:hover  { background: var(--brand-dark); box-shadow: var(--shadow-8); }
.btn-compute:active { transform: scale(.98); }
.btn-compute.loading { opacity: .7; pointer-events: none; }
.spinner {
  width: 16px; height: 16px; border: 2px solid rgba(255,255,255,.4);
  border-top-color: #fff; border-radius: 50%;
  animation: spin .7s linear infinite; display: none;
}
.btn-compute.loading .spinner { display: block; }

/* ── Result Panel ── */
.result-placeholder {
  text-align: center; padding: 48px 24px;
  color: var(--text-disabled);
}
.result-placeholder .icon { font-size: 48px; margin-bottom: 16px; display: block; }
.result-placeholder p { font-size: 14px; line-height: 1.6; }

.result-section { animation: fadeUp .35s ease; }
@keyframes fadeUp {
  from { opacity:0; transform: translateY(12px); }
  to   { opacity:1; transform: translateY(0); }
}

.wqi-gauge {
  display: flex; flex-direction: column; align-items: center;
  padding: 28px 24px 20px;
  border-bottom: 1px solid var(--border);
}
.wqi-ring {
  position: relative; width: 140px; height: 140px; margin-bottom: 16px;
}
.wqi-ring svg { width: 140px; height: 140px; transform: rotate(-90deg); }
.wqi-ring circle.bg  { fill: none; stroke: var(--surface-03); stroke-width: 10; }
.wqi-ring circle.fg  {
  fill: none; stroke-width: 10;
  stroke-linecap: round;
  stroke-dasharray: 345;
  transition: stroke-dashoffset 1s cubic-bezier(.4,0,.2,1), stroke .4s;
}
.wqi-center {
  position: absolute; inset: 0;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
}
.wqi-number { font-size: 36px; font-weight: 700; line-height: 1; }
.wqi-label  { font-size: 11px; color: var(--text-secondary); margin-top: 2px; }

.badge {
  display: inline-block;
  padding: 4px 14px;
  border-radius: 20px;
  font-size: 13px; font-weight: 600;
  color: #fff;
  margin-bottom: 8px;
}

.recommendation-box {
  margin: 20px 24px;
  padding: 16px;
  background: var(--surface-01);
  border-radius: var(--radius-m);
  border-left: 4px solid var(--brand-primary);
}
.recommendation-box h4 {
  font-size: 12px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .08em; color: var(--text-secondary);
  margin-bottom: 8px;
}
.recommendation-box p { font-size: 14px; line-height: 1.6; color: var(--text-primary); }

/* ── Chart Area ── */
.chart-area { padding: 20px 24px; border-top: 1px solid var(--border); }
.chart-area h4 {
  font-size: 13px; font-weight: 600; color: var(--text-secondary);
  margin-bottom: 12px; text-transform: uppercase; letter-spacing: .06em;
}
.chart-area img { width: 100%; border-radius: var(--radius-s); }

/* ── Info strip ── */
.info-strip {
  background: var(--brand-light);
  border: 1px solid #b3d7f5;
  border-radius: var(--radius-m);
  padding: 16px 20px;
  display: flex; gap: 12px; align-items: flex-start;
  margin-bottom: 24px;
}
.info-strip .info-icon { font-size: 18px; flex-shrink: 0; margin-top: 1px; }
.info-strip p { font-size: 13px; color: #004578; line-height: 1.6; }

/* ── Footer ── */
footer {
  text-align: center; padding: 28px;
  font-size: 12px; color: var(--text-disabled);
  border-top: 1px solid var(--border);
}
</style>
</head>
<body>

<!-- ── Hero ── -->
<div class="hero">
  <div class="hero-dots">
    {% for _ in range(48) %}<span></span>{% endfor %}
  </div>
  <div class="hero-inner">
    <div class="hero-badge">💧 Fuzzy Logic AI System</div>
    <h1>AquaIQ Water Quality Assessment</h1>
    <p>Enter sensor readings below. The Mamdani Fuzzy Inference Engine evaluates 15 rules
       and computes the Water Quality Index via Centroid defuzzification.</p>
  </div>
</div>

<!-- ── Main ── -->
<div class="main">

  <div class="info-strip">
    <span class="info-icon">ℹ️</span>
    <p><strong>How it works:</strong> Your five sensor values are fuzzified using trapezoidal &amp;
    triangular membership functions. 15 IF-THEN Mamdani rules fire in parallel.
    The aggregated output surface is collapsed to a crisp WQI score (0–100) using
    the <em>Centroid (Center of Gravity)</em> method.</p>
  </div>

  <div class="grid">

    <!-- ── INPUT CARD ── -->
    <div class="card">
      <div class="card-header">
        <div class="card-icon">🔬</div>
        <div>
          <h2>Sensor Inputs</h2>
          <p>Adjust all 5 water quality parameters</p>
        </div>
      </div>
      <div class="card-body">

        <!-- pH -->
        <div class="control">
          <div class="ctrl-label">
            <div>
              <span class="ctrl-name">pH Level</span>
              <span class="ctrl-unit"> — Acidity / Alkalinity</span>
            </div>
            <span class="ctrl-value" id="val-ph">7.0</span>
          </div>
          <div class="slider-track">
            <input type="range" id="sl-ph" min="0" max="14" step="0.1" value="7.0"
                   oninput="document.getElementById('val-ph').textContent=parseFloat(this.value).toFixed(1)"/>
          </div>
          <div class="range-labels"><span>0 (Acidic)</span><span>7 (Neutral)</span><span>14 (Alkaline)</span></div>
        </div>

        <!-- Turbidity -->
        <div class="control">
          <div class="ctrl-label">
            <div>
              <span class="ctrl-name">Turbidity</span>
              <span class="ctrl-unit"> — NTU</span>
            </div>
            <span class="ctrl-value" id="val-turb">10</span>
          </div>
          <input type="range" id="sl-turb" min="0" max="100" step="1" value="10"
                 oninput="document.getElementById('val-turb').textContent=this.value"/>
          <div class="range-labels"><span>0 (Clear)</span><span>50</span><span>100 (Cloudy)</span></div>
        </div>

        <!-- Dissolved Oxygen -->
        <div class="control">
          <div class="ctrl-label">
            <div>
              <span class="ctrl-name">Dissolved Oxygen</span>
              <span class="ctrl-unit"> — mg/L</span>
            </div>
            <span class="ctrl-value" id="val-do">9.0</span>
          </div>
          <input type="range" id="sl-do" min="0" max="20" step="0.1" value="9.0"
                 oninput="document.getElementById('val-do').textContent=parseFloat(this.value).toFixed(1)"/>
          <div class="range-labels"><span>0 (Hypoxic)</span><span>10</span><span>20 (Saturated)</span></div>
        </div>

        <!-- Temperature -->
        <div class="control">
          <div class="ctrl-label">
            <div>
              <span class="ctrl-name">Temperature</span>
              <span class="ctrl-unit"> — °C</span>
            </div>
            <span class="ctrl-value" id="val-temp">22</span>
          </div>
          <input type="range" id="sl-temp" min="0" max="50" step="1" value="22"
                 oninput="document.getElementById('val-temp').textContent=this.value"/>
          <div class="range-labels"><span>0 °C</span><span>25 °C</span><span>50 °C</span></div>
        </div>

        <!-- Conductivity -->
        <div class="control">
          <div class="ctrl-label">
            <div>
              <span class="ctrl-name">Conductivity</span>
              <span class="ctrl-unit"> — µS/cm</span>
            </div>
            <span class="ctrl-value" id="val-cond">250</span>
          </div>
          <input type="range" id="sl-cond" min="0" max="2000" step="10" value="250"
                 oninput="document.getElementById('val-cond').textContent=this.value"/>
          <div class="range-labels"><span>0</span><span>1000</span><span>2000 µS/cm</span></div>
        </div>

        <button class="btn-compute" id="btn" onclick="compute()">
          <div class="spinner" id="spinner"></div>
          <span id="btn-text">⚡ Evaluate Water Quality</span>
        </button>

      </div>
    </div>

    <!-- ── OUTPUT CARD ── -->
    <div class="card">
      <div class="card-header">
        <div class="card-icon">📊</div>
        <div>
          <h2>Assessment Results</h2>
          <p>Fuzzy inference output &amp; treatment recommendation</p>
        </div>
      </div>

      <div id="result-area">
        <div class="result-placeholder">
          <span class="icon">🔵</span>
          <p>Set your sensor values and press<br/><strong>"Evaluate Water Quality"</strong><br/>to run the fuzzy inference engine.</p>
        </div>
      </div>
    </div>

  </div><!-- /grid -->
</div><!-- /main -->

<footer>
  AquaIQ — Mamdani Fuzzy Logic System &nbsp;·&nbsp; scikit-fuzzy + Flask &nbsp;·&nbsp; Microsoft Fluent Design
</footer>

<script>
const COLORS = {
  'Excellent': '#107c10',
  'Good':      '#498205',
  'Acceptable':'#ff8c00',
  'Poor':      '#d83b01',
  'Very Poor': '#a80000',
};

async function compute() {
  const btn = document.getElementById('btn');
  btn.classList.add('loading');
  document.getElementById('btn-text').textContent = 'Computing…';

  const payload = {
    ph:           parseFloat(document.getElementById('sl-ph').value),
    turbidity:    parseFloat(document.getElementById('sl-turb').value),
    do_level:     parseFloat(document.getElementById('sl-do').value),
    temperature:  parseFloat(document.getElementById('sl-temp').value),
    conductivity: parseFloat(document.getElementById('sl-cond').value),
  };

  try {
    const res  = await fetch('/evaluate', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    renderResult(data);
  } catch(e) {
    alert('Server error: ' + e.message);
  } finally {
    btn.classList.remove('loading');
    document.getElementById('btn-text').textContent = '⚡ Evaluate Water Quality';
  }
}

function renderResult(d) {
  const color = COLORS[d.category] || '#0078d4';
  // Ring: circumference 345, offset = 345 - (wqi/100)*345
  const offset = 345 - (d.wqi / 100) * 345;

  document.getElementById('result-area').innerHTML = `
    <div class="result-section">
      <div class="wqi-gauge">
        <div class="wqi-ring">
          <svg viewBox="0 0 120 120">
            <circle class="bg" cx="60" cy="60" r="55"/>
            <circle class="fg" cx="60" cy="60" r="55"
              id="ring-fg"
              stroke="${color}"
              stroke-dashoffset="345"
              style="stroke-dashoffset:${offset}"/>
          </svg>
          <div class="wqi-center">
            <span class="wqi-number" style="color:${color}">${d.wqi}</span>
            <span class="wqi-label">/ 100</span>
          </div>
        </div>
        <span class="badge" style="background:${color}">${d.category}</span>
        <span style="font-size:13px;color:var(--text-secondary)">Water Quality Index</span>
      </div>

      <div class="recommendation-box">
        <h4>🔧 Treatment Recommendation</h4>
        <p>${d.recommendation}</p>
      </div>

      <div class="chart-area">
        <h4>Output Membership Function</h4>
        <img src="data:image/png;base64,${d.chart}" alt="WQI membership chart"/>
      </div>
    </div>
  `;

  // Animate ring after render
  setTimeout(() => {
    const ring = document.getElementById('ring-fg');
    if (ring) ring.style.transition = 'stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)';
  }, 50);
}
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════════
# 8. FLASK APP + ROUTES
# ══════════════════════════════════════════════════════════════════════
app = Flask(__name__)

@app.route('/')
def index():
    """Serve the main Fluent Design HTML page."""
    return render_template_string(HTML_PAGE)

@app.route('/evaluate', methods=['POST'])
def evaluate_route():
    """
    POST /evaluate
    Accepts JSON with ph, turbidity, do_level, temperature, conductivity.
    Returns JSON with wqi, category, recommendation, color, chart (base64 PNG).
    """
    data = request.get_json(force=True)
    result = evaluate(
        ph_val   = data.get('ph',           7.0),
        turb_val = data.get('turbidity',    10.0),
        do_val   = data.get('do_level',     9.0),
        temp_val = data.get('temperature',  22.0),
        cond_val = data.get('conductivity', 250.0),
    )
    return jsonify(result)

# ══════════════════════════════════════════════════════════════════════
# 9. MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("╔══════════════════════════════════════════╗")
    print("║  AquaIQ — Fuzzy Logic Water Assessment  ║")
    print("╠══════════════════════════════════════════╣")
    print("║  🌐  http://127.0.0.1:5000              ║")
    print("╚══════════════════════════════════════════╝")
    app.run(debug=False, port=5000)