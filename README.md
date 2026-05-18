# 💧 AquaIQ v3 — Unified Fuzzy Logic Control Panel

AquaIQ v3 is a state-of-the-art, dual-architecture engineering application that implements high-fidelity **Mamdani Fuzzy Logic Inference Systems** to solve two core engineering problems: **Multi-Variable Smart Water Quality Assessment** and **Closed-Loop Thermodynamic Process Control (Fuzzy PID)**.

The application features a responsive, premium glassmorphic dashboard built using HTML5, modern HSL design tokens, Vanilla CSS Grid, and dynamic client-side animations. It provides real-time Matplotlib-rendered vector diagnostics (including 3D control surfaces and radar charts), an interactive timeline scrubbing sandbox, side-by-side textual and visual rule compilers, and strict PDF-only reporting pipelines.

---

## 🚀 Quick Start (Windows Launcher)

To run the application immediately, simply double-click the Windows batch file in your repository:
*   [run_aquaiq.bat](file:///f:/CSE%203rd/Digital_control/run_aquaiq.bat)

This script will automatically:
1. Verify your local Python environment (Python 3.8+ required).
2. Check and dynamically install missing package dependencies (`numpy`, `scipy`, `scikit-fuzzy`, `matplotlib`, `flask`, `reportlab`).
3. Automatically launch your default web browser to [http://127.0.0.1:5000/](http://127.0.0.1:5000/).
4. Launch the local Flask server.

---

## 📐 Mathematical & Physics Foundations

### 1. The Mamdani Fuzzy Inference Pipeline
Both control sub-systems (WQI & PID) operate under a multi-stage Mamdani Fuzzy Logic Controller (FLC) following standard fuzzy set mathematics:

*   **Fuzzification:** Telemetry inputs $x_i$ are mapped to a degree of membership $\mu_{A_i}(x_i) \in [0, 1]$ using customized triangular and trapezoidal membership functions:
    $$\mu_{\text{tri}}(x; a, b, c) = \max\left(0, \min\left(\frac{x-a}{b-a}, \frac{c-x}{c-b}\right)\right)$$
    $$\mu_{\text{trap}}(x; a, b, c, d) = \max\left(0, \min\left(\frac{x-a}{b-a}, 1, \frac{d-x}{d-c}\right)\right)$$
*   **Rule Inference Engine:** Rule firing strengths $\alpha_r$ are evaluated using the Mamdani **Min (Intersection)** operator:
    $$\alpha_r = \min\left(\mu_{A_{1r}}(x_1), \mu_{A_{2r}}(x_2), \dots, \mu_{A_{nr}}(x_n)\right)$$
*   **Aggregation:** Rule output membership functions are combined using the **Max (Union)** operator:
    $$\mu_{\text{agg}}(y) = \max_{r}\left(\min\left(\alpha_r, \mu_{B_r}(y)\right)\right)$$
*   **Defuzzification (Centroid Method):** Aggregated fuzzy output curves are integrated to calculate a crisp physical command ($y^*$) representing the **Center of Gravity (CoG)**:
    $$y^* = \frac{\int \mu_{\text{agg}}(y) \cdot y \, dy}{\int \mu_{\text{agg}}(y) \, dy}$$
    *Note: If custom rules are defined such that no rules fire ($\int \mu_{\text{agg}}(y) \, dy = 0$), the backend intercepts the centroid calculation crash and applies a safe fallback rating ($50\%$) to guarantee robust uptime.*

### 2. Thermodynamic Process Closed-Loop Physics
The Fuzzy PID Simulator regulates the temperature of a dynamic liquid system subjected to incoming cold-water load disturbances. The process variable (PV), Water Temperature $T(k)$, is updated at each timestep $k$ using a 2nd-order thermodynamic energy balance:

$$T(k+1) = T(k) + \alpha \cdot (T_{\text{ambient}} - T(k)) + \beta \cdot \frac{u(k)}{10} - \gamma \cdot d(k) \cdot (T(k) - T_{\text{inlet}})$$

Where:
*   $T(k)$ is the current Water Temperature Process Variable (°C).
*   $T_{\text{ambient}} = 22.0$ °C is the ambient air temperature.
*   $\alpha = 0.03$ is the passive thermal loss rate coefficient.
*   $u(k) \in [0, 100]\%$ is the control power supplied by the heater actuator.
*   $\beta = 0.35$ is the heater electrical thermal conversion efficiency.
*   $d(k) \in [0, 10]$ is the cold-water inlet flow rate (disturbance load).
*   $T_{\text{inlet}} = 10.0$ °C is the temperature of the incoming cold water.
*   $\gamma = 0.05$ is the convective thermal absorption factor of the incoming disturbance fluid.

---

## 📋 Dual System Architectures

### Sub-System A: Smart Water Quality Assessment (WQI)
Designed to analyze 5 physical, chemical, and biological sensor telemetry inputs to yield a unified Water Quality Index rating and direct corresponding biological treatment directives.
*   **Inputs:**
    1.  *pH Level* (0.0 - 14.0) $\rightarrow$ overlapping sets: `acidic`, `neutral`, `alkaline`
    2.  *Turbidity* (0.0 - 100.0 NTU) $\rightarrow$ overlapping sets: `clear`, `moderate`, `cloudy`
    3.  *Dissolved Oxygen (DO)* (0.0 - 20.0 mg/L) $\rightarrow$ overlapping sets: `low`, `medium`, `high`
    4.  *Temperature* (0.0 - 50.0 °C) $\rightarrow$ overlapping sets: `cold`, `optimal`, `hot`
    5.  *Conductivity* (0.0 - 2000.0 µS/cm) $\rightarrow$ overlapping sets: `low`, `medium`, `high`
*   **Output:** *Water Quality Index (WQI)* (0.0% - 100.0%) $\rightarrow$ `very_poor`, `poor`, `acceptable`, `good`, `excellent`

---

### Sub-System B: Fuzzy PID Control Loop Simulator (PID)
An advanced intelligent closed-loop control system that replaces static gain-scheduled PID controllers. It manages non-linear dynamics, thermal load shifts, and saturation limits.
*   **Inputs:**
    1.  *Tracking Error* ($e = Setpoint - PV$, -10.0 to +10.0 °C) $\rightarrow$ sets: `negative`, `zero`, `positive`
    2.  *Change in Error* ($de/dt$, -5.0 to +5.0 °C/sec) $\rightarrow$ sets: `negative`, `zero`, `positive`
    3.  *Integral Error* ($\int e \, dt$, -20.0 to +20.0 °C) $\rightarrow$ sets: `negative`, `zero`, `positive`
    4.  *Inflow Disturbance* ($d$, 0.0 to 10.0 rate) $\rightarrow$ sets: `low`, `medium`, `high`
*   **Output:** *Heater Power Duty Cycle* ($u$, 0.0% - 100.0%) $\rightarrow$ `cool_fast`, `cool_slow`, `maintain`, `heat_slow`, `heat_fast`

---

## 📊 Core Diagnostic Diagrams (All 5 Vector Plots)

For deep inspection, both subsystems dynamically render a 5-chart vector analytical payload:

1.  **Output MF (Centroid Balance):** Displays the aggregated fuzzy output membership volume. A dashed vertical cursor marks the exact calculated centroid point ($y^*$).
2.  **Active Firing Matrix:** A horizontal bar chart plotting the activation level ($\alpha_r$) of each of the 15 active rules.
3.  **Breakdown Bars (Correlation):** A vertical bar chart demonstrating the individual membership function (linguistic) values of each input variable.
4.  **3D Control Surface Mapping:** An interactive 3D control surface representing the non-linear transfer landscape ($x_1 \times x_2 \rightarrow y$) generated under active rule bounds.
5.  **Telemetry Radar Chart:** A normalized radar/spider chart showing multi-sensor balance compared to physical saturation limits.

---

## 📋 Quick-Test Presentation Scenarios (WQI & PID)

Use these engineered test cases to demonstrate extreme and ideal operational performance:

### WQI Assessment Test Cases
*   **Pristine Spring Water (Excellent):**
    *   *Inputs:* pH = 7.2, Turbidity = 1.5 NTU, DO = 9.5 mg/L, Temp = 16.0 °C, Conductivity = 110 µS/cm
    *   *Output WQI:* $>85\%$ (`Excellent` - Green)
    *   *Physics Explanation:* Perfect neutral pH, ultra-low turbidity, and high oxygen trigger high-weight rules, mapping the centroid to the maximum output set.
*   **Industrial Pollution Runoff (Very Poor):**
    *   *Inputs:* pH = 3.0, Turbidity = 75.0 NTU, DO = 2.0 mg/L, Temp = 38.0 °C, Conductivity = 1450 µS/cm
    *   *Output WQI:* $<20\%$ (`Very Poor` - Red)
    *   *Physics Explanation:* Highly acidic chemical waste, high mineral density, and extreme thermal dumping trigger active rules for heavy pollution. Recommended for Multi-stage reverse osmosis and biological treatment.

### Fuzzy PID Simulation Test Cases
*   **Cold Start Step-Response:**
    *   *Inputs:* Setpoint = 45.0 °C, Initial Temp = 15.0 °C, Disturbance = 1.5
    *   *System Response:* Temperature (PV) rises rapidly with **zero overshoot** and settles at exactly 45.0 °C. The heater power ($u$) peaks at 100% and then throttles down smoothly to ~50% to maintain steady state.
*   **Sudden Inflow Disturbance (Load Test):**
    *   *Inputs:* Setpoint = 30.0 °C, Initial Temp = 30.0 °C, Disturbance = 8.5
    *   *System Response:* Proactive heater adjustment (Predictive Feedforward). Even though tracking error was initially zero, the controller detects high disturbance inflow and instantly raises heater output ($u$) to ~75% before the water temperature falls, neutralizing the thermal load shift.

---

## 🛠️ Code Architecture & Compatibility

*   **Main Server Engine:** [app.py](file:///f:/CSE%203rd/Digital_control/app.py) handles physical process models, fuzzy controller evaluation, dynamic matplotlib rendering, and anti-cache middleware headers.
*   **Python 3.8.10 Compatibility Patch:** A custom monkeypatch is injected at the top of `app.py` to intercept `ReportLab` MD5 signature calls, resolving `TypeError: 'usedforsecurity' is an invalid keyword argument for openssl_md5()` crashes on older Python environments.
*   **Anti-Caching Middleware:** Implements global browser caching invalidation headers, forcing the client to pull hot-reloaded JS/CSS packages on every frame.
