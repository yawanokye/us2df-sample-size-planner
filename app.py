# app.py
import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="US²DF Sample Size Planner", layout="wide")

# ----------------------------
# Utilities
# ----------------------------
def _norm_ppf(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        return num / den
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        num = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        return num / den

    q = p - 0.5
    r = q*q
    num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
    den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    return num / den

def z_from_alpha(alpha: float) -> float:
    alpha = max(1e-9, min(0.999999999, float(alpha)))
    return _norm_ppf(1 - alpha / 2)

def z_from_power(power: float) -> float:
    power = max(1e-9, min(0.999999999, float(power)))
    return _norm_ppf(power)

def z_value(conf_level: float) -> float:
    alpha = 1 - conf_level
    return _norm_ppf(1 - alpha / 2)

# ----------------------------
# Adam (2020) precision logic
# ----------------------------
def adam_epsilon(rho: float, e: float, t: float) -> float:
    return (rho * e) / t

def adam_n_precision(N: int, epsilon: float) -> int:
    n = N / (1 + N * (epsilon ** 2))
    return int(math.ceil(n))

# ----------------------------
# US²DF Benchmarks (PER GROUP)
# ----------------------------
POWER_BENCHMARKS_PER_GROUP = {"Small": 400, "Medium": 100, "Large": 50}

# Analytical two-proportion (balanced groups)
def n_two_proportions_total(alpha: float, power: float, p1: float, p2: float) -> int:
    p1 = max(1e-6, min(1 - 1e-6, float(p1)))
    p2 = max(1e-6, min(1 - 1e-6, float(p2)))
    d = abs(p1 - p2)
    d = max(d, 1e-9)

    z_a = z_from_alpha(alpha)
    z_b = z_from_power(power)

    pbar = (p1 + p2) / 2
    term1 = z_a * math.sqrt(2 * pbar * (1 - pbar))
    term2 = z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    n_per_group = ((term1 + term2) ** 2) / (d ** 2)
    return int(math.ceil(2 * n_per_group))

# ----------------------------
# Model-based heuristics
# ----------------------------
def green_regression_min(k: int, which: str = "individual") -> int:
    k = max(1, int(k))
    return int(50 + 8*k) if which == "overall" else int(104 + k)

def logistic_epv_min(k: int, event_rate: float, epv: int = 10) -> int:
    k = max(1, int(k))
    epv = max(5, int(epv))
    event_rate = max(1e-9, min(0.999999999, float(event_rate)))
    return int(math.ceil((epv * k) / event_rate))

def approx_cfa_free_params(latents: int, indicators_per_latent: int) -> int:
    L = max(1, int(latents))
    m = max(2, int(indicators_per_latent))
    total_ind = L * m
    loadings = (m - 1) * L
    errors = total_ind
    latent_var_cov = L + (L * (L - 1)) // 2
    return int(loadings + errors + latent_var_cov)

def sem_min_n_by_ratio(params: int, ratio: int = 10) -> int:
    params = max(1, int(params))
    ratio = max(5, int(ratio))
    return int(params * ratio)

# ============================================================
# Sidebar inputs
# ============================================================
st.sidebar.title("Inputs (Step-by-step)")

st.sidebar.markdown("### Step 1, Select applicable components")
use_precision = st.sidebar.checkbox("Precision component", value=True)
use_power = st.sidebar.checkbox("Power component", value=True)
use_model = st.sidebar.checkbox("Model component", value=False)
if not (use_precision or use_power or use_model):
    st.sidebar.error("Select at least one component.")

st.sidebar.markdown("### Step 2, Population and measurement")
N = int(st.sidebar.number_input("Population size (N)", min_value=1, value=50000, step=100))

outcome_type = st.sidebar.selectbox(
    "Measurement Scale of the Estimand",
    ["Categorical (proportions)", "Continuous (means, scales)"],
    index=0
)
st.sidebar.caption("Default is Categorical (proportions).")

conf_level = st.sidebar.radio("Confidence level", ["95%", "99%"], index=0)
conf_level_val = 0.95 if conf_level == "95%" else 0.99
t = z_value(conf_level_val)

# Precision
st.sidebar.markdown("### Step 3, Precision settings (Adam, 2020)")
is_categorical = outcome_type.startswith("Categorical")
rho = 2.0 if is_categorical else 4.0
default_e = 0.05 if is_categorical else 0.03
e_key = "e_cat" if is_categorical else "e_cont"
e = float(st.sidebar.number_input(
    "Desired degree of accuracy (e)",
    min_value=0.001, max_value=0.20,
    value=float(default_e), step=0.001,
    disabled=not use_precision,
    key=e_key
))
epsilon = adam_epsilon(rho=rho, e=e, t=t) if use_precision else None

# Power
st.sidebar.markdown("### Step 4, Power settings")
st.sidebar.caption("Benchmark mode drives n_power by default. Analytical mode is shown as a check unless you override.")

power_mode = st.sidebar.radio(
    "Power mode",
    ["Benchmark drives (recommended)", "Analytical drives (override)"],
    index=0,
    disabled=not use_power
)

alpha = float(st.sidebar.number_input(
    "Significance level (α)",
    min_value=0.001, max_value=0.20,
    value=0.05, step=0.001,
    disabled=not use_power
))
target_power = float(st.sidebar.number_input(
    "Target power (1−β)",
    min_value=0.50, max_value=0.99,
    value=0.80, step=0.01,
    disabled=not use_power
))

effect_size = st.sidebar.radio(
    "Expected effect size",
    ["Small", "Medium", "Large"],
    index=1,
    disabled=not use_power
)

design_type = st.sidebar.selectbox(
    "Power design",
    ["Single group (one sample)", "Two independent groups", "One-way ANOVA (k groups)"],
    index=1,
    disabled=not use_power
)

groups_k = 1
if use_power and design_type in ["Two independent groups", "One-way ANOVA (k groups)"]:
    groups_k = int(st.sidebar.number_input(
        "Number of groups (k)",
        min_value=2,
        value=2 if design_type == "Two independent groups" else 3,
        step=1
    ))

# Optional p1/p2 for analytical two-group categorical
complementary_mode = False
p1, p2 = 0.50, 0.50
show_analytic = False
if use_power and is_categorical and design_type == "Two independent groups":
    show_analytic = True
    st.sidebar.markdown("#### Optional, Complementary proportions")
    complementary_mode = st.sidebar.checkbox("Force p2 = 1 − p1 (complementary)", value=False)

    p1 = float(st.sidebar.number_input(
        "Group 1 proportion (p1)",
        min_value=0.01, max_value=0.99,
        value=0.50, step=0.01, format="%.2f"
    ))

    if complementary_mode:
        p2 = 1.0 - p1
        st.sidebar.number_input(
            "Group 2 proportion (p2 = 1 − p1)",
            min_value=0.01, max_value=0.99,
            value=float(round(p2, 2)),
            step=0.01, format="%.2f",
            disabled=True
        )
    else:
        p2 = float(st.sidebar.number_input(
            "Group 2 proportion (p2)",
            min_value=0.01, max_value=0.99,
            value=0.50, step=0.01, format="%.2f"
        ))

# Model
st.sidebar.markdown("### Step 5, Model settings")
model_context = st.sidebar.selectbox(
    "Model type",
    ["None", "Multiple regression", "Logistic regression", "SEM / CFA"],
    index=0,
    disabled=not use_model
)

k_predictors = 10
event_rate = 0.20
epv = 10
latents = 3
indicators_per_latent = 4
sem_ratio = 10

if use_model and model_context == "Multiple regression":
    k_predictors = int(st.sidebar.number_input("Number of predictors (k)", min_value=1, value=10, step=1))
elif use_model and model_context == "Logistic regression":
    k_predictors = int(st.sidebar.number_input("Number of predictors (k)", min_value=1, value=10, step=1))
    event_rate = float(st.sidebar.number_input("Event rate", min_value=0.01, max_value=0.99, value=0.20, step=0.01))
    epv = int(st.sidebar.number_input("EPV (events per variable)", min_value=5, max_value=50, value=10, step=1))
elif use_model and model_context == "SEM / CFA":
    latents = int(st.sidebar.number_input("Latent variables (L)", min_value=1, value=3, step=1))
    indicators_per_latent = int(st.sidebar.number_input("Indicators per latent (m)", min_value=2, value=4, step=1))
    sem_ratio = int(st.sidebar.number_input("n per parameter ratio", min_value=5, max_value=30, value=10, step=1))

# Field adjustments
st.sidebar.markdown("### Step 6, Field adjustments")
use_deff = st.sidebar.radio("Apply DEFF?", ["No", "Yes"], horizontal=True, key="use_deff")
deff_val = st.sidebar.number_input("DEFF", min_value=1.0, max_value=10.0, value=1.0, step=0.1,
                                   disabled=(use_deff == "No"), key="deff_input")

use_hvif = st.sidebar.radio("Apply HVIF?", ["No", "Yes"], horizontal=True, key="use_hvif")
hvif_val = st.sidebar.number_input("HVIF", min_value=1.0, max_value=5.0, value=1.0, step=0.1,
                                   disabled=(use_hvif == "No"), key="hvif_input")

use_nr = st.sidebar.radio("Apply Nonresponse adjustment?", ["No", "Yes"], horizontal=True, key="use_nr")
nr_val = st.sidebar.number_input("Nonresponse rate (r)", min_value=0.0, max_value=0.90, value=0.05, step=0.01,
                                 disabled=(use_nr == "No"), key="nr_input")

DEFF = float(deff_val) if use_deff == "Yes" else 1.0
HVIF = float(hvif_val) if use_hvif == "Yes" else 1.0
r = float(nr_val) if use_nr == "Yes" else 0.0

# ============================================================
# Core calculations
# ============================================================
n_precision = adam_n_precision(N=N, epsilon=epsilon) if use_precision else None

# Power: benchmark (always computed), analytical (optional check)
n_power_benchmark = None
n_power_analytic = None
power_note = "Not applied"

if use_power:
    per_group = int(POWER_BENCHMARKS_PER_GROUP[effect_size])

    if design_type == "Single group (one sample)":
        n_power_benchmark = per_group
    elif design_type == "Two independent groups":
        n_power_benchmark = per_group * 2
    else:
        n_power_benchmark = per_group * groups_k

    power_note = f"Benchmark: {effect_size} = {per_group} per group (scaled by groups)."

    if show_analytic:
        # Analytical check is shown, but only drives n_power if user overrides
        n_power_analytic = n_two_proportions_total(alpha=alpha, power=target_power, p1=p1, p2=p2)

# Decide which n_power is used in max-rule
if not use_power:
    n_power = None
else:
    if power_mode == "Analytical drives (override)" and n_power_analytic is not None:
        n_power = n_power_analytic
        power_note += f" Using analytical override (α={alpha:.3f}, power={target_power:.2f}, p1={p1:.2f}, p2={p2:.2f})."
    else:
        n_power = n_power_benchmark
        power_note += f" α={alpha:.3f}, power={target_power:.2f} are not used to change n in benchmark mode."

# Model-based
n_model = None
model_note = "Not applied"
if use_model:
    if model_context == "Multiple regression":
        n_model = green_regression_min(k=k_predictors, which="individual")
        model_note = f"Green (1991): n ≥ 104 + k, k={k_predictors}."
    elif model_context == "Logistic regression":
        n_model = logistic_epv_min(k=k_predictors, event_rate=event_rate, epv=epv)
        model_note = f"EPV planning: n ≥ (EPV×k)/event_rate with EPV={epv}, k={k_predictors}, event rate={event_rate:g}."
    elif model_context == "SEM / CFA":
        p = approx_cfa_free_params(latents=latents, indicators_per_latent=indicators_per_latent)
        n_model = sem_min_n_by_ratio(params=p, ratio=sem_ratio)
        model_note = f"SEM planning: params≈{p}, ratio={sem_ratio}:1 ⇒ n≈{n_model}."
    else:
        model_note = "Model type not selected."

# Max-rule
candidates = []
if n_precision is not None: candidates.append(n_precision)
if n_power is not None: candidates.append(n_power)
if n_model is not None: candidates.append(n_model)
n_star = int(max(candidates)) if candidates else None

binding = "—"
if n_star is not None:
    tied = []
    if n_precision is not None and n_precision == n_star: tied.append("Precision")
    if n_power is not None and n_power == n_star: tied.append("Power")
    if n_model is not None and n_model == n_star: tied.append("Model")
    binding = ", ".join(tied)

inflator = (DEFF * HVIF) / max(1e-9, (1 - r))
n_inflated_raw = int(math.ceil(n_star * inflator)) if n_star is not None else None
exceeds_population = (n_inflated_raw is not None) and (n_inflated_raw > N)
n_inflated = N if exceeds_population else n_inflated_raw

# ============================================================
# Main UI
# ============================================================
st.title("US²DF Sample Size Planner")

c1, c2, c3 = st.columns(3)
c1.metric("Sample Size Estimate based on Precision", f"{n_precision:,}" if n_precision is not None else "—")
c2.metric("Sample Size Estimate based on Power", f"{n_power:,}" if n_power is not None else "—")
c3.metric("Sample Size Estimate based on Model", f"{n_model:,}" if n_model is not None else "—")

st.markdown("## US²DF Recommendation")
colA, colB = st.columns(2)
colA.metric("Base sample size (max-rule), n*", f"{n_star:,}" if n_star is not None else "—")
colB.metric("Adjusted and Final Recommended Sample Size", f"{n_inflated:,}" if n_inflated is not None else "—")

if n_star is not None:
    st.success(f"Binding constraint: **{binding}**")

if exceeds_population:
    st.warning(
        f"**Caution:** The adjusted sample size ({n_inflated_raw:,}) exceeds the population (N={N:,}). "
        f"US²DF recommends a **census/near-census** approach where feasible."
    )

# Show analytical as a check (so user sees it change)
if use_power and show_analytic and n_power_analytic is not None:
    st.markdown("### Power analytical check (diagnostic)")
    st.info(
        f"Analytical two-proportion check (balanced groups): total n ≈ {n_power_analytic:,} "
        f"for α={alpha:.3f}, power={target_power:.2f}, p1={p1:.2f}, p2={p2:.2f}. "
        f"{'Complementary applied (p2=1−p1).' if complementary_mode else ''}"
    )
    if n_power_benchmark is not None:
        ratio = n_power_analytic / max(1, n_power_benchmark)
        if ratio < 0.5:
            st.warning("Analytical check is far smaller than the benchmark. Treat it as optimistic unless you have strong prior evidence.")
        if ratio > 2.0:
            st.warning("Analytical check is far larger than the benchmark. That usually means the assumed effect is tiny.")

# Breakdown table
rows = [
    {"Component": "Precision", "Value": n_precision if n_precision is not None else "—",
     "Notes": f"Adam (2020): ε=ρe/t with ρ={rho:g}, e={e:g}, z={t:.4f}; n=N/(1+Nε²)" if use_precision else "Not applied"},
    {"Component": "Power", "Value": n_power if n_power is not None else "—", "Notes": power_note},
    {"Component": "Model", "Value": n_model if n_model is not None else "—", "Notes": model_note},
    {"Component": "Base sample size (n*)", "Value": n_star if n_star is not None else "—", "Notes": "Max-rule across selected components"},
    {"Component": "Inflation factors", "Value": f"DEFF={DEFF:g}, HVIF={HVIF:g}, r={r:g}", "Notes": "n_inflated = n* × DEFF × HVIF × 1/(1−r)"},
    {"Component": "Final recommended n", "Value": n_inflated if n_inflated is not None else "—", "Notes": "Capped at N if needed"},
]
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)

st.markdown("## Downloads")
st.download_button(
    "Download breakdown table (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="US2DF_Breakdown.csv",
    mime="text/csv",
)
