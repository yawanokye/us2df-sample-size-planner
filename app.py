# app.py
import math
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# US²DF Sample Size Planner
# - Components selected via checkboxes (Precision, Power, Model)
# - Max-rule: n* = max(selected components)
# - Inflation: n_inflated = n* × DEFF × HVIF × 1/(1-r)
# - Caps n_inflated at N (population) with caution message
# - Power uses US²DF benchmarks (per group): 400, 100, 50 for Small/Medium/Large
# ============================================================

st.set_page_config(page_title="US²DF Sample Size Planner", layout="wide")

# ----------------------------
# Utilities (no SciPy needed)
# ----------------------------
def _norm_ppf(p: float) -> float:
    """Approx inverse CDF of standard normal (Acklam approximation)."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")

    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]

    plow = 0.02425
    phigh = 1 - plow

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


def z_value(conf_level: float) -> float:
    alpha = 1 - conf_level
    return _norm_ppf(1 - alpha / 2)


# ----------------------------
# Adam (2020) precision logic
# ----------------------------
def adam_epsilon(rho: float, e: float, t: float) -> float:
    # ε = ρe/t
    return (rho * e) / t


def adam_n_precision(N: int, epsilon: float) -> int:
    # n = N / (1 + N*ε^2)
    n = N / (1 + N * (epsilon ** 2))
    return int(math.ceil(n))


# ----------------------------
# US²DF Power Benchmarks (PER GROUP)
# ----------------------------
POWER_BENCHMARKS_PER_GROUP = {"Small": 400, "Medium": 100, "Large": 50}


# ----------------------------
# Model-based heuristics
# ----------------------------
def green_regression_min(k: int, which: str = "individual") -> int:
    """
    Green (1991) rules of thumb:
    - Overall model test: n >= 50 + 8k
    - Individual predictors: n >= 104 + k
    """
    k = max(1, int(k))
    if which == "overall":
        return int(50 + 8 * k)
    return int(104 + k)


def logistic_epv_min(k: int, event_rate: float, epv: int = 10) -> int:
    """
    EPV planning heuristic: required events >= EPV * k
    total n >= (EPV * k) / event_rate
    """
    k = max(1, int(k))
    epv = max(5, int(epv))
    event_rate = max(1e-9, min(0.999999999, float(event_rate)))
    return int(math.ceil((epv * k) / event_rate))


def approx_cfa_free_params(latents: int, indicators_per_latent: int) -> int:
    """
    Simple planning approximation for CFA/SEM free parameters (planning only).
    """
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
# Sidebar Inputs (Step-by-step)
# ============================================================
st.sidebar.title("Inputs (Step-by-step)")

st.sidebar.markdown("### Step 1, Select applicable components")
st.sidebar.caption("Choose one, two, or all three. US²DF applies the max-rule on your selections.")
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
st.sidebar.caption("Default is Categorical (proportions). Change only if your estimand is continuous.")

conf_level = st.sidebar.radio("Confidence level", ["95%", "99%"], index=0)
conf_level_val = 0.95 if conf_level == "95%" else 0.99
t = z_value(conf_level_val)

# ============================================================
# Precision (Adam, 2020)
# ============================================================
st.sidebar.markdown("### Step 3, Precision settings (Adam, 2020)")
st.sidebar.caption("Only applies if you selected Precision.")

is_categorical = outcome_type.startswith("Categorical")
rho = 2.0 if is_categorical else 4.0
default_e = 0.05 if is_categorical else 0.03
e_key = "e_categorical" if is_categorical else "e_continuous"

e = float(
    st.sidebar.number_input(
        "Desired degree of accuracy (e)",
        min_value=0.001, max_value=0.20,
        value=float(default_e),
        step=0.001,
        disabled=not use_precision,
        key=e_key,
        help="Defaults: 0.05 (categorical), 0.03 (continuous)."
    )
)

epsilon = adam_epsilon(rho=rho, e=e, t=t) if use_precision else None

# ============================================================
# Power (US²DF benchmarks)
# ============================================================
st.sidebar.markdown("### Step 4, Power settings")
st.sidebar.caption("Only applies if you selected Power. Benchmarks are PER GROUP.")

alpha = float(
    st.sidebar.number_input(
        "Significance level (α)",
        min_value=0.001, max_value=0.20,
        value=0.05, step=0.001,
        disabled=not use_power
    )
)
target_power = float(
    st.sidebar.number_input(
        "Target power (1−β)",
        min_value=0.50, max_value=0.99,
        value=0.80, step=0.01,
        disabled=not use_power
    )
)

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
    groups_k = int(
        st.sidebar.number_input(
            "Number of groups (k)",
            min_value=2,
            value=2 if design_type == "Two independent groups" else 3,
            step=1
        )
    )

st.sidebar.caption("Medium is the default effect size. Default proportions are not required in benchmark mode.")

# ============================================================
# Model
# ============================================================
st.sidebar.markdown("### Step 5, Model settings")
st.sidebar.caption("Only applies if you selected Model.")

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

# ============================================================
# Step 6: Field adjustments
# ============================================================
st.sidebar.markdown("### Step 6, Field adjustments")
st.sidebar.caption("Select Yes to enable each adjustment.")

use_deff = st.sidebar.radio("Apply DEFF?", ["No", "Yes"], horizontal=True, key="use_deff")
deff_val = st.sidebar.number_input(
    "DEFF",
    min_value=1.0, max_value=10.0, value=1.0, step=0.1,
    disabled=(use_deff == "No"), key="deff_input"
)

use_hvif = st.sidebar.radio("Apply HVIF?", ["No", "Yes"], horizontal=True, key="use_hvif")
hvif_val = st.sidebar.number_input(
    "HVIF",
    min_value=1.0, max_value=5.0, value=1.0, step=0.1,
    disabled=(use_hvif == "No"), key="hvif_input"
)

use_nr = st.sidebar.radio("Apply Nonresponse adjustment?", ["No", "Yes"], horizontal=True, key="use_nr")
nr_val = st.sidebar.number_input(
    "Nonresponse rate (r)",
    min_value=0.0, max_value=0.90, value=0.05, step=0.01,
    disabled=(use_nr == "No"), key="nr_input"
)

DEFF = float(deff_val) if use_deff == "Yes" else 1.0
HVIF = float(hvif_val) if use_hvif == "Yes" else 1.0
r = float(nr_val) if use_nr == "Yes" else 0.0

# ============================================================
# Core calculations
# ============================================================
n_precision = adam_n_precision(N=N, epsilon=epsilon) if use_precision else None

# Power: benchmarks are PER GROUP, then scaled by number of groups
n_power_per_group = None
n_power = None
power_note = "Not applied"

if use_power:
    n_power_per_group = int(POWER_BENCHMARKS_PER_GROUP[effect_size])

    if design_type == "Single group (one sample)":
        n_power = n_power_per_group
        power_note = (
            f"US²DF benchmark: {effect_size} = {n_power_per_group} (single group). "
            f"Defaults shown for reporting: α={alpha:.3f}, power={target_power:.2f}."
        )
    elif design_type == "Two independent groups":
        n_power = n_power_per_group * 2
        power_note = (
            f"US²DF benchmark: {effect_size} = {n_power_per_group} per group × 2 groups = {n_power} total. "
            f"Defaults shown for reporting: α={alpha:.3f}, power={target_power:.2f}."
        )
    else:
        n_power = n_power_per_group * groups_k
        power_note = (
            f"US²DF benchmark: {effect_size} = {n_power_per_group} per group × k={groups_k} groups = {n_power} total. "
            f"Defaults shown for reporting: α={alpha:.3f}, power={target_power:.2f}."
        )

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
        n_model = None
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

# Inflation
inflator = (DEFF * HVIF) / max(1e-9, (1 - r))
n_inflated_raw = int(math.ceil(n_star * inflator)) if n_star is not None else None
exceeds_population = (n_inflated_raw is not None) and (n_inflated_raw > N)
n_inflated = N if exceeds_population else n_inflated_raw

# ============================================================
# Main UI
# ============================================================
st.title("US²DF Sample Size Planner")

st.write(
    "Select the applicable components (Precision, Power, Model). "
    "US²DF applies the max-rule n* = max(selected components), then adjusts for DEFF, HVIF, and nonresponse."
)

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
        f"US²DF recommends a **census/near-census** approach where feasible. "
        f"If a census is not feasible, revise inflation drivers (DEFF/HVIF/nonresponse) or revise "
        f"precision/power/model targets and report this limitation clearly."
    )

# ============================================================
# Breakdown table
# ============================================================
rows = []
rows.append({
    "Component": "Sample Size Estimate based on Precision",
    "Value": n_precision if n_precision is not None else "—",
    "Notes": f"Adam (2020): ε=ρe/t with ρ={rho:g}, e={e:g}, z={t:.4f}; n=N/(1+Nε²)" if use_precision else "Not applied"
})
rows.append({
    "Component": "Sample Size Estimate based on Power",
    "Value": n_power if n_power is not None else "—",
    "Notes": power_note
})
rows.append({
    "Component": "Sample Size Estimate based on Model",
    "Value": n_model if n_model is not None else "—",
    "Notes": model_note
})
rows.append({
    "Component": "Base sample size (max-rule), n*",
    "Value": n_star if n_star is not None else "—",
    "Notes": "n* = max(selected components)"
})
rows.append({
    "Component": "DEFF applied?",
    "Value": "Yes" if use_deff == "Yes" else "No",
    "Notes": f"DEFF = {DEFF:g}"
})
rows.append({
    "Component": "HVIF applied?",
    "Value": "Yes" if use_hvif == "Yes" else "No",
    "Notes": f"HVIF = {HVIF:g}"
})
rows.append({
    "Component": "Nonresponse adjustment applied?",
    "Value": "Yes" if use_nr == "Yes" else "No",
    "Notes": f"r = {r:g}"
})
rows.append({
    "Component": "Final Recommended Sample Size",
    "Value": n_inflated if n_inflated is not None else "—",
    "Notes": "Adjusted Sample Size = n* × DEFF × HVIF × 1/(1−r) (capped at N if needed)"
})

df_breakdown = pd.DataFrame(rows)
st.dataframe(df_breakdown, use_container_width=True)

# ============================================================
# Copy-ready Methods text
# ============================================================
st.subheader("Copy-ready Methods text")

selected_parts = []
if use_precision: selected_parts.append("precision-based estimation")
if use_power: selected_parts.append("power-based requirements")
if use_model: selected_parts.append("model-based constraints")
selected_text = ", ".join(selected_parts) if selected_parts else "the selected components"

methods_text = (
    f"Sample size was determined using the US²DF framework by combining {selected_text} "
    f"and applying the max-rule (n* = max(n_precision, n_power, n_model) over the selected components). "
    f"For power planning, US²DF uses per-group benchmarks of 400, 100, and 50 for small, medium, and large effects, "
    f"and scales total required sample size by the number of groups (k). "
    f"The base sample was then adjusted for field conditions using DEFF={DEFF:g}, HVIF={HVIF:g}, "
    f"and an anticipated nonresponse rate r={r:g}, yielding a final recommended sample size of n={n_inflated:,}."
)

st.code(methods_text, language="text")

# ============================================================
# Cite as + Reference
# ============================================================
st.markdown("---")
st.markdown(
    """
**Cite as:**

> **Adam, A.M., Gyasi, R.M., Owusu Junior, P. & Gyamfi, E.N. (2026).**  
> *Unified Sample Size Determination Framework (US²DF): Integrating Precision, Power, and Model-Based Requirements for Survey Research.*

**Reference (as listed in the paper):**  
Adam, A.M., Gyasi, R.M., Owusu Junior, P., & Gyamfi, E.N. (2026). Unified Sample Size Determination Framework (US²DF): Integrating Precision, Power, and Model-Based Requirements for Survey Research.
""",
    unsafe_allow_html=False
)

# ============================================================
# Downloads
# ============================================================
st.markdown("## Downloads")
st.download_button(
    "Download breakdown table (CSV)",
    data=df_breakdown.to_csv(index=False).encode("utf-8"),
    file_name="US2DF_Breakdown.csv",
    mime="text/csv",
)
