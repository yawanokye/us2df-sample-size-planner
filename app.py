# app.py
import math
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# US²DF Sample Size Planner
# - Max-rule: n* = max(n_precision, n_power, n_model)
# - Inflation: n_inflated = n* × DEFF × HVIF × 1/(1-r)
# - Caps n_inflated at N (population) with caution message
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
# Power defaults (US²DF) — PER GROUP
# ----------------------------
POWER_BENCHMARKS_PER_GROUP = {
    "Small": 400,
    "Medium": 100,
    "Large": 50
}


# ----------------------------
# Model-based heuristics
# ----------------------------
def green_regression_min(k: int, which: str = "individual") -> int:
    """
    Green (1991) rules of thumb:
    - Overall model test: n >= 50 + 8k
    - Individual predictors: n >= 104 + k
    """
    if which == "overall":
        return int(50 + 8 * k)
    return int(104 + k)


def logistic_epv_min(k: int, event_rate: float, epv: int = 10) -> int:
    """
    EPV planning heuristic: required events >= EPV * k
    total n >= (EPV * k) / event_rate
    """
    event_rate = max(1e-9, float(event_rate))
    return int(math.ceil((epv * k) / event_rate))


def approx_cfa_free_params(latents: int, indicators_per_latent: int) -> int:
    """
    Simple planning approximation for CFA/SEM free parameters (planning only).
    """
    L = int(latents)
    m = int(indicators_per_latent)
    total_ind = L * m

    loadings = (m - 1) * L          # 1 loading fixed per latent
    errors = total_ind              # indicator error variances
    latent_var_cov = L + (L * (L - 1)) // 2

    p = loadings + errors + latent_var_cov
    return int(p)


def sem_min_n_by_ratio(params: int, ratio: int = 10) -> int:
    return int(params * ratio)


# ============================================================
# Sidebar Inputs
# ============================================================
st.sidebar.title("Inputs")

study_purpose = st.sidebar.radio(
    "Study purpose",
    [
        "Descriptive estimation only",
        "Hypothesis testing / inferential study",
        "Model-based analysis",
        "Combination of the above",
    ],
    index=1
)

outcome_type = st.sidebar.selectbox(
    "Measurement Scale of the Estimand",
    ["Categorical (proportions)", "Continuous (means, scales)"],
    index=1
)

N = int(st.sidebar.number_input("Population size (N)", min_value=1, value=50000, step=100))

conf_level = st.sidebar.radio("Confidence level", ["95%", "99%"], index=0)
conf_level_val = 0.95 if conf_level == "95%" else 0.99
t = z_value(conf_level_val)

# ----------------------------
# Precision settings (Adam, 2020)
# ----------------------------
st.sidebar.subheader("Precision settings (Adam, 2020)")

is_categorical = outcome_type.startswith("Categorical")

# Adam (2020) scale parameters
rho = 2.0 if is_categorical else 4.0
default_e = 0.05 if is_categorical else 0.03

# Separate widget keys preserve user edits per scale
e_key = "e_categorical" if is_categorical else "e_continuous"

e = st.sidebar.number_input(
    "Desired degree of accuracy (e)",
    min_value=0.001,
    max_value=0.20,
    value=default_e,
    step=0.001,
    key=e_key
)

epsilon = adam_epsilon(rho=rho, e=e, t=t)
st.sidebar.caption(
    "Recommended defaults: e = 0.05 (categorical), e = 0.03 (continuous)"
)


# ----------------------------
# Power settings (only if relevant)
# ----------------------------
power_in_play = study_purpose in ["Hypothesis testing / inferential study", "Combination of the above"]

effect_size = None
design_type = None
groups_g = 1

if power_in_play:
    st.sidebar.subheader("Power settings (inferential)")

    # Medium default (index=1)
    effect_size = st.sidebar.radio("Expected effect size", ["Small", "Medium", "Large"], index=1)

    design_type = st.sidebar.selectbox(
        "Inferential design",
        ["Two-group comparison (t-test)", "ANOVA (3+ groups)"],
        index=0
    )

    if design_type == "Two-group comparison (t-test)":
        groups_g = 2
        st.sidebar.caption("Groups (g): 2")
    else:
        groups_g = int(
            st.sidebar.number_input("Number of groups (g)", min_value=3, max_value=50, value=3, step=1)
        )

# ----------------------------
# Model settings (show model choice first, then settings)
# ----------------------------
st.sidebar.subheader("Model settings (if applicable)")

model_context = st.sidebar.selectbox(
    "Model type",
    ["None", "Multiple regression", "Logistic regression", "SEM / CFA"],
    index=0
)

# Defaults so variables exist
k_predictors = 10
event_rate = 0.20
epv = 10
latents = 3
indicators_per_latent = 4
sem_ratio = 10

if model_context == "Multiple regression":
    k_predictors = int(st.sidebar.number_input("Number of predictors (k)", min_value=1, value=10, step=1))

elif model_context == "Logistic regression":
    k_predictors = int(st.sidebar.number_input("Number of predictors (k)", min_value=1, value=10, step=1))
    event_rate = float(
        st.sidebar.number_input(
            "Event rate (for logistic), e.g., 0.20",
            min_value=0.01, max_value=0.99, value=0.20, step=0.01
        )
    )
    epv = int(st.sidebar.number_input("EPV (events per variable)", min_value=5, max_value=50, value=10, step=1))

elif model_context == "SEM / CFA":
    latents = int(st.sidebar.number_input("Latent variables (SEM/CFA)", min_value=1, value=3, step=1))
    indicators_per_latent = int(st.sidebar.number_input("Indicators per latent", min_value=2, value=4, step=1))
    sem_ratio = int(st.sidebar.number_input("n per parameter ratio", min_value=5, max_value=30, value=10, step=1))

# ----------------------------
# Field adjustments with Yes/No enable switches
# ----------------------------
st.sidebar.subheader("Field adjustments")

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
    min_value=0.0, max_value=0.90, value=0.05, step=0.05,
    disabled=(use_nr == "No"), key="nr_input"
)

DEFF = float(deff_val) if use_deff == "Yes" else 1.0
HVIF = float(hvif_val) if use_hvif == "Yes" else 1.0
r = float(nr_val) if use_nr == "Yes" else 0.0

# ============================================================
# Core calculations
# ============================================================
# Precision
n_precision = adam_n_precision(N=N, epsilon=epsilon)

# Power (PER GROUP benchmark -> TOTAL depends on number of groups)
n_power = None
n_power_per_group = None
if power_in_play:
    n_power_per_group = int(POWER_BENCHMARKS_PER_GROUP[effect_size])
    n_power = int(n_power_per_group * groups_g)

# Model-based
model_in_play = study_purpose in ["Model-based analysis", "Combination of the above"]
n_model = None

if model_in_play:
    if model_context == "Multiple regression":
        n_model = green_regression_min(k=k_predictors, which="individual")
    elif model_context == "Logistic regression":
        n_model = logistic_epv_min(k=k_predictors, event_rate=event_rate, epv=epv)
    elif model_context == "SEM / CFA":
        p = approx_cfa_free_params(latents=latents, indicators_per_latent=indicators_per_latent)
        n_model = sem_min_n_by_ratio(params=p, ratio=sem_ratio)
    else:
        n_model = None

# Max-rule
candidates = {"Precision": n_precision}
if n_power is not None:
    candidates["Power"] = n_power
if n_model is not None:
    candidates["Model"] = n_model

n_star = int(max(candidates.values()))

# Binding constraint(s) (handles ties)
binding_constraints = [k for k, v in candidates.items() if v == n_star]
binding_text = ", ".join(binding_constraints)

# Inflation
inflator = (DEFF * HVIF) / max(1e-9, (1 - r))
n_inflated_raw = int(math.ceil(n_star * inflator))

exceeds_population = n_inflated_raw > N
n_inflated = N if exceeds_population else n_inflated_raw

# ============================================================
# Main UI
# ============================================================
st.title("Unified Sample Size Determination Framework (US²DF) Sample Size Planner")
st.write(
    "Computes sample size using the max-rule: n* = max(n_precision, n_power, n_model), "
    "then inflates for DEFF, HVIF, and nonresponse."
)

c1, c2, c3 = st.columns(3)
c1.metric("Sample Size Estimate based on Precision", f"{n_precision:,}")

if n_power is None:
    c2.metric("Sample Size Estimate based on Power", "—")
else:
    c2.metric("Sample Size Estimate based on Power", f"{n_power:,}")

c3.metric("Sample Size Estimate based on Model", f"{n_model:,}" if n_model is not None else "—")

st.markdown("## US²DF Recommendation")
colA, colB = st.columns(2)
colA.metric("Base sample size (max-rule), n*", f"{n_star:,}")
colB.metric("Adjusted and Final Recommended Sample Size", f"{n_inflated:,}")

st.success(f"Binding constraint(s): **{binding_text}**")

if exceeds_population:
    st.warning(
        f"**Caution:** The adjusted sample size ({n_inflated_raw:,}) exceeds the population (N={N:,}). "
        f"US²DF recommends a **census/near-census** approach where feasible. "
        f"If a census is not feasible, revise inflation drivers (DEFF/HVIF/nonresponse assumptions), "
        f"or revise precision/power/model targets, and report this limitation clearly."
    )

# Breakdown table
rows = []

rows.append({
    "Component": "Sample Size Estimate based on Precision",
    "Value": n_precision,
    "Notes": f"Adam (2020): ε=ρe/t with ρ={rho:g}, e={e:g}, z={t:.4f}; n=N/(1+Nε²)"
})

if n_power is None:
    rows.append({
        "Component": "Sample Size Estimate based on Power",
        "Value": "—",
        "Notes": "Not applied (study purpose does not include inferential power)."
    })
else:
    rows.append({
        "Component": "Sample Size Estimate based on Power",
        "Value": n_power,
        "Notes": (
            f"US²DF per-group benchmark: {effect_size}={n_power_per_group} per group "
            f"(α=0.05, 80% power). Design: {design_type}. Groups g={groups_g} → total n={n_power}."
        )
    })

rows.append({
    "Component": "Sample Size Estimate based on Model",
    "Value": n_model if n_model is not None else "—",
    "Notes": (
        "Regression (Green, 1991), Logistic (EPV), or SEM/CFA planning ratio, depending on model selection."
        if n_model is not None else "Not applied for this study purpose/model choice."
    )
})

rows.append({
    "Component": "Base sample size (max-rule), n*",
    "Value": n_star,
    "Notes": "n* = max(n_precision, n_power, n_model)"
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
    "Component": "Final n_inflated",
    "Value": n_inflated,
    "Notes": "n_inflated = n* × DEFF × HVIF × 1/(1−r) (capped at N if needed)"
})

df_breakdown = pd.DataFrame(rows)
st.dataframe(df_breakdown, use_container_width=True)

# Copy-ready methods text
st.subheader("Copy-ready Methods text")

power_clause = ""
if n_power is not None:
    power_clause = (
        f"For power, US²DF applies per-group benchmarks of {n_power_per_group:,} "
        f"({effect_size.lower()} effects) with g={groups_g} groups, giving n_power={n_power:,}. "
    )

model_clause = ""
if n_model is not None:
    model_clause = f"Model-based requirements were also assessed (n_model={n_model:,}). "

methods_text = (
    f"The base sample size was determined using the US²DF max-rule "
    f"(n* = max(n_precision, n_power, n_model)). "
    f"{power_clause}"
    f"{model_clause}"
    f"This value was adjusted for field conditions with DEFF={DEFF:g}, HVIF={HVIF:g}, and r={r:g}, "
    f"resulting in a final recommended sample size of n={n_inflated:,} "
    f"(Adam, Gyasi, Owusu Jnr & Gyamfi, 2026)."
)

st.code(methods_text, language="text")

# Required reference line under the methods text
st.markdown("---")
st.markdown(
    """
**Cite as:**

> **Adam, A.M., Gyasi, R.M., Owusu Jnr, P. & Gyamfi, E.N. (2026).**  
*Unified Sample Size Determination Framework (US²DF): Integrating Precision, Power, and Model-Based Requirements for Survey Studies.*
""",
    unsafe_allow_html=False
)

# Downloads
st.markdown("## Downloads")
st.download_button(
    "Download breakdown table (CSV)",
    data=df_breakdown.to_csv(index=False).encode("utf-8"),
    file_name="US2DF_Breakdown.csv",
    mime="text/csv",
)

# Note: Two-Layer Decision Table has been removed as requested.
#Update UI toggles and reporting labels
#Update UI toggles and reporting labels
#Update UI toggles and reporting labels








