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
    """
    Approx inverse CDF of standard normal (Acklam approximation).
    Good enough for z-values (e.g., 0.975 -> 1.96, 0.995 -> 2.576).
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")

    # Coefficients in rational approximations
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
    # two-sided: z = Phi^{-1}(1 - alpha/2)
    alpha = 1 - conf_level
    return _norm_ppf(1 - alpha/2)


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
# Power defaults (US²DF)
# ----------------------------
POWER_BENCHMARKS = {
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
        return int(50 + 8*k)
    return int(104 + k)


def logistic_epv_min(k: int, event_rate: float, epv: int = 10) -> int:
    """
    EPV rule (planning heuristic): required events >= EPV * k
    total n >= (EPV * k) / event_rate
    """
    event_rate = max(1e-9, float(event_rate))
    return int(math.ceil((epv * k) / event_rate))


def approx_cfa_free_params(latents: int, indicators_per_latent: int) -> int:
    """
    Simple planning approximation for CFA/SEM free parameters.
    (Not exact, intended for planning grid only)
    """
    L = int(latents)
    m = int(indicators_per_latent)
    total_ind = L * m

    # Loadings: 1 fixed per latent for scale -> (m-1)*L free loadings
    loadings = (m - 1) * L

    # Indicator error variances: total_ind
    errors = total_ind

    # Latent variances + covariances: L variances + L(L-1)/2 covariances
    latent_var_cov = L + (L * (L - 1)) // 2

    # Latent means not included (typical CFA standardised)
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
    "Outcome type",
    ["Categorical (proportions)", "Continuous (means, scales)"],
    index=1
)

N = int(st.sidebar.number_input("Population size (N)", min_value=1, value=50000, step=100))

conf_level = st.sidebar.radio("Confidence level", ["95%", "99%"], index=0)
conf_level_val = 0.95 if conf_level == "95%" else 0.99
t = z_value(conf_level_val)

# Expected effect size (only relevant when power is in play)
effect_size = st.sidebar.radio("Expected effect size", ["Small", "Medium", "Large"], index=0)

# Precision settings (Adam 2020 uses rho)
st.sidebar.subheader("Precision settings (Adam, 2020)")
if outcome_type.startswith("Categorical"):
    rho = 2.0
    default_e = 0.05
else:
    rho = 4.0
    default_e = 0.03

e = float(st.sidebar.number_input("Desired degree of accuracy (e)", min_value=0.001, max_value=0.20, value=default_e, step=0.001))
epsilon = adam_epsilon(rho=rho, e=e, t=t)

# Model settings (if relevant)
st.sidebar.subheader("Model settings (if applicable)")

model_context = st.sidebar.selectbox(
    "Model type",
    ["None", "Multiple regression", "Logistic regression", "SEM / CFA"],
    index=0
)

# Defaults (so variables exist even when hidden)
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
    event_rate = float(st.sidebar.number_input("Event rate (for logistic), e.g., 0.20", min_value=0.01, max_value=0.99, value=0.20, step=0.01))
    epv = int(st.sidebar.number_input("EPV (events per variable)", min_value=5, max_value=50, value=10, step=1))

elif model_context == "SEM / CFA":
    latents = int(st.sidebar.number_input("Latent variables (SEM/CFA)", min_value=1, value=3, step=1))
    indicators_per_latent = int(st.sidebar.number_input("Indicators per latent", min_value=2, value=4, step=1))
    sem_ratio = int(st.sidebar.number_input("n per parameter ratio", min_value=5, max_value=30, value=10, step=1))

# Field adjustments with Yes/No enable switches
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

# Power (only for inferential or combination)
n_power = None
if study_purpose in ["Hypothesis testing / inferential study", "Combination of the above"]:
    n_power = int(POWER_BENCHMARKS[effect_size])

# Model-based
n_model = None
if study_purpose in ["Model-based analysis", "Combination of the above"]:
    if model_context == "Multiple regression":
        # default to individual predictor rule (stricter)
        n_model = green_regression_min(k=k_predictors, which="individual")
    elif model_context == "Logistic regression":
        n_model = logistic_epv_min(k=k_predictors, event_rate=event_rate, epv=epv)
    elif model_context == "SEM / CFA":
        p = approx_cfa_free_params(latents=latents, indicators_per_latent=indicators_per_latent)
        n_model = sem_min_n_by_ratio(params=p, ratio=sem_ratio)
    else:
        n_model = None

# Max-rule
candidates = [n_precision]
if n_power is not None:
    candidates.append(n_power)
if n_model is not None:
    candidates.append(n_model)

n_star = int(max(candidates))

# Binding constraint
binding = "Precision"
if (n_power is not None) and (n_star == n_power):
    binding = "Power"
if (n_model is not None) and (n_star == n_model):
    binding = "Model"

# Inflation
inflator = (DEFF * HVIF) / max(1e-9, (1 - r))
n_inflated_raw = int(math.ceil(n_star * inflator))

exceeds_population = n_inflated_raw > N
n_inflated = N if exceeds_population else n_inflated_raw

# ============================================================
# Main UI
# ============================================================
st.title("Universal Sample Size Determination Framework (US²DF) Sample Size Planner")
st.write(
    "Determines sample size using the max-rule: n* = max(n_precision, n_power, n_model), "
    "then inflates for DEFF, HVIF, and nonresponse."
)

c1, c2, c3 = st.columns(3)
c1.metric("Sample Size Estimate based on Precision", f"{n_precision:,}")
c2.metric("Sample Size Estimate based on Power", f"{n_power:,}" if n_power is not None else "—")
c3.metric("Sample Size Estimate based on Model", f"{n_model:,}" if n_model is not None else "—")

st.markdown("## US²DF Recommendation")
colA, colB = st.columns(2)
colA.metric("Base sample size (max-rule), n*", f"{n_star:,}")
colB.metric("**Adjusted and Final Recommended Sample Size**", f"{n_inflated:,}")

st.success(f"Binding constraint: **{binding}**")

if exceeds_population:
    st.warning(
        f"**Caution:** The Adjusted Sample Size ({n_inflated_raw:,}) exceeds the population (N={N:,}). "
        f"US²DF recommends a **census/near-census** approach where feasible. "
        f"If a census is not feasible, revise inflation drivers (DEFF/HVIF/nonresponse assumptions) "
        f"or revise precision/power/model targets, and report this limitation clearly."
    )

# Breakdown table
rows = []

rows.append({
    "Component": "Sample Size Estimate based on Precision",
    "Value": n_precision,
    "Notes": f"Adam (2020): ε=ρe/t with ρ={rho:g}, e={e:g}, z={t:.4f}; n=N/(1+Nε²)"
})
rows.append({
    "Component": "Sample Size Estimate based on Power",
    "Value": n_power if n_power is not None else "—",
    "Notes": f"US²DF benchmark: {effect_size}= {POWER_BENCHMARKS[effect_size]} (α=0.05, 80% power) if inferential"
})
rows.append({
    "Component": "Sample Size Estimate based on Model",
    "Value": n_model if n_model is not None else "—",
    "Notes": (
        "Regression (Green, 1991) / Logistic (EPV) / SEM planning ratio, depending on model selection"
        if n_model is not None else "Not applied for this study purpose/model choice"
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
methods_text = (
    f"We determined the minimum sample size using the Universal Sample Size Determination Framework (US²DF), "
    f"which applies a max-rule across precision-, power-, and model-based requirements: "
    f"n* = max(n_precision, n_power, n_model). Precision was computed using Adam’s (2020) adjusted Yamane approach, "
    f"where ε = ρe/t and n_precision = N/(1 + Nε²) with N={N:,}, confidence level={int(conf_level_val*100)}% (z={t:.3f}), "
    f"ρ={rho:g}, and e={e:g}, giving n_precision={n_precision:,}. "
    f"{'Power was planned using the US²DF benchmark for ' + effect_size.lower() + ' effects (n_power=' + str(n_power) + '). ' if n_power is not None else ''}"
    f"{'Model-based planning yielded n_model=' + str(n_model) + ' under the selected model specification. ' if n_model is not None else ''}"
    f"The base recommendation was n*={n_star:,} (binding constraint: {binding}). "
    f"We then inflated the target for field realities using n_inflated = n* × DEFF × HVIF × 1/(1−r), "
    f"with DEFF={DEFF:g}, HVIF={HVIF:g}, and r={r:g}, resulting in recommended sample size of {n_inflated:,} "(Adam, Gyasi, Owusu Jnr & Gyamfi, 2026)"."
    + (f" Because the inflated target exceeded the population (raw target={n_inflated_raw:,} > N={N:,}), "
       f"we recommend a census/near-census approach or revised assumptions." if exceeds_population else "")
)
st.code(methods_text, language="text")

# Required reference line under the methods text
st.markdown("---")

st.markdown(
    """
**Cite as:**

> **Adam, A.M.,  Gyasi, R.M., Owusu Jnr, P. & Gyamfi, E.N. (2026).**   *Universal Sample Size Determination Framework (US²DF):  A Unified Approach Integrating Precision, Power, and Model-Based  Requirements for Survey Studies.*
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
