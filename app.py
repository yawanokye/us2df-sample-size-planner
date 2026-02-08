# app.py
import math
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
# Lightweight styling
# ----------------------------
st.markdown(
    """
<style>
/* App-wide polish */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }

/* Sidebar spacing */
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

/* Card look for sidebar groups */
.us2df-card {
  border: 1px solid rgba(49, 51, 63, 0.14);
  border-radius: 14px;
  padding: 12px 12px 8px 12px;
  margin-bottom: 10px;
  background: rgba(255,255,255,0.55);
}
.us2df-card-title {
  font-weight: 700;
  font-size: 0.95rem;
  margin-bottom: 4px;
}
.us2df-card-sub {
  color: rgba(49, 51, 63, 0.72);
  font-size: 0.82rem;
  margin-bottom: 6px;
}

/* Popover content styling */
.us2df-pop p { margin: 0.35rem 0; line-height: 1.35; }
.us2df-pop .tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 700;
  margin-bottom: 6px;
}
.us2df-pop .tag.prec { background: rgba(33,150,243,0.14); color: rgb(24, 90, 150); }
.us2df-pop .tag.pow  { background: rgba(76,175,80,0.14);  color: rgb(40, 110, 55); }
.us2df-pop .tag.mod  { background: rgba(255,152,0,0.14);  color: rgb(150, 95, 10); }

/* Metric row */
div[data-testid="stMetric"] {
  border: 1px solid rgba(49, 51, 63, 0.12);
  border-radius: 14px;
  padding: 10px;
  background: rgba(255,255,255,0.55);
}

/* Dataframe container */
div[data-testid="stDataFrame"] {
  border: 1px solid rgba(49, 51, 63, 0.12);
  border-radius: 14px;
  padding: 6px;
  background: rgba(255,255,255,0.55);
}
</style>
""",
    unsafe_allow_html=True
)

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
    return (rho * e) / t


def adam_n_precision(N: int, epsilon: float) -> int:
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
    k = max(1, int(k))
    return int(50 + 8 * k) if which == "overall" else int(104 + k)


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
# Sidebar layout (Step-by-step)
# ============================================================
st.sidebar.markdown("## US²DF Inputs")
st.sidebar.caption("Work top to bottom. The final recommendation is the maximum of selected components, then adjusted for field conditions.")

# ----------------------------
# Step 1: components
# ----------------------------
st.sidebar.markdown("### Step 1, Components")

st.sidebar.markdown(
    '<div class="us2df-card"><div class="us2df-card-title">Choose what applies</div>'
    '<div class="us2df-card-sub">You can select one, two, or three. US²DF uses the binding constraint (max-rule).</div></div>',
    unsafe_allow_html=True
)

r1 = st.sidebar.columns([0.78, 0.22])
use_precision = r1[0].checkbox("Precision component", value=True)
with r1[1].popover("ℹ", use_container_width=True):
    st.markdown(
        """
<div class="us2df-pop">
<span class="tag prec">PRECISION</span>
<p><span style="color:rgb(24, 90, 150); font-weight:700;">What it represents</span><br>
A margin-of-error requirement for estimating a mean or proportion in a finite population.</p>
<p><span style="color:rgb(24, 90, 150); font-weight:700;">What it uses</span><br>
Adam (2020): ε=ρe/t and n=N/(1+Nε²).</p>
<p><span style="color:rgb(24, 90, 150); font-weight:700;">When to select</span><br>
Descriptive surveys or when you must report a confidence bound on an estimate.</p>
</div>
""",
        unsafe_allow_html=True
    )

r2 = st.sidebar.columns([0.78, 0.22])
use_power = r2[0].checkbox("Power component", value=True)
with r2[1].popover("ℹ", use_container_width=True):
    st.markdown(
        """
<div class="us2df-pop">
<span class="tag pow">POWER</span>
<p><span style="color:rgb(40, 110, 55); font-weight:700;">What it represents</span><br>
Minimum sample size needed to detect effects in hypothesis testing.</p>
<p><span style="color:rgb(40, 110, 55); font-weight:700;">What it uses</span><br>
US²DF empirical benchmarks per group: Small=400, Medium=100, Large=50. Total scales by number of groups.</p>
<p><span style="color:rgb(40, 110, 55); font-weight:700;">When to select</span><br>
Group comparisons, associations, treatment contrasts, inferential objectives.</p>
</div>
""",
        unsafe_allow_html=True
    )

r3 = st.sidebar.columns([0.78, 0.22])
use_model = r3[0].checkbox("Model component", value=False)
with r3[1].popover("ℹ", use_container_width=True):
    st.markdown(
        """
<div class="us2df-pop">
<span class="tag mod">MODEL</span>
<p><span style="color:rgb(150, 95, 10); font-weight:700;">What it represents</span><br>
Minimum sample size to support the planned model complexity.</p>
<p><span style="color:rgb(150, 95, 10); font-weight:700;">What it uses</span><br>
Green (1991) for regression, EPV for logistic regression, and an SEM/CFA planning ratio.</p>
<p><span style="color:rgb(150, 95, 10); font-weight:700;">When to select</span><br>
If your main results depend on a regression/SEM/logistic model with many parameters.</p>
</div>
""",
        unsafe_allow_html=True
    )

if not (use_precision or use_power or use_model):
    st.sidebar.error("Select at least one component to proceed.")

# ----------------------------
# Step 2: population + measurement
# ----------------------------
st.sidebar.markdown("### Step 2, Population and measurement")
N = int(st.sidebar.number_input("Population size (N)", min_value=1, value=50000, step=100))

outcome_type = st.sidebar.selectbox(
    "Measurement scale of the estimand",
    ["Categorical (proportions)", "Continuous (means, scales)"],
    index=0
)
st.sidebar.caption("Default is categorical.")

conf_level = st.sidebar.radio("Confidence level", ["95%", "99%"], index=0)
conf_level_val = 0.95 if conf_level == "95%" else 0.99
t = z_value(conf_level_val)

# ----------------------------
# Step 3: precision settings
# ----------------------------
st.sidebar.markdown("### Step 3, Precision settings")
st.sidebar.caption("Used only if Precision is selected.")

is_categorical = outcome_type.startswith("Categorical")
rho = 2.0 if is_categorical else 4.0
default_e = 0.05 if is_categorical else 0.03
e_key = "e_categorical" if is_categorical else "e_continuous"

e = float(
    st.sidebar.number_input(
        "Margin of error / accuracy target (e)",
        min_value=0.001, max_value=0.20,
        value=float(default_e),
        step=0.001,
        disabled=not use_precision,
    )
)
epsilon = adam_epsilon(rho=rho, e=e, t=t) if use_precision else None

# ----------------------------
# Step 4: power settings (benchmarks)
# ----------------------------
st.sidebar.markdown("### Step 4, Power settings")
st.sidebar.caption("Used only if Power is selected. Benchmarks are per group.")

effect_size = st.sidebar.radio(
    "Expected effect size (benchmark)",
    ["Small", "Medium", "Large"],
    index=1,
    disabled=not use_power
)

design_type = st.sidebar.selectbox(
    "Design structure (for scaling groups)",
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

with st.sidebar.expander("Optional, reporting defaults (do not change benchmark n)", expanded=False):
    st.caption("These appear in the Methods text only. The benchmark n does not recompute from α and power here.")
    alpha = float(st.number_input("Significance level (α)", min_value=0.001, max_value=0.20, value=0.05, step=0.001))
    target_power = float(st.number_input("Target power (1−β)", min_value=0.50, max_value=0.99, value=0.80, step=0.01))

if "alpha" not in locals():
    alpha = 0.05
if "target_power" not in locals():
    target_power = 0.80

# ----------------------------
# Step 5: model settings
# ----------------------------
st.sidebar.markdown("### Step 5, Model settings")
st.sidebar.caption("Used only if Model is selected.")

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

# ----------------------------
# Step 6: field adjustments
# ----------------------------
st.sidebar.markdown("### Step 6, Field adjustments")

use_deff = st.sidebar.radio("Apply DEFF?", ["No", "Yes"], horizontal=True, key="use_deff")
deff_val = st.sidebar.number_input("DEFF", 1.0, 10.0, 1.0, 0.1, disabled=(use_deff == "No"), key="deff_input")

use_hvif = st.sidebar.radio("Apply HVIF?", ["No", "Yes"], horizontal=True, key="use_hvif")
hvif_val = st.sidebar.number_input("HVIF", 1.0, 5.0, 1.0, 0.1, disabled=(use_hvif == "No"), key="hvif_input")

use_nr = st.sidebar.radio("Apply Nonresponse adjustment?", ["No", "Yes"], horizontal=True, key="use_nr")
nr_val = st.sidebar.number_input("Nonresponse rate (r)", 0.0, 0.90, 0.05, 0.01, disabled=(use_nr == "No"), key="nr_input")

DEFF = float(deff_val) if use_deff == "Yes" else 1.0
HVIF = float(hvif_val) if use_hvif == "Yes" else 1.0
r = float(nr_val) if use_nr == "Yes" else 0.0

# ============================================================
# Core calculations
# ============================================================
n_precision = adam_n_precision(N=N, epsilon=epsilon) if use_precision else None

n_power_per_group = None
n_power = None
power_note = "Not applied"
if use_power:
    n_power_per_group = int(POWER_BENCHMARKS_PER_GROUP[effect_size])
    if design_type == "Single group (one sample)":
        n_power = n_power_per_group
        power_note = f"Benchmark per group used as total (single group). α={alpha:.3f}, power={target_power:.2f} for reporting."
    elif design_type == "Two independent groups":
        n_power = n_power_per_group * 2
        power_note = f"{n_power_per_group} per group × 2 = {n_power} total. α={alpha:.3f}, power={target_power:.2f} for reporting."
    else:
        n_power = n_power_per_group * groups_k
        power_note = f"{n_power_per_group} per group × k={groups_k} = {n_power} total. α={alpha:.3f}, power={target_power:.2f} for reporting."

n_model = None
model_note = "Not applied"
if use_model:
    if model_context == "Multiple regression":
        n_model = green_regression_min(k=k_predictors, which="individual")
        model_note = f"Green (1991): n ≥ 104 + k, k={k_predictors}."
    elif model_context == "Logistic regression":
        n_model = logistic_epv_min(k=k_predictors, event_rate=event_rate, epv=epv)
        model_note = f"EPV: n ≥ (EPV×k)/event_rate with EPV={epv}, k={k_predictors}, event rate={event_rate:g}."
    elif model_context == "SEM / CFA":
        p = approx_cfa_free_params(latents=latents, indicators_per_latent=indicators_per_latent)
        n_model = sem_min_n_by_ratio(params=p, ratio=sem_ratio)
        model_note = f"SEM planning: params≈{p}, ratio={sem_ratio}:1 ⇒ n≈{n_model}."
    else:
        model_note = "Model type not selected."

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
st.markdown("# US²DF Sample Size Planner")
st.caption("US²DF selects the binding constraint across chosen components, then adjusts for field conditions.")

m1, m2, m3 = st.columns(3)
m1.metric("Precision-based sample size", f"{n_precision:,}" if n_precision is not None else "—")
m2.metric("Power-based sample size", f"{n_power:,}" if n_power is not None else "—")
m3.metric("Model-based sample size", f"{n_model:,}" if n_model is not None else "—")

st.markdown("## US²DF Recommendation")
a, b = st.columns(2)
a.metric("Base sample size (max-rule), n*", f"{n_star:,}" if n_star is not None else "—")
b.metric("Adjusted and Final Recommended Sample Size", f"{n_inflated:,}" if n_inflated is not None else "—")

if n_star is not None:
    st.success(f"Binding constraint: **{binding}**")

if exceeds_population:
    st.warning(
        f"Adjusted sample size ({n_inflated_raw:,}) exceeds N={N:,}. "
        f"Consider census/near-census, or revisit inflation assumptions."
    )

st.markdown("## Breakdown")
rows = [
    {"Component": "Precision", "Value": n_precision if n_precision is not None else "—",
     "Notes": f"Adam (2020): ε=ρe/t with ρ={rho:g}, e={e:g}, z={t:.4f}; n=N/(1+Nε²)" if use_precision else "Not applied"},
    {"Component": "Power", "Value": n_power if n_power is not None else "—", "Notes": power_note},
    {"Component": "Model", "Value": n_model if n_model is not None else "—", "Notes": model_note},
    {"Component": "Base (n*)", "Value": n_star if n_star is not None else "—", "Notes": "Max-rule across selected components"},
    {"Component": "Field adjustments", "Value": f"DEFF={DEFF:g}, HVIF={HVIF:g}, r={r:g}", "Notes": "n_inflated = n* × DEFF × HVIF × 1/(1−r)"},
    {"Component": "Final n", "Value": n_inflated if n_inflated is not None else "—", "Notes": "Capped at N if needed"},
]
df_breakdown = pd.DataFrame(rows)
st.dataframe(df_breakdown, use_container_width=True)

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

st.markdown("---")
st.markdown(
    """
**Cite as:**

> **Adam, A.M., Gyasi, R.M., Owusu Junior, P. & Gyamfi, E.N. (2026).**  
> *Unified Sample Size Determination Framework (US²DF): Integrating Precision, Power, and Model-Based Requirements for Survey Research.*

**Reference:**  
Adam, A.M., Gyasi, R.M., Owusu Junior, P., & Gyamfi, E.N. (2026). Unified Sample Size Determination Framework (US²DF): Integrating Precision, Power, and Model-Based Requirements for Survey Research.
""",
    unsafe_allow_html=False
)

st.markdown("## Downloads")
st.download_button(
    "Download breakdown table (CSV)",
    data=df_breakdown.to_csv(index=False).encode("utf-8"),
    file_name="US2DF_Breakdown.csv",
    mime="text/csv",
)
