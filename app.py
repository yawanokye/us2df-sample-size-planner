# app.py
import math
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# US²DF Sample Size Planner (Unified / Universal Framework)
# - User selects applicable components via checkboxes:
#     Precision, Power, Model
# - Max-rule: n* = max(selected components)
# - Inflation: n_inflated = n* × DEFF × HVIF × 1/(1-r)
# - Caps n_inflated at N with a caution recommendation
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


def z_value_two_sided(alpha: float) -> float:
    # Two-sided critical value: z_(1 - alpha/2)
    alpha = max(1e-12, min(0.999999999999, float(alpha)))
    return _norm_ppf(1 - alpha / 2)


def z_value_one_sided(alpha: float) -> float:
    # One-sided critical value: z_(1 - alpha)
    alpha = max(1e-12, min(0.999999999999, float(alpha)))
    return _norm_ppf(1 - alpha)


def z_value_power(power: float) -> float:
    # z_(power)
    power = max(1e-12, min(0.999999999999, float(power)))
    return _norm_ppf(power)


# ----------------------------
# Adam (2020) precision logic
# ----------------------------
def adam_epsilon(rho: float, e: float, z: float) -> float:
    # ε = ρe/z
    return (rho * e) / z


def adam_n_precision(N: int, epsilon: float) -> int:
    # n = N / (1 + N*ε^2)
    n = N / (1 + N * (epsilon ** 2))
    return int(math.ceil(n))


# ----------------------------
# Power calculations (anchored)
# ----------------------------
# Cohen conventional effect sizes
COHEN_D = {"Small": 0.20, "Medium": 0.50, "Large": 0.80}
COHEN_F = {"Small": 0.10, "Medium": 0.25, "Large": 0.40}  # One-way ANOVA

def n_per_group_two_sample(d: float, alpha: float, power: float) -> int:
    """
    Normal-approx planning formula for two independent groups (equal n):
    n_per_group ≈ 2 * (z_(1-α/2) + z_power)^2 / d^2
    """
    d = max(1e-9, float(d))
    z_alpha = z_value_two_sided(alpha)
    z_pow = z_value_power(power)
    n = 2.0 * (z_alpha + z_pow) ** 2 / (d ** 2)
    return int(math.ceil(n))


def n_per_group_oneway_anova(f: float, g: int, alpha: float, power: float) -> int:
    """
    Practical planning approximation for one-way ANOVA (equal group sizes).

    Exact ANOVA power requires a noncentral F. To keep the tool dependency-free,
    we use a conservative approximation by mapping ANOVA effect f to an
    "equivalent" pairwise difference roughly d ≈ 2f, then using the two-group
    planning formula for per-group size.

    This errs on the safer side when groups > 2.
    """
    g = max(2, int(g))
    f = max(1e-9, float(f))
    d_equiv = 2.0 * f
    return n_per_group_two_sample(d_equiv, alpha=alpha, power=power)


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
    EPV rule (planning heuristic): required events >= EPV * k
    total n >= (EPV * k) / event_rate
    """
    k = max(1, int(k))
    epv = max(5, int(epv))
    event_rate = max(1e-9, min(0.999999999, float(event_rate)))
    return int(math.ceil((epv * k) / event_rate))


def approx_cfa_free_params(latents: int, indicators_per_latent: int) -> int:
    """
    Simple planning approximation for CFA/SEM free parameters.
    (Not exact, intended for planning only)
    """
    L = max(1, int(latents))
    m = max(2, int(indicators_per_latent))
    total_ind = L * m

    loadings = (m - 1) * L
    errors = total_ind
    latent_var_cov = L + (L * (L - 1)) // 2

    p = loadings + errors + latent_var_cov
    return int(p)


def sem_min_n_by_ratio(params: int, ratio: int = 10) -> int:
    params = max(1, int(params))
    ratio = max(5, int(ratio))
    return int(params * ratio)


# ============================================================
# Sidebar Inputs (Step-by-step)
# ============================================================
st.sidebar.title("Inputs (Step-by-step)")

st.sidebar.markdown("### Step 1, Choose which components apply")
st.sidebar.caption(
    "Tick only what is applicable for your study. You can select one, two, or all three."
)
use_precision = st.sidebar.checkbox("Use Precision component", value=True)
use_power = st.sidebar.checkbox("Use Power component", value=True)
use_model = st.sidebar.checkbox("Use Model component", value=False)

if not (use_precision or use_power or use_model):
    st.sidebar.error("Select at least one component: Precision, Power, or Model.")

st.sidebar.markdown("### Step 2, Study set-up")
# Measurement Scale default should be Categorical
outcome_type = st.sidebar.selectbox(
    "Measurement Scale of the Estimand",
    ["Categorical (proportions)", "Continuous (means, scales)"],
    index=0  # default categorical
)
st.sidebar.caption("Default is **Categorical (proportions)**. Change it if your estimand is continuous.")

# Use number_input with steppers (+ / -) instead of sliders
N = int(
    st.sidebar.number_input(
        "Population size (N)",
        min_value=1,
        value=50000,
        step=100
    )
)

conf_level = st.sidebar.radio("Confidence level", ["95%", "99%"], index=0)
conf_level_val = 0.95 if conf_level == "95%" else 0.99
# Convert confidence level to alpha for precision z
alpha_precision = 1.0 - conf_level_val
z_precision = z_value_two_sided(alpha_precision)

# ----------------------------
# Precision (Step 3)
# ----------------------------
st.sidebar.markdown("### Step 3, Precision settings (Adam, 2020)")
st.sidebar.caption("Only needed if you selected the Precision component.")

# Default e depends on outcome type, but user can still adjust
if outcome_type.startswith("Categorical"):
    rho = 2.0
    default_e = 0.05
else:
    rho = 4.0
    default_e = 0.03

e = float(
    st.sidebar.number_input(
        "Desired degree of accuracy (e)",
        min_value=0.001,
        max_value=0.20,
        value=float(default_e),
        step=0.001,
        disabled=not use_precision,
        help="Default: 0.05 for categorical, 0.03 for continuous. You can change it."
    )
)

epsilon = adam_epsilon(rho=rho, e=e, z=z_precision) if use_precision else None

# ----------------------------
# Power (Step 4)
# ----------------------------
st.sidebar.markdown("### Step 4, Power settings")
st.sidebar.caption("Only needed if you selected the Power component.")

alpha = float(
    st.sidebar.number_input(
        "Significance level (α)",
        min_value=0.001,
        max_value=0.20,
        value=0.05,
        step=0.001,
        disabled=not use_power
    )
)

target_power = float(
    st.sidebar.number_input(
        "Target power (1−β)",
        min_value=0.50,
        max_value=0.99,
        value=0.80,
        step=0.01,
        disabled=not use_power
    )
)

effect_size_label = st.sidebar.radio(
    "Expected effect size (Cohen convention)",
    ["Small", "Medium", "Large"],
    index=1,  # Medium default
    disabled=not use_power
)

# Support 2+ groups (ANOVA-style)
design_type = st.sidebar.selectbox(
    "Inferential design (for power)",
    ["Single group / one sample", "Two independent groups", "One-way ANOVA (k groups)"],
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

st.sidebar.caption(
    "Tip: For **Two independent groups**, the app computes **n per group** then multiplies by k. "
    "For **ANOVA**, it uses a conservative approximation to keep the tool dependency-free."
)

# ----------------------------
# Model (Step 5)
# ----------------------------
st.sidebar.markdown("### Step 5, Model settings")
st.sidebar.caption("Only needed if you selected the Model component.")

model_context = st.sidebar.selectbox(
    "Model type",
    ["None", "Multiple regression", "Logistic regression", "SEM / CFA"],
    index=0,
    disabled=not use_model
)

# Defaults so vars exist
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
    event_rate = float(
        st.sidebar.number_input(
            "Event rate (for logistic), e.g., 0.20",
            min_value=0.01, max_value=0.99, value=0.20, step=0.01
        )
    )
    epv = int(st.sidebar.number_input("EPV (events per variable)", min_value=5, max_value=50, value=10, step=1))
elif use_model and model_context == "SEM / CFA":
    latents = int(st.sidebar.number_input("Latent variables (L)", min_value=1, value=3, step=1))
    indicators_per_latent = int(st.sidebar.number_input("Indicators per latent (m)", min_value=2, value=4, step=1))
    sem_ratio = int(st.sidebar.number_input("n per parameter ratio", min_value=5, max_value=30, value=10, step=1))

# ----------------------------
# Field adjustments (Step 6)
# ----------------------------
st.sidebar.markdown("### Step 6, Field adjustments")
st.sidebar.caption("Use Yes/No first. Only Yes enables the input.")

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
n_precision = None
if use_precision:
    n_precision = adam_n_precision(N=N, epsilon=epsilon)

n_power = None
power_note = "—"
if use_power:
    if design_type == "Single group / one sample":
        # Use two-group formula without the factor 2 (rough planning) by halving the two-group requirement.
        # Conservative: we still keep it close to two-group by not halving too aggressively.
        d = COHEN_D[effect_size_label]
        n_pg = n_per_group_two_sample(d=d, alpha=alpha, power=target_power)
        n_power = int(math.ceil(n_pg))  # treat as "total n"
        power_note = f"Computed from α={alpha:g}, power={target_power:g}, effect={effect_size_label} (single-group planning)."
    elif design_type == "Two independent groups":
        d = COHEN_D[effect_size_label]
        n_pg = n_per_group_two_sample(d=d, alpha=alpha, power=target_power)
        n_power = int(n_pg * groups_k)
        power_note = f"Two-group planning: n≈{n_pg} per group × k={groups_k} (α={alpha:g}, power={target_power:g})."
    else:  # One-way ANOVA
        f = COHEN_F[effect_size_label]
        n_pg = n_per_group_oneway_anova(f=f, g=groups_k, alpha=alpha, power=target_power)
        n_power = int(n_pg * groups_k)
        power_note = f"ANOVA planning (approx): n≈{n_pg} per group × k={groups_k} (α={alpha:g}, power={target_power:g})."

n_model = None
model_note = "—"
if use_model:
    if model_context == "Multiple regression":
        n_model = green_regression_min(k=k_predictors, which="individual")
        model_note = f"Green (1991) individual-predictor rule: n ≥ 104 + k (k={k_predictors})."
    elif model_context == "Logistic regression":
        n_model = logistic_epv_min(k=k_predictors, event_rate=event_rate, epv=epv)
        model_note = f"EPV planning: n ≥ (EPV×k)/event_rate with EPV={epv}, k={k_predictors}, event rate={event_rate:g}."
    elif model_context == "SEM / CFA":
        p = approx_cfa_free_params(latents=latents, indicators_per_latent=indicators_per_latent)
        n_model = sem_min_n_by_ratio(params=p, ratio=sem_ratio)
        model_note = f"SEM planning: params≈{p}, ratio={sem_ratio}:1 ⇒ n≈{n_model}."
    else:
        n_model = None
        model_note = "Model not selected."

candidates = []
labels = []
if n_precision is not None:
    candidates.append(n_precision); labels.append("Precision")
if n_power is not None:
    candidates.append(n_power); labels.append("Power")
if n_model is not None:
    candidates.append(n_model); labels.append("Model")

n_star = int(max(candidates)) if candidates else None

# Binding constraint (handle ties cleanly)
binding = "—"
if n_star is not None:
    tied = []
    if n_precision is not None and n_precision == n_star: tied.append("Precision")
    if n_power is not None and n_power == n_star: tied.append("Power")
    if n_model is not None and n_model == n_star: tied.append("Model")
    binding = ", ".join(tied) if tied else "—"

inflator = (DEFF * HVIF) / max(1e-9, (1 - r))
n_inflated_raw = int(math.ceil((n_star or 0) * inflator)) if n_star is not None else None
exceeds_population = (n_inflated_raw is not None) and (n_inflated_raw > N)
n_inflated = N if exceeds_population else n_inflated_raw

# ============================================================
# Main UI
# ============================================================
st.title("US²DF Sample Size Planner")
st.write(
    "Select the applicable components (Precision, Power, Model). "
    "US²DF uses the max-rule n* = max(selected components), then inflates for DEFF, HVIF, and nonresponse."
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
        f"If a census is not feasible, revise inflation drivers (DEFF/HVIF/nonresponse) "
        f"or revise precision/power/model targets, and report this limitation clearly."
    )

# ============================================================
# Breakdown table
# ============================================================
rows = []
rows.append({
    "Component": "Precision selected?",
    "Value": "Yes" if use_precision else "No",
    "Notes": "Adam (2020) adjusted finite-population precision logic."
})
rows.append({
    "Component": "Power selected?",
    "Value": "Yes" if use_power else "No",
    "Notes": "Power computed from α and target power, with effect-size convention and group design."
})
rows.append({
    "Component": "Model selected?",
    "Value": "Yes" if use_model else "No",
    "Notes": "Model-based planning heuristics for regression, logistic EPV, and SEM/CFA."
})

rows.append({
    "Component": "Sample Size Estimate based on Precision",
    "Value": n_precision if n_precision is not None else "—",
    "Notes": (
        f"Adam (2020): ε=ρe/z with ρ={rho:g}, e={e:g}, z={z_precision:.4f}; n=N/(1+Nε²)"
        if use_precision else "Not applied"
    )
})
rows.append({
    "Component": "Sample Size Estimate based on Power",
    "Value": n_power if n_power is not None else "—",
    "Notes": power_note if use_power else "Not applied"
})
rows.append({
    "Component": "Sample Size Estimate based on Model",
    "Value": n_model if n_model is not None else "—",
    "Notes": model_note if use_model else "Not applied"
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
    "Component": "Final n_inflated",
    "Value": n_inflated if n_inflated is not None else "—",
    "Notes": "n_inflated = n* × DEFF × HVIF × 1/(1−r) (capped at N if needed)"
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
    f"The base sample was then adjusted for field conditions using DEFF={DEFF:g}, HVIF={HVIF:g}, and "
    f"an anticipated nonresponse rate r={r:g}, yielding a final recommended sample size of n={n_inflated:,} "
    f"(Adam, Gyasi, Owusu Jnr & Gyamfi, 2026)."
)

st.code(methods_text, language="text")

st.markdown("---")
st.markdown(
    """
**Cite as:**

> **Adam, A.M., Gyasi, R.M., Owusu Jnr, P. & Gyamfi, E.N. (2026).**  
> *Unified Sample Size Determination Framework (US²DF): Integrating Precision, Power, and Model-Based Requirements for Survey Studies.*
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
