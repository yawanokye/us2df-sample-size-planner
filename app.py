# app.py
import math
import pandas as pd
import streamlit as st

# ============================================================
# US²DF Sample Size Planner
# - Choose any combination of:
#   (1) Descriptive estimation (Precision)
#   (2) Hypothesis testing (Power)
#   (3) Model-based analysis (Model)
# - Max-rule: n* = max(selected components)
# - Inflation: n_inflated = n* × DEFF × HVIF × 1/(1-r)
# - Caps n_inflated at N (population) with caution message
#
# Notes:
# - Power component uses a simple normal-approximation planning formula (no SciPy).
# - SEM/CFA parameter count is a planning approximation.
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


def z_value_from_alpha(alpha: float) -> float:
    """Two-sided z critical for given alpha (e.g., alpha=0.05 => 1.96)."""
    alpha = float(alpha)
    alpha = min(max(alpha, 1e-9), 0.999999999)
    return _norm_ppf(1 - alpha / 2)


def z_value_from_conf(conf_level: float) -> float:
    """Two-sided z critical for confidence level (e.g., 0.95 => 1.96)."""
    alpha = 1 - float(conf_level)
    return z_value_from_alpha(alpha)


def z_beta_from_power(power: float) -> float:
    """z value for desired power: z_(1-beta)."""
    power = float(power)
    power = min(max(power, 1e-9), 0.999999999)
    return _norm_ppf(power)


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
    Identification choices can change this in real models.
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


# ----------------------------
# Power planning (no SciPy)
# ----------------------------
def power_n_two_group_continuous(alpha: float, power: float, d: float, allocation: float = 0.5) -> int:
    """
    Two-group comparison, continuous outcome (standardized mean difference d).
    Balanced groups by default (allocation=0.5).
    Total n ≈ (zα/2 + zβ)^2 * (1/p + 1/(1-p)) / d^2
    For p=0.5, (1/p + 1/(1-p)) = 4.
    """
    d = max(1e-9, float(d))
    p = min(max(float(allocation), 0.05), 0.95)
    z_a = z_value_from_alpha(alpha)
    z_b = z_beta_from_power(power)
    n_total = ((z_a + z_b) ** 2) * (1 / p + 1 / (1 - p)) / (d ** 2)
    return int(math.ceil(n_total))


def power_n_two_proportions(alpha: float, power: float, p1: float, p2: float, allocation: float = 0.5) -> int:
    """
    Two-group comparison, proportions.
    Uses normal approximation planning formula with pooled variance.
    Total n ≈ [ zα/2*sqrt(2*pbar(1-pbar)) + zβ*sqrt(p1(1-p1)+p2(1-p2)) ]^2 / (p1-p2)^2
    Then adjusted for allocation p (defaults balanced).
    """
    p1 = min(max(float(p1), 1e-6), 1 - 1e-6)
    p2 = min(max(float(p2), 1e-6), 1 - 1e-6)
    diff = abs(p1 - p2)
    diff = max(diff, 1e-9)

    z_a = z_value_from_alpha(alpha)
    z_b = z_beta_from_power(power)

    pbar = (p1 + p2) / 2
    term1 = z_a * math.sqrt(2 * pbar * (1 - pbar))
    term2 = z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    n_per_group = ((term1 + term2) ** 2) / (diff ** 2)

    # allocation adjustment (optional)
    alloc = min(max(float(allocation), 0.05), 0.95)
    # If unbalanced, total rises by factor (1/alloc + 1/(1-alloc))/4 relative to balanced
    adj = (1 / alloc + 1 / (1 - alloc)) / 4
    n_total = 2 * n_per_group * adj
    return int(math.ceil(n_total))


def power_n_anova_continuous(alpha: float, power: float, f: float, groups: int) -> int:
    """
    Practical planning approximation for one-way ANOVA with Cohen's f.
    Uses a rough normal-approx style: n_total ≈ ( (zα + zβ)^2 / f^2 ) * (groups - 1)
    This is a planning guide, not an exact F-based calculation.
    """
    f = max(1e-9, float(f))
    g = max(3, int(groups))
    z_a = z_value_from_alpha(alpha)
    z_b = z_beta_from_power(power)
    n_total = ((z_a + z_b) ** 2) * (g - 1) / (f ** 2)
    return int(math.ceil(n_total))


# ============================================================
# Sidebar Inputs (Step-by-step)
# ============================================================
st.sidebar.title("Inputs (Step-by-step)")

st.sidebar.markdown("### Step 1. Choose components")
st.sidebar.caption("Tick what applies. You can choose one, two, or all three.")
precision_in_play = st.sidebar.checkbox("Descriptive estimation (Precision)", value=False)
power_in_play = st.sidebar.checkbox("Hypothesis testing (Power)", value=True)
model_in_play = st.sidebar.checkbox("Model-based analysis (Model)", value=False)

if not (precision_in_play or power_in_play or model_in_play):
    st.sidebar.error("Select at least one component.")
    precision_in_play = True

st.sidebar.markdown("### Step 2. Outcome type")
st.sidebar.caption("Default is Categorical (proportions).")
outcome_type = st.sidebar.selectbox(
    "Measurement Scale of the Estimand",
    ["Categorical (proportions)", "Continuous (means, scales)"],
    index=0,
)
is_categorical = outcome_type.startswith("Categorical")

st.sidebar.markdown("### Step 3. Population and confidence")
N = int(st.sidebar.number_input("Population size (N)", min_value=1, value=50000, step=100))
conf_level = st.sidebar.radio("Confidence level (for precision)", ["95%", "99%"], index=0)
conf_level_val = 0.95 if conf_level == "95%" else 0.99
t_conf = z_value_from_conf(conf_level_val)

# ----------------------------
# Precision settings (Step 4) — only show if selected
# ----------------------------
epsilon = None
rho = None
e = None
n_precision = None

st.sidebar.markdown("### Step 4. Precision settings")
if precision_in_play:
    st.sidebar.caption("If you’re estimating a proportion or mean, set your desired accuracy (e).")
    rho = 2.0 if is_categorical else 4.0
    default_e = 0.05 if is_categorical else 0.03
    e_key = "e_categorical" if is_categorical else "e_continuous"

    e = st.sidebar.number_input(
        "Desired degree of accuracy (e)",
        min_value=0.001,
        max_value=0.20,
        value=default_e,
        step=0.001,
        key=e_key
    )
    epsilon = adam_epsilon(rho=rho, e=e, t=t_conf)
    st.sidebar.caption("Typical defaults: e=0.05 for proportions, e=0.03 for continuous scales.")
else:
    st.sidebar.caption("Not selected, precision calculation will be skipped.")

# ----------------------------
# Power settings (Step 5) — only show if selected
# ----------------------------
alpha = None
power = None
design_type = None
groups_g = 1
n_power = None
n_power_note = ""

st.sidebar.markdown("### Step 5. Power settings")
if power_in_play:
    st.sidebar.caption("Set α and power. Defaults are α=0.05 and 80% power.")

    alpha = st.sidebar.number_input(
        "Significance level (α)",
        min_value=0.001, max_value=0.20, value=0.05, step=0.001
    )
    power = st.sidebar.number_input(
        "Desired power (1−β)",
        min_value=0.50, max_value=0.99, value=0.80, step=0.01
    )

    design_type = st.sidebar.selectbox(
        "Inferential design",
        ["Two-group comparison", "ANOVA (3+ groups)"],
        index=0
    )

    if design_type == "Two-group comparison":
        groups_g = 2
        st.sidebar.caption("Groups (g): 2")

        if is_categorical:
            st.sidebar.caption("For proportions, enter expected p1 and p2.")
            p1 = st.sidebar.number_input("Expected proportion in Group 1 (p1)", min_value=0.01, max_value=0.99, value=0.50, step=0.01)
            p2 = st.sidebar.number_input("Expected proportion in Group 2 (p2)", min_value=0.01, max_value=0.99, value=0.60, step=0.01)
            n_power = power_n_two_proportions(alpha=alpha, power=power, p1=p1, p2=p2)
            n_power_note = f"Two-proportion planning: p1={p1:.2f}, p2={p2:.2f}, α={alpha:.3f}, power={power:.2f}."
        else:
            st.sidebar.caption("For continuous outcomes, enter Cohen’s d (standardised mean difference).")
            d = st.sidebar.number_input("Expected effect size (Cohen’s d)", min_value=0.10, max_value=2.00, value=0.50, step=0.05)
            n_power = power_n_two_group_continuous(alpha=alpha, power=power, d=d)
            n_power_note = f"Two-sample planning: d={d:.2f}, α={alpha:.3f}, power={power:.2f}."

    else:
        groups_g = int(st.sidebar.number_input("Number of groups (g)", min_value=3, max_value=50, value=3, step=1))
        if is_categorical:
            st.sidebar.caption("ANOVA is for continuous outcomes. For categorical outcomes, consider logistic models or chi-square planning.")
            # Still allow a rough planning via continuous f to avoid blocking the UI completely
        st.sidebar.caption("Enter Cohen’s f (small=0.10, medium=0.25, large=0.40).")
        f = st.sidebar.number_input("Expected effect size (Cohen’s f)", min_value=0.05, max_value=1.00, value=0.25, step=0.05)
        n_power = power_n_anova_continuous(alpha=alpha, power=power, f=f, groups=groups_g)
        n_power_note = f"ANOVA planning: f={f:.2f}, g={groups_g}, α={alpha:.3f}, power={power:.2f}."
else:
    st.sidebar.caption("Not selected, power calculation will be skipped.")

# ----------------------------
# Model settings (Step 6) — only show if selected
# ----------------------------
st.sidebar.markdown("### Step 6. Model settings")
model_context = "None"
k_predictors = 10
event_rate = 0.20
epv = 10
latents = 3
indicators_per_latent = 4
sem_ratio = 10
n_model = None
sem_params_p = None
model_note = ""

if model_in_play:
    st.sidebar.caption("Pick the main analysis model to apply a planning heuristic.")
    model_context = st.sidebar.selectbox(
        "Model type",
        ["Multiple regression", "Logistic regression", "SEM / CFA"],
        index=0
    )

    if model_context == "Multiple regression":
        k_predictors = int(st.sidebar.number_input("Number of predictors (k)", min_value=1, value=10, step=1))
        n_model = green_regression_min(k=k_predictors, which="individual")
        model_note = f"Green (1991) individual predictors rule: n ≥ 104 + k (k={k_predictors})."

    elif model_context == "Logistic regression":
        k_predictors = int(st.sidebar.number_input("Number of predictors (k)", min_value=1, value=10, step=1))
        event_rate = float(st.sidebar.number_input("Event rate", min_value=0.01, max_value=0.99, value=0.20, step=0.01))
        epv = int(st.sidebar.number_input("EPV (events per variable)", min_value=5, max_value=50, value=10, step=1))
        n_model = logistic_epv_min(k=k_predictors, event_rate=event_rate, epv=epv)
        model_note = f"EPV planning: total n ≥ (EPV×k)/event_rate = ({epv}×{k_predictors})/{event_rate:.2f}."

    elif model_context == "SEM / CFA":
        latents = int(st.sidebar.number_input("Latent variables", min_value=1, value=3, step=1))
        indicators_per_latent = int(st.sidebar.number_input("Indicators per latent", min_value=2, value=4, step=1))
        sem_ratio = int(st.sidebar.number_input("n per parameter ratio", min_value=5, max_value=30, value=10, step=1))
        sem_params_p = approx_cfa_free_params(latents=latents, indicators_per_latent=indicators_per_latent)
        n_model = sem_min_n_by_ratio(params=sem_params_p, ratio=sem_ratio)
        model_note = f"SEM/CFA planning: p≈{sem_params_p} free params, n≈p×{sem_ratio}."
else:
    st.sidebar.caption("Not selected, model-based calculation will be skipped.")

# ----------------------------
# Field adjustments (Step 7)
# ----------------------------
st.sidebar.markdown("### Step 7. Field adjustments")
st.sidebar.caption("Use these when you expect clustering, measurement inflation, or nonresponse.")

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
candidates = {}

# Precision
if precision_in_play:
    n_precision = adam_n_precision(N=N, epsilon=epsilon)
    candidates["Precision"] = n_precision

# Power
if power_in_play:
    candidates["Power"] = n_power

# Model
if model_in_play:
    candidates["Model"] = n_model

# Max-rule (selected components only)
n_star = int(max(candidates.values()))
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
    "This planner lets you tick the components that apply (Precision, Power, Model). "
    "It then applies the max-rule and adjusts for field conditions."
)

c1, c2, c3 = st.columns(3)
c1.metric("Sample Size Estimate based on Precision", f"{n_precision:,}" if precision_in_play else "—")
c2.metric("Sample Size Estimate based on Power", f"{n_power:,}" if power_in_play else "—")
c3.metric("Sample Size Estimate based on Model", f"{n_model:,}" if model_in_play else "—")

st.markdown("## US²DF Recommendation")
colA, colB = st.columns(2)
colA.metric("Base (max-rule) sample size, n*", f"{n_star:,}")
colB.metric("Adjusted and Final Recommended Sample Size", f"{n_inflated:,}")

st.success(f"Binding constraint(s): **{binding_text}**")

if exceeds_population:
    st.warning(
        f"**Caution:** The adjusted sample size ({n_inflated_raw:,}) exceeds the population (N={N:,}). "
        f"Consider a **census/near-census** where feasible. "
        f"If not feasible, revise DEFF/HVIF/nonresponse assumptions, or revise precision/power/model targets, "
        f"and report this limitation clearly."
    )

# Breakdown table
rows = []

rows.append({
    "Component": "Sample Size Estimate based on Precision",
    "Value": n_precision if precision_in_play else "—",
    "Notes": (
        f"Adam (2020): ε=ρe/t with ρ={rho:g}, e={e:g}, z={t_conf:.4f}; n=N/(1+Nε²)"
        if precision_in_play else "Not selected."
    )
})

rows.append({
    "Component": "Sample Size Estimate based on Power",
    "Value": n_power if power_in_play else "—",
    "Notes": n_power_note if power_in_play else "Not selected."
})

rows.append({
    "Component": "Sample Size Estimate based on Model",
    "Value": n_model if model_in_play else "—",
    "Notes": model_note if model_in_play else "Not selected."
})

rows.append({
    "Component": "Base sample size (max-rule), n*",
    "Value": n_star,
    "Notes": f"n* = max({', '.join(candidates.keys())})"
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
    "Component": "Inflation factor",
    "Value": round(inflator, 4),
    "Notes": "Inflator = DEFF × HVIF ÷ (1−r)"
})
rows.append({
    "Component": "Final Recommended Sample Size",
    "Value": n_inflated,
    "Notes": "n_inflated = n* × DEFF × HVIF × 1/(1−r) (capped at N if needed)"
})

df_breakdown = pd.DataFrame(rows)
st.dataframe(df_breakdown, use_container_width=True)

# Copy-ready methods text
st.subheader("Copy-ready Methods text")

selected = ", ".join(candidates.keys())

precision_clause = ""
if precision_in_play:
    precision_clause = (
        f"Precision requirements were computed using Adam (2020) "
        f"(ε=ρe/t; ρ={rho:g}, e={e:g}, z={t_conf:.4f}) giving n_precision={n_precision:,}. "
    )

power_clause = ""
if power_in_play:
    power_clause = f"Power planning used α={alpha:.3f} and power={power:.2f} ({n_power_note}) giving n_power={n_power:,}. "

model_clause = ""
if model_in_play:
    model_clause = f"Model-based requirements were assessed using planning heuristics (n_model={n_model:,}). "

methods_text = (
    f"Sample size planning followed the US²DF framework using selected components ({selected}) "
    f"and the max-rule (n* = max(n_precision, n_power, n_model) over selected components). "
    f"{precision_clause}"
    f"{power_clause}"
    f"{model_clause}"
    f"The base sample size (n*={n_star:,}) was adjusted for field conditions using "
    f"DEFF={DEFF:g}, HVIF={HVIF:g}, and nonresponse r={r:g}, "
    f"yielding a final recommended sample size of n={n_inflated:,} "
    f"(Adam, Gyasi, Owusu Junior & Gyamfi, 2026)."
)

st.code(methods_text, language="text")

# Required reference line under the methods text
st.markdown("---")
st.markdown(
    """
**Cite as:**

> **Adam, A.M., Gyasi, R.M., Owusu Junior, P. & Gyamfi, E.N. (2026).**  
*Unified Sample Size Determination Framework (US²DF): Integrating Precision, Power, and Model-Based Requirements for Survey Research.*
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
