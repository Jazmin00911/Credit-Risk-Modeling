"""Thin Streamlit front end. All modelling logic lives in the `creditrisk` package."""
from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without `pip install -e .`

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import streamlit as st  # noqa: E402

from creditrisk.config import Config  # noqa: E402
from creditrisk.model import CreditModelBundle  # noqa: E402
from creditrisk.stress import SCENARIOS, scenario_z, vasicek_pd  # noqa: E402
from creditrisk.train import main as train_main  # noqa: E402

cfg = Config()
JOB_LABELS = {0: "0 - unskilled, non-resident", 1: "1 - unskilled, resident",
              2: "2 - skilled", 3: "3 - highly skilled / management"}


@st.cache_resource(show_spinner="First launch: training both models from the committed data (about a minute)...")
def load_bundle(name: str) -> CreditModelBundle:
    """Load a trained bundle. If none exists (fresh clone or cloud deploy), train into a temp directory."""
    path = cfg.artifact_dir / f"{name}_v1.joblib"
    if not path.exists():
        tmp = Path(tempfile.gettempdir())
        build = dataclasses.replace(cfg, artifact_dir=tmp / "creditrisk_artifacts", report_dir=tmp / "creditrisk_reports")
        path = build.artifact_dir / f"{name}_v1.joblib"
        if not path.exists():
            train_main(build)
    return CreditModelBundle.load(path)


st.set_page_config(page_title="Credit Risk Decision Support", layout="wide")
st.title("Credit Risk Decision Support")
st.caption("Demonstration on the public German Credit dataset. Not for real lending decisions.")

name = st.sidebar.radio("Model", ["scorecard", "xgb"],
                        format_func=lambda n: {"scorecard": "WoE scorecard (champion)",
                                               "xgb": "XGBoost (challenger)"}[n])
bundle = load_bundle(name)
cats, ranges = bundle.metadata["categories"], bundle.metadata["numeric_ranges"]
st.sidebar.metric("Decline cutoff (PD)", f"{bundle.threshold:.1%}")
st.sidebar.caption(f"Cutoff minimises expected cost with c_FN={bundle.metadata['costs']['c_fn']:g}, "
                   f"c_FP={bundle.metadata['costs']['c_fp']:g}.")

c1, c2 = st.columns(2)
with c1:
    job = st.selectbox("Job", list(JOB_LABELS), index=2, format_func=JOB_LABELS.get)
    housing = st.selectbox("Housing", cats["Housing"])
    saving = st.selectbox("Saving accounts", cats["Saving accounts"], index=cats["Saving accounts"].index("little"))
    checking = st.selectbox("Checking account", cats["Checking account"], index=cats["Checking account"].index("little"))
with c2:
    amount = st.number_input("Credit amount", min_value=0, value=int(np.median(ranges["Credit amount"])), step=100)
    duration = st.number_input("Duration (months)", min_value=1, max_value=120, value=18)
    purpose = st.selectbox("Purpose", cats["Purpose"])

applicant = pd.DataFrame([{"Job": job, "Credit amount": amount, "Duration": duration, "Housing": housing,
                           "Saving accounts": saving, "Checking account": checking, "Purpose": purpose}])

if st.button("Assess application", type="primary"):
    pd_hat = float(bundle.predict_proba(applicant)[0])
    declined = pd_hat >= bundle.threshold
    t_dec, t_why, t_stress, t_about = st.tabs(["Decision", "Why this decision", "Stress test", "Model card"])

    with t_dec:
        m1, m2 = st.columns(2)
        m1.metric("Probability of default", f"{pd_hat:.1%}", delta=f"{pd_hat - bundle.threshold:+.1%} vs cutoff",
                  delta_color="inverse")
        (m2.error if declined else m2.success)("DECLINE" if declined else "APPROVE")
        if declined:
            st.subheader("Principal reasons")
            for r in bundle.reason_codes(applicant):
                st.write(f"- **{r['reason']}** ({r['feature']} = {r['applicant_value']})")

    with t_why:
        st.write("Contribution of each factor to the applicant's log-odds of default. "
                 "Red pushes toward decline, blue toward approval.")
        shap.plots.waterfall(bundle.explain(applicant)[0], show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

    with t_stress:
        rows = [{"scenario": s, "PD": float(vasicek_pd(np.array([pd_hat]), scenario_z(sh), bundle.metadata["rho"])[0])}
                for s, sh in SCENARIOS.items()]
        df = pd.DataFrame(rows).assign(decision=lambda d: np.where(d["PD"] >= bundle.threshold, "DECLINE", "APPROVE"))
        st.dataframe(df.style.format({"PD": "{:.1%}"}), hide_index=True)
        st.caption("Macro loadings and asset correlation are stated assumptions, not estimates.")

    with t_about:
        h = bundle.metadata["holdout"]
        st.write(f"Trained on {bundle.metadata['n_rows']} applicants; default rate {bundle.metadata['default_rate']:.1%}.")
        st.write(f"Holdout Gini {h['test']['gini']:.3f} (95% CI {h['test']['gini_ci_low']:.2f} to "
                 f"{h['test']['gini_ci_high']:.2f}); cross-validated Gini {h['cv_oof']['gini']:.3f}.")
        st.json(bundle.metadata["libs"])
