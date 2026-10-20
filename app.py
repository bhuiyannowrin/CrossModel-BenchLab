import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"

st.set_page_config(
    page_title="CrossModel BenchLab",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 { letter-spacing: -0.02em; }
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 12px;
    padding: 14px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_csv(filename):
    path = RESULTS / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_json(filename):
    path = RESULTS / filename
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


folds = load_csv("fold_results.csv")
summary = load_csv("benchmark_summary.csv")
audit = load_csv("fairness_audit_report.csv")
diagnostics = load_csv("violation_audit_details.csv")
ranking = load_csv("ranking_comparison.csv")
paired = load_csv("paired_f1_differences.csv")
changes = load_csv("fold_ranking_changes.csv")
holdout = load_json("holdout_metrics.json")
environment = load_json("experiment_environment.json")

st.title("🔬 CrossModel BenchLab")
st.caption(
    "An Auditable Framework for Fair, Reproducible "
    "and Cost-Aware Machine Learning Benchmarking"
)
st.markdown(
    "**Team Coder Girls** | Track 5 — Research & "
    "Publications | Challenge 06"
)


# Temporary dashboard diagnostics
with st.sidebar.expander("Data Diagnostics", expanded=True):

    st.write("Results folder:", str(RESULTS))
    st.write("Folder exists:", RESULTS.exists())

    required_files = [
        "fold_results.csv",
        "benchmark_summary.csv",
        "fairness_audit_report.csv",
        "violation_audit_details.csv",
        "ranking_comparison.csv",
        "paired_f1_differences.csv",
        "fold_ranking_changes.csv",
        "holdout_metrics.json"
    ]

    for filename in required_files:
        path = RESULTS / filename

        if path.is_file():
            st.success(f"{filename}: FOUND")
        else:
            st.error(f"{filename}: MISSING")

    st.write(
        "Fold data loaded:",
        folds is not None
    )

    if folds is not None:
        st.write(
            "Fold data columns:",
            list(folds.columns)
        )
        st.write(
            "Number of rows:",
            len(folds)
        )



st.sidebar.title("Research Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Model Comparison",
        "Fairness Audit",
        "Violation Experiments",
        "Ranking Sensitivity",
        "Independent Holdout Test",
        "Research Evidence"
    ]
)

st.sidebar.divider()
st.sidebar.caption(
    "Dataset: Breast Cancer Wisconsin (Diagnostic)"
)
st.sidebar.caption(
    "Audit is based on recorded protocol evidence."
)


if page == "Overview":
    st.header("Research Overview")

    st.write(
        "CrossModel BenchLab evaluates whether "
        "machine learning classifiers are compared "
        "under consistent and reproducible conditions."
    )

    if folds is not None:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Models", folds["model"].nunique())
        c2.metric("Outer Evaluations", len(folds))
        c3.metric(
            "Mean F1",
            f"{folds['f1'].mean():.4f}"
        )
        c4.metric(
            "Baseline Audit Checks",
            len(audit) if audit is not None else "N/A"
        )

        st.subheader("F1 Distribution")

        fig = px.box(
            folds,
            x="model",
            y="f1",
            color="model",
            points="all"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "The mean F1 metric above is pooled across "
            "controlled outer-fold evaluations. "
            "It is not the independent holdout result."
        )
    else:
        st.error("fold_results.csv not found.")


elif page == "Model Comparison":
    st.header("Controlled Model Comparison")

    if folds is not None:
        metric = st.selectbox(
            "Evaluation Metric",
            [
                "f1",
                "accuracy",
                "precision",
                "recall",
                "roc_auc",
                "train_seconds",
                "inference_ms_per_sample",
                "model_size_kb"
            ]
        )

        table = (
            folds.groupby("model")[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        st.dataframe(
            table.round(5),
            use_container_width=True
        )

        fig = px.bar(
            table,
            x="model",
            y="mean",
            error_y="std",
            color="model",
            title=f"Mean {metric} by Model"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Fold-Level Results")
        st.dataframe(
            folds.round(5),
            use_container_width=True
        )
    else:
        st.error("Model comparison data unavailable.")


elif page == "Fairness Audit":
    st.header("Benchmark Fairness Audit")

    st.write(
        "The audit checks recorded experimental "
        "metadata against specified protocol rules."
    )

    if audit is not None:
        counts = audit["status"].value_counts()

        c1, c2, c3 = st.columns(3)
        for col, label in zip(
            [c1, c2, c3],
            ["PASS", "FLAGGED", "NOT VERIFIABLE"]
        ):
            col.metric(label, int(counts.get(label, 0)))

        st.dataframe(
            audit,
            use_container_width=True
        )

        fig = px.histogram(
            audit,
            x="check",
            color="status",
            barmode="group",
            title="Protocol Audit Status by Check"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.warning(
            "PASS means the recorded evidence satisfies "
            "the specified rule. It does not prove that "
            "hidden implementation errors are absent."
        )
    else:
        st.error("Fairness audit file unavailable.")


elif page == "Violation Experiments":
    st.header("Intentional Violation Experiments")

    if diagnostics is not None:
        st.dataframe(
            diagnostics,
            use_container_width=True
        )

        valid = diagnostics[
            diagnostics["status"] != "NOT VERIFIABLE"
        ].copy()

        truth = valid["expected_violation"].astype(bool)
        detected = valid["status"].eq("FLAGGED")

        tp = int((truth & detected).sum())
        fp = int((~truth & detected).sum())

        fn = int((
            diagnostics["expected_violation"].astype(bool)
            & diagnostics["status"].ne("FLAGGED")
        ).sum())

        tn = int((~truth & ~detected).sum())

        rate = (
            tp / (tp + fn)
            if tp + fn else None
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Positives", tp)
        c2.metric("False Positives", fp)
        c3.metric("False Negatives", fn)
        c4.metric(
            "Detection Rate",
            f"{rate:.1%}" if rate is not None else "N/A"
        )

        figure = px.histogram(
            diagnostics,
            x="scenario",
            color="status",
            barmode="group",
            title="Audit Outcomes by Scenario"
        )
        st.plotly_chart(
            figure,
            use_container_width=True
        )

        st.info(
            "Detection rate applies only to the "
            "documented injected protocol violations."
        )
    else:
        st.error("Diagnostic audit results unavailable.")


elif page == "Ranking Sensitivity":
    st.header("Model Ranking Sensitivity")

    if ranking is not None:
        st.dataframe(
            ranking.round(5),
            use_container_width=True
        )

        fig = px.bar(
            ranking,
            x="model",
            y="mean",
            color="scenario",
            barmode="group",
            title="Mean F1 Under Different Protocols"
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )

        if paired is not None:
            st.subheader("Paired SVM F1 Differences")
            st.dataframe(
                paired.round(5),
                use_container_width=True
            )

            shift = px.box(
                paired,
                x="scenario",
                y="delta_f1",
                color="scenario",
                points="all"
            )
            st.plotly_chart(
                shift,
                use_container_width=True
            )

        if changes is not None:
            st.subheader("Fold-Level Ranking Changes")
            st.dataframe(
                changes,
                use_container_width=True
            )

        st.caption(
            "Only SVM is modified in diagnostic "
            "scenarios. LR and RF remain at reference "
            "conditions. The results are exploratory."
        )
    else:
        st.error("Ranking results unavailable.")


elif page == "Independent Holdout Test":
    st.header("Independent Holdout Evaluation")

    if holdout is not None:
        st.success(
            "Selected model: "
            + str(holdout["selected_model"])
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Test F1",
            f"{holdout['f1']:.4f}"
        )
        c2.metric(
            "Test Accuracy",
            f"{holdout['accuracy']:.4f}"
        )
        c3.metric(
            "Test ROC-AUC",
            f"{holdout['roc_auc']:.4f}"
        )

        metric_names = [
            "accuracy", "precision",
            "recall", "f1", "roc_auc"
        ]

        metrics = pd.DataFrame({
            "Metric": metric_names,
            "Value": [
                holdout[key] for key in metric_names
            ]
        })

        st.dataframe(
            metrics,
            use_container_width=True
        )

        image_path = (
            RESULTS / "holdout_confusion_matrix.png"
        )

        if image_path.exists():
            st.image(
                str(image_path),
                caption="Independent Holdout Confusion Matrix"
            )

        st.info(
            "The holdout partition was not used "
            "during model selection or diagnostic "
            "protocol violation experiments. "
            "Positive class 1 represents benign cases."
        )
    else:
        st.error("Holdout metrics unavailable.")


elif page == "Research Evidence":
    st.header("Research Evidence")

    st.write(
        "Download the experimental records and "
        "inspect the reproducibility metadata."
    )

    if environment is not None:
        st.subheader("Experiment Environment")
        st.json(environment)

    available = sorted(RESULTS.glob("*"))

    if not available:
        st.warning("No research evidence files found.")

    for path in available:
        if path.is_file():
            st.download_button(
                label=f"Download {path.name}",
                data=path.read_bytes(),
                file_name=path.name,
                mime="application/octet-stream",
                key=path.name
            )

    st.divider()
    st.subheader("Research Limitations")

    st.markdown("""
- One public dataset and three classifier families.
- Two deliberately altered SVM conditions.
- Metadata-based checks cannot detect every hidden error.
- Diagnostic results use five outer folds.
- Repeated folds are not independent observations.
- Equal configuration counts do not imply equal compute.
- Holdout results are not a clinical validation.
""")

    st.subheader("Development Attribution")
    st.write(
        "AI-assisted coding and documentation were "
        "used during development. The team is "
        "responsible for reviewing, testing and "
        "explaining the submitted work."
    )

st.divider()
st.caption(
    "CrossModel BenchLab | Team Coder Girls | "
    "Research prototype — not a medical device"
)
