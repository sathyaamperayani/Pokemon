
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Pokémon Type Classifier", page_icon="🔮", layout="wide")

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    raw = pd.read_csv("data/raw_data.csv")
    clean = pd.read_csv("data/clean_data.csv")
    return raw, clean


@st.cache_data
def load_metrics():
    with open("model_metrics.json") as f:
        return json.load(f)


@st.cache_resource
def load_model_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder


raw_df, clean_df = load_data()
metrics = load_metrics()
FEATURES = metrics["features"]
CLASS_LABELS = metrics["class_labels"]
BEST_MODEL_KEY = metrics["best_model"]
BEST_MODEL_NAME = BEST_MODEL_KEY.replace("_", " ").title()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("🔮 Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "🗂️ Data Overview",
        "🔍 EDA",
        "📈 Model Performance",
        "🎯 Live Prediction",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter (EDA & charts)")
type_filter = st.sidebar.multiselect(
    "Primary type(s)",
    options=sorted(clean_df["primary_type"].unique()),
    default=sorted(clean_df["primary_type"].unique()),
)
filtered_df = clean_df[clean_df["primary_type"].isin(type_filter)]

# ---------------------------------------------------------------------------
# Section 1: Overview
# ---------------------------------------------------------------------------

if section == "🏠 Overview":
    st.title("🔮 Pokémon Primary Type Classifier")
    st.markdown(
        """
        This dashboard walks through a full data science pipeline built on the free,
        no-auth **[PokéAPI](https://pokeapi.co/)**: pulling ~1,000 Pokémon records via
        a paginated REST API, cleaning and engineering features from their base stats,
        training and comparing classification models, and serving live predictions of
        a Pokémon's **primary type** from its stats alone.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows (clean data)", f"{len(clean_df):,}")
    col2.metric("Classes", len(CLASS_LABELS))
    col3.metric("Best model", BEST_MODEL_NAME)
    best_f1 = metrics["all_model_results"][BEST_MODEL_KEY]["macro_f1"]
    col4.metric("Best macro F1", f"{best_f1:.3f}")

    st.info(
        "Heads up: with 18 fairly overlapping classes and only base stats to go on, "
        "accuracy here is modest (~21%) — well above the ~5.6% random-guess baseline, "
        "but a reminder that base stats alone don't fully determine a Pokémon's type. "
        "See the **Model Performance** section for the full picture."
    )

# ---------------------------------------------------------------------------
# Section 2: Data Overview
# ---------------------------------------------------------------------------

elif section == "🗂️ Data Overview":
    st.title("🗂️ Data Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw rows", f"{len(raw_df):,}")
    c2.metric("Raw columns", raw_df.shape[1])
    c3.metric("Clean rows", f"{len(clean_df):,}")
    c4.metric("Clean columns", clean_df.shape[1])

    tab1, tab2 = st.tabs(["Raw data preview", "Clean data preview"])
    with tab1:
        st.dataframe(raw_df.head(20), use_container_width=True)
    with tab2:
        st.dataframe(clean_df.head(20), use_container_width=True)

    st.subheader("Missing values: raw vs. clean")
    raw_missing = raw_df.isna().sum()
    raw_missing = raw_missing[raw_missing > 0]
    clean_missing = clean_df.isna().sum()
    clean_missing = clean_missing[clean_missing > 0]

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.caption("Raw data")
        if raw_missing.empty:
            st.success("No missing values found in raw data.")
        else:
            st.dataframe(raw_missing.rename("missing count"))
    with mcol2:
        st.caption("Clean data")
        if clean_missing.empty:
            st.success("No missing values remain after cleaning.")
        else:
            st.dataframe(clean_missing.rename("missing count"))

    st.caption(
        f"Total missing cells — raw: {int(raw_df.isna().sum().sum())}, "
        f"clean: {int(clean_df.isna().sum().sum())}"
    )

# ---------------------------------------------------------------------------
# Section 3: EDA
# ---------------------------------------------------------------------------

elif section == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")
    st.caption("Use the sidebar filter to include/exclude primary types from both charts below.")

    st.subheader("Class balance")
    counts = filtered_df["primary_type"].value_counts().reset_index()
    counts.columns = ["primary_type", "count"]
    fig_bar = px.bar(
        counts.sort_values("count", ascending=False),
        x="primary_type",
        y="count",
        color="primary_type",
        title="Number of Pokémon per primary type",
    )
    fig_bar.update_layout(showlegend=False, xaxis_title="Primary type", yaxis_title="Count")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Feature correlation heatmap")
    numeric_cols = FEATURES + ["n_types"]
    numeric_cols = [c for c in numeric_cols if c in filtered_df.columns]
    corr = filtered_df[numeric_cols].corr()
    fig_heat = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation between numeric features",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.caption(f"Showing {len(filtered_df):,} of {len(clean_df):,} Pokémon based on the sidebar filter.")

# ---------------------------------------------------------------------------
# Section 4: Model Performance
# ---------------------------------------------------------------------------

elif section == "📈 Model Performance":
    st.title("📈 Model Performance")

    st.subheader("Model comparison")
    results_df = pd.DataFrame(metrics["all_model_results"]).T
    results_df.index = [i.replace("_", " ").title() for i in results_df.index]
    results_df = results_df.rename(
        columns={
            "accuracy": "Accuracy",
            "macro_precision": "Macro Precision",
            "macro_recall": "Macro Recall",
            "macro_f1": "Macro F1",
        }
    )
    st.dataframe(results_df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)
    st.caption(f"Best model selected by macro F1: **{BEST_MODEL_NAME}**")

    st.subheader(f"Confusion matrix — {BEST_MODEL_NAME}")
    cm = np.array(metrics["confusion_matrix"])
    fig_cm = px.imshow(
        cm,
        x=CLASS_LABELS,
        y=CLASS_LABELS,
        color_continuous_scale="Blues",
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig_cm.update_layout(height=650)
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Feature importance")
    fi = pd.Series(metrics["feature_importance"]).sort_values(ascending=True)
    fig_fi = px.bar(
        fi,
        x=fi.values,
        y=fi.index,
        orientation="h",
        title=f"Feature importance — {BEST_MODEL_NAME}",
        labels={"x": "Importance", "y": "Feature"},
    )
    st.plotly_chart(fig_fi, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 5: Live Prediction
# ---------------------------------------------------------------------------

elif section == "🎯 Live Prediction":
    st.title("🎯 Live Prediction")
    st.markdown("Set a Pokémon's base stats below, then hit **Predict** to classify its primary type.")

    try:
        model, scaler, label_encoder = load_model_artifacts()
        artifacts_ok = True
    except Exception as e:
        artifacts_ok = False
        st.error(
            "Couldn't load model.pkl / scaler.pkl / label_encoder.pkl. "
            "Make sure all three files are committed at the repo root and were "
            "downloaded/uploaded as raw binary (not through a text editor). "
            f"Details: {e}"
        )

    input_values = {}
    cols = st.columns(3)
    for i, feat in enumerate(FEATURES):
        col_min = int(clean_df[feat].min())
        col_max = int(clean_df[feat].max())
        col_mean = int(clean_df[feat].mean())
        with cols[i % 3]:
            input_values[feat] = st.slider(
                feat.replace("_", " ").title(),
                min_value=col_min,
                max_value=col_max,
                value=col_mean,
            )

    if st.button("🔮 Predict", type="primary", disabled=not artifacts_ok):
        X = pd.DataFrame([[input_values[f] for f in FEATURES]], columns=FEATURES)
        X_scaled = scaler.transform(X)
        pred_encoded = model.predict(X_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        st.success(f"Predicted primary type: **{pred_label.upper()}**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]
            proba_df = pd.DataFrame(
                {"type": label_encoder.inverse_transform(np.arange(len(proba))), "probability": proba}
            ).sort_values("probability", ascending=False)
            fig_proba = px.bar(
                proba_df,
                x="probability",
                y="type",
                orientation="h",
                title="Predicted probability by type",
            )
            fig_proba.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_proba, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption(f"Data source: PokéAPI (pokeapi.co)")
st.sidebar.caption(f"Last updated: {date.today().isoformat()}")