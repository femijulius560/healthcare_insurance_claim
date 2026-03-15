import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import uuid
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Page Config
st.set_page_config(
    page_title="💎 Interactive Insurance Claim Dashboard",
    layout="wide",
    page_icon="💰"
)


# Load Model
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "rf_pipeline.pkl")
if not os.path.exists(MODEL_PATH):
    alternative_paths = [
        os.path.join(os.getcwd(), "models", "rf_pipeline.pkl"),
        "models/rf_pipeline.pkl",
    ]
    for alt_path in alternative_paths:
        if os.path.exists(alt_path):
            MODEL_PATH = alt_path
            break
    else:
        st.error(f"Model not found at {MODEL_PATH}")
        st.stop()


def normalize_cat_cols(X):   # ensure consistent formatting for categorical columns
    X = X.copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype(str).str.lower().str.strip()
    return X


def predict_claims_from_log_model(pipeline, features_df): # predict claims from log model, applying smearing factor if available.
    pred_log = pipeline.predict(features_df)
    smear_factor = getattr(pipeline, "smearing_factor_", 1.0)
    pred_claims = np.exp(pred_log) * smear_factor - 1
    return np.maximum(pred_claims, 0)


def apply_tail_adjustment(pred_claims, quantile, multiplier):  
    adjusted = pred_claims.copy()
    threshold = np.quantile(adjusted, quantile)
    tail_mask = adjusted >= threshold
    if tail_mask.sum() >= 1:
        adjusted[tail_mask] = adjusted[tail_mask] * multiplier
    return adjusted


def build_single_input_df(age, bmi, bloodpressure, children, gender, smoker, diabetic, region):
    return pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "bloodpressure": [bloodpressure],
        "children": [children],
        "gender": [gender],
        "smoker": [smoker],
        "diabetic": [diabetic],
        "region": [region]
    })


def compute_single_shap_contributions(pipeline, features_df):
    shap_module = __import__("shap")

    if not hasattr(pipeline, "named_steps"):
        raise ValueError("Loaded model is not a sklearn Pipeline with named_steps.")

    estimator = pipeline.named_steps[list(pipeline.named_steps)[-1]]
    X_proc = pipeline[:-1].transform(features_df)

    # Target the ColumnTransformer step directly to get post-encoding feature names;
    # avoids hitting the FunctionTransformer normalizer which has no get_feature_names_out.
    col_transformer = pipeline.named_steps.get("preprocessor", None)
    if col_transformer is not None and hasattr(col_transformer, "get_feature_names_out"):
        feature_names = col_transformer.get_feature_names_out().tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(X_proc.shape[1])]

    explainer = shap_module.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_proc)

    if isinstance(shap_values, list):
        shap_row = np.array(shap_values[0])[0]
    else:
        shap_row = np.array(shap_values)[0]

    # Convert log-space SHAP values to dollar contributions.
    # Model predicts log1p(claim), so for each feature i:
    #   dollar_i = exp(log_pred) * smear * (1 - exp(-shap_i))
    # This isolates each feature's marginal dollar effect on the final prediction.
    log_pred = pipeline.predict(features_df)[0]
    smear_factor = getattr(pipeline, "smearing_factor_", 1.0)
    dollar_contribs = np.exp(log_pred) * smear_factor * (1 - np.exp(-shap_row))

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_row,
        "dollar_contribution": dollar_contribs
    })
    shap_df["abs_contrib"] = shap_df["dollar_contribution"].abs()
    shap_df = shap_df.sort_values("abs_contrib", ascending=False)
    return shap_df

model = joblib.load(MODEL_PATH)  # load the trained model pipeline


# Header
st.markdown("<h1 style='text-align:center; color:#4B0082;'>💎 Healthcare Insurance Claim Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload your clean dataset, filter, predict, and explore claims interactively.</p>", unsafe_allow_html=True)
st.markdown("---")

tab_single, tab_batch, tab_fraud = st.tabs([
    "📌 Single Prediction & Explainability",
    "📂 Batch Analytics",
    "🚨 Fraud Detection System"
])

with tab_single:
    st.subheader(" Flash Prediction")
    st.caption("Fast claim estimate for one profile.")

    if "session_predictions" not in st.session_state:
        st.session_state.session_predictions = pd.DataFrame(columns=[
            "prediction_id",
            "age",
            "bmi",
            "bloodpressure",
            "children",
            "gender",
            "smoker",
            "diabetic",
            "region",
            "predicted_claim",
            "timestamp"
        ])

    age = st.number_input("Age", min_value=0, max_value=120, value=30, key="single_age")
    bmi = st.number_input("BMI", min_value=5.0, max_value=70.0, value=25.0, format="%.1f", key="single_bmi")
    bloodpressure = st.number_input("Blood Pressure", min_value=50, max_value=250, value=120, key="single_bp")
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, key="single_children")
    gender = st.selectbox("Gender", ["female", "male"], key="single_gender")
    smoker = st.selectbox("Smoker", ["no", "yes"], key="single_smoker")
    diabetic = st.selectbox("Diabetic", ["no", "yes"], key="single_diabetic")
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"], key="single_region")

    if st.button("Predict Claim", key="single_predict_btn"):
        input_df = build_single_input_df(age, bmi, bloodpressure, children, gender, smoker, diabetic, region)
        pred_claim = predict_claims_from_log_model(model, input_df)[0]

        prediction_id = str(uuid.uuid4())
        prediction_ts = datetime.now().isoformat(timespec="seconds")
        profile_record = {
            "prediction_id": prediction_id,
            "age": age,
            "bmi": bmi,
            "bloodpressure": bloodpressure,
            "children": children,
            "gender": gender,
            "smoker": smoker,
            "diabetic": diabetic,
            "region": region,
            "predicted_claim": float(pred_claim),
            "timestamp": prediction_ts
        }

        st.session_state.latest_prediction = profile_record
        st.session_state.latest_input_df = input_df.copy()
        st.session_state.sc_age = int(age)
        st.session_state.sc_bmi = float(bmi)
        st.session_state.sc_bp = int(bloodpressure)
        st.session_state.sc_children = int(children)
        st.session_state.sc_gender = str(gender)
        st.session_state.sc_smoker = str(smoker)
        st.session_state.sc_diabetic = str(diabetic)
        st.session_state.sc_region = str(region)
        new_pred_df = pd.DataFrame([profile_record])
        if st.session_state.session_predictions.empty:
            st.session_state.session_predictions = new_pred_df
        else:
            st.session_state.session_predictions = pd.concat(
                [st.session_state.session_predictions, new_pred_df],
                ignore_index=True
            )
        # Reset scenario table whenever the baseline changes
        st.session_state.scenario_comparison = pd.DataFrame(columns=[
            "scenario", "age", "bmi", "bloodpressure", "children",
            "gender", "smoker", "diabetic", "region", "predicted_claim", "delta_vs_current"
        ])

        st.success(f"💰 Predicted Insurance Claim: **${pred_claim:,.2f}**")
        st.caption(f"Prediction ID: {prediction_id}")

    if "latest_prediction" in st.session_state:

        st.markdown("### SHAP Explainability (Single Prediction)")
        try:
            shap_df = compute_single_shap_contributions(model, st.session_state.latest_input_df)
            shap_display = shap_df[["feature", "shap_value", "dollar_contribution"]].copy()
            shap_display["shap_value"] = shap_display["shap_value"].round(4)
            shap_display["dollar_contribution"] = shap_display["dollar_contribution"].apply(lambda v: f"${v:+,.2f}")
            shap_display.columns = ["Feature", "SHAP Value (log-space)", "$ Contribution to Predicted Claim"]
            st.dataframe(shap_display, width="stretch", hide_index=True)

            top_shap = shap_df.head(12).sort_values("dollar_contribution")
            top_shap["direction"] = np.where(top_shap["dollar_contribution"] >= 0, "Adds to claim", "Reduces claim")
            fig_shap = px.bar(
                top_shap,
                x="dollar_contribution",
                y="feature",
                color="direction",
                orientation="h",
                title="Feature Dollar Contributions to Predicted Claim",
                color_discrete_map={"Adds to claim": "#DC2626", "Reduces claim": "#2563EB"}
            )
            fig_shap.update_layout(
                yaxis_title="",
                xaxis_title="$ Contribution",
                xaxis_tickprefix="$",
                xaxis_separatethousands=True,
                showlegend=False
            )
            st.plotly_chart(fig_shap, width="stretch")
        except Exception as ex:
            st.warning(f"SHAP explainability is unavailable for this model pipeline: {ex}")

        st.markdown("### Scenario Simulation (Health / Lifestyle Changes)")
        base_profile = st.session_state.latest_prediction
        st.info(
            f"Baseline ({base_profile['prediction_id']}): **${base_profile['predicted_claim']:,.2f}** — "
            f"edit inputs below and click *Add Scenario* to record each variation."
        )

        if "scenario_comparison" not in st.session_state:
            st.session_state.scenario_comparison = pd.DataFrame(columns=[
                "scenario", "age", "bmi", "bloodpressure", "children",
                "gender", "smoker", "diabetic", "region", "predicted_claim", "delta_vs_current"
            ])

        # Widget values are controlled by session_state keys to avoid
        # "default value + Session State" warnings in Streamlit.
        st.session_state.setdefault("sc_age", int(base_profile["age"]))
        st.session_state.setdefault("sc_bmi", float(base_profile["bmi"]))
        st.session_state.setdefault("sc_bp", int(base_profile["bloodpressure"]))
        st.session_state.setdefault("sc_children", int(base_profile["children"]))
        st.session_state.setdefault("sc_gender", str(base_profile["gender"]))
        st.session_state.setdefault("sc_smoker", str(base_profile["smoker"]))
        st.session_state.setdefault("sc_diabetic", str(base_profile["diabetic"]))
        st.session_state.setdefault("sc_region", str(base_profile["region"]))

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            s_age = st.number_input("Age", min_value=0, max_value=120, key="sc_age")
            s_bmi = st.number_input("BMI", min_value=5.0, max_value=70.0, format="%.1f", key="sc_bmi")
        with sc2:
            s_bp = st.number_input("Blood Pressure", min_value=50, max_value=250, key="sc_bp")
            s_children = st.number_input("Children", min_value=0, max_value=10, key="sc_children")
        with sc3:
            s_gender = st.selectbox("Gender", ["female", "male"], key="sc_gender")
            s_smoker = st.selectbox("Smoker", ["no", "yes"], key="sc_smoker")
        with sc4:
            s_diabetic = st.selectbox("Diabetic", ["no", "yes"], key="sc_diabetic")
            s_region = st.selectbox(
                "Region", ["northeast", "northwest", "southeast", "southwest"],
                key="sc_region"
            )

        btn_col, clear_col = st.columns([2, 1])
        with btn_col:
            add_scenario = st.button("➕ Add Scenario to Comparison", key="single_add_scenario")
        with clear_col:
            clear_scenarios = st.button("🗑 Clear All Scenarios", key="single_clear_scenarios")

        if add_scenario:
            sc_input_df = build_single_input_df(s_age, s_bmi, s_bp, s_children, s_gender, s_smoker, s_diabetic, s_region)
            sc_pred = predict_claims_from_log_model(model, sc_input_df)[0]
            sc_num = len(st.session_state.scenario_comparison) + 1
            new_row = {
                "scenario": f"Scenario {sc_num}",
                "age": s_age, "bmi": s_bmi, "bloodpressure": s_bp, "children": s_children,
                "gender": s_gender, "smoker": s_smoker, "diabetic": s_diabetic, "region": s_region,
                "predicted_claim": float(sc_pred),
                "delta_vs_current": float(sc_pred) - float(base_profile["predicted_claim"])
            }
            new_sc_df = pd.DataFrame([new_row])
            if st.session_state.scenario_comparison.empty:
                st.session_state.scenario_comparison = new_sc_df
            else:
                st.session_state.scenario_comparison = pd.concat(
                    [st.session_state.scenario_comparison, new_sc_df],
                    ignore_index=True
                )
            delta = float(sc_pred) - float(base_profile["predicted_claim"])
            st.success(f"Scenario {sc_num} added — Predicted: **${sc_pred:,.2f}** (Δ ${delta:+,.2f})")

        if clear_scenarios:
            st.session_state.scenario_comparison = pd.DataFrame(columns=[
                "scenario", "age", "bmi", "bloodpressure", "children",
                "gender", "smoker", "diabetic", "region", "predicted_claim", "delta_vs_current"
            ])

        if not st.session_state.scenario_comparison.empty:
            st.markdown("#### Scenario Comparison Table")
            baseline_for_display = {
                "scenario": "Baseline",
                "age": base_profile["age"], "bmi": base_profile["bmi"],
                "bloodpressure": base_profile["bloodpressure"], "children": base_profile["children"],
                "gender": base_profile["gender"], "smoker": base_profile["smoker"],
                "diabetic": base_profile["diabetic"], "region": base_profile["region"],
                "predicted_claim": float(base_profile["predicted_claim"]),
                "delta_vs_current": 0.0
            }
            display_comparison = pd.concat(
                [pd.DataFrame([baseline_for_display]), st.session_state.scenario_comparison],
                ignore_index=True
            )
            st.dataframe(display_comparison, width="stretch", hide_index=True)
            st.download_button(
                "📥 Download Scenario Comparison (CSV)",
                data=display_comparison.to_csv(index=False).encode("utf-8"),
                file_name=f"scenario_comparison_{base_profile['prediction_id']}.csv",
                mime="text/csv",
                key="single_scenario_download"
            )

    st.markdown("---")
    st.markdown("### Session Prediction History")
    st.dataframe(st.session_state.session_predictions, width="stretch", hide_index=True)
    st.download_button(
        "📥 Download Session Predictions (CSV)",
        data=st.session_state.session_predictions.to_csv(index=False).encode("utf-8"),
        file_name="session_predictions.csv",
        mime="text/csv",
        key="single_session_download"
    )

with tab_batch:
    st.subheader("Batch Prediction, Filters, and KPI Dashboard")
    st.info("Upload a CSV with columns: age, bmi, bloodpressure, children, gender, smoker, diabetic, region")
    uploaded_file = st.file_uploader("Upload CSV for batch analytics", type=["csv"], key="batch_upload")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        high_claim_boost = st.slider(
            "High-Claim Boost",
            min_value=0.90,
            max_value=1.50,
            value=1.10,
            step=0.01,
            help="Applies additional scaling to top predicted claims when actual claims are unavailable.",
            key="batch_boost"
        )

        required_cols = ["age", "bmi", "bloodpressure", "children", "gender", "smoker", "diabetic", "region"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            if "Predicted_Claims" not in df.columns:
                df["Predicted_Claims"] = predict_claims_from_log_model(model, df[required_cols])

            pred_col = "Predicted_Claims"
            if "claim" in df.columns:
                tail_threshold = df[pred_col].quantile(0.95)
                tail_mask = df[pred_col] >= tail_threshold
                if tail_mask.sum() >= 5:
                    tail_ratio = np.quantile(
                        df.loc[tail_mask, "claim"] / (df.loc[tail_mask, pred_col] + 1e-9),
                        0.75
                    )
                    tail_ratio = float(np.clip(tail_ratio, 0.90, 2.00))
                    df["Predicted_Claims_Adjusted"] = df[pred_col]
                    df.loc[tail_mask, "Predicted_Claims_Adjusted"] = df.loc[tail_mask, pred_col] * tail_ratio

                    extreme_mask = df[pred_col] >= df[pred_col].quantile(0.99)
                    if extreme_mask.sum() >= 1:
                        extreme_ratio = np.quantile(
                            df.loc[extreme_mask, "claim"] / (df.loc[extreme_mask, pred_col] + 1e-9),
                            0.80
                        )
                        extreme_ratio = float(np.clip(extreme_ratio, 1.00, 2.20))
                        df.loc[extreme_mask, "Predicted_Claims_Adjusted"] = (
                            df.loc[extreme_mask, "Predicted_Claims_Adjusted"] * extreme_ratio
                        )

                    pred_col = "Predicted_Claims_Adjusted"
                    st.caption(f"Applied tail calibration: top 5% x{tail_ratio:.2f} and top 1% extra uplift.")
            else:
                global_tail_q = float(getattr(model, "global_tail_quantile_", 0.95))
                global_tail_m = float(getattr(model, "global_tail_multiplier_", 1.18))
                global_tail_q = float(np.clip(global_tail_q, 0.85, 0.99))
                global_tail_m = float(np.clip(global_tail_m, 1.00, 2.00))
                adjusted = apply_tail_adjustment(
                    df[pred_col].to_numpy(),
                    quantile=global_tail_q,
                    multiplier=global_tail_m,
                )

                extreme_q = float(getattr(model, "global_extreme_quantile_", 0.99))
                extreme_m = float(getattr(model, "global_extreme_multiplier_", 1.25))
                extreme_q = float(np.clip(extreme_q, 0.95, 0.999))
                extreme_m = float(np.clip(extreme_m, 1.00, 2.20))
                adjusted = apply_tail_adjustment(adjusted, quantile=extreme_q, multiplier=extreme_m)
                adjusted = apply_tail_adjustment(adjusted, quantile=0.99, multiplier=high_claim_boost)

                df["Predicted_Claims_Adjusted"] = adjusted
                pred_col = "Predicted_Claims_Adjusted"
                st.caption(f"Applied global tail calibration + High-Claim Boost x{high_claim_boost:.2f}.")

            df["smoker"] = df["smoker"].astype(str).str.lower().str.strip()
            df["high_bmi"] = (df["bmi"] >= 25).astype(int)
            df["high_bp"] = (df["bloodpressure"] >= 130).astype(int)
            df["risk_flag"] = np.where(
                ((df["smoker"] == "yes") & (df["high_bmi"] == 1))
                | ((df["smoker"] == "yes") & (df["high_bp"] == 1))
                | ((df["high_bmi"] == 1) & (df["high_bp"] == 1)),
                1,
                0,
            )

            filt1, filt2, filt3 = st.columns(3)
            with filt1:
                region_filter = st.multiselect(
                    "Region Filter",
                    options=sorted(df["region"].astype(str).unique()),
                    default=sorted(df["region"].astype(str).unique()),
                    key="batch_region_filter"
                )
            with filt2:
                smoker_filter = st.multiselect(
                    "Smoker Filter",
                    options=sorted(df["smoker"].astype(str).unique()),
                    default=sorted(df["smoker"].astype(str).unique()),
                    key="batch_smoker_filter"
                )
            with filt3:
                gender_filter = st.multiselect(
                    "Gender Filter",
                    options=sorted(df["gender"].astype(str).unique()),
                    default=sorted(df["gender"].astype(str).unique()),
                    key="batch_gender_filter"
                )

            range1, range2 = st.columns(2)
            with range1:
                age_range = st.slider(
                    "Age Range",
                    int(df["age"].min()),
                    int(df["age"].max()),
                    (int(df["age"].min()), int(df["age"].max())),
                    key="batch_age_range"
                )
            with range2:
                bmi_range = st.slider(
                    "BMI Range",
                    float(df["bmi"].min()),
                    float(df["bmi"].max()),
                    (float(df["bmi"].min()), float(df["bmi"].max())),
                    key="batch_bmi_range"
                )

            filtered_df = df[
                df["region"].astype(str).isin(region_filter)
                & df["smoker"].astype(str).isin(smoker_filter)
                & df["gender"].astype(str).isin(gender_filter)
                & df["age"].between(age_range[0], age_range[1])
                & df["bmi"].between(bmi_range[0], bmi_range[1])
            ]

            st.markdown(f"**Filtered Records:** {filtered_df.shape[0]} rows")

            total_claims = filtered_df[pred_col].sum()
            avg_claims = filtered_df[pred_col].mean()
            max_claim = filtered_df[pred_col].max()
            high_risk_claims = filtered_df.loc[filtered_df["risk_flag"] == 1, pred_col].sum()
            high_risk_pct = (high_risk_claims / total_claims * 100) if total_claims > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            if "claim" in filtered_df.columns:
                actual_total_claims = filtered_df["claim"].sum()
                actual_avg_claims = filtered_df["claim"].mean()
                total_delta = total_claims - actual_total_claims
                avg_delta = avg_claims - actual_avg_claims
                col1.metric("💰 Pred Total Claims", f"${total_claims:,.0f}", delta=f"${total_delta:,.0f} vs actual")
                col2.metric("📈 Pred Average Claim", f"${avg_claims:,.2f}", delta=f"${avg_delta:,.2f} vs actual")
            else:
                col1.metric("💰 Total Claims", f"${total_claims:,.0f}")
                col2.metric("📈 Average Claim", f"${avg_claims:,.2f}")

            col3.metric("⚠️ High-Risk Claims (%)", f"{high_risk_pct:.1f}%")
            if "claim" in filtered_df.columns:
                actual_max_claim = filtered_df["claim"].max()
                max_delta = max_claim - actual_max_claim
                col4.metric("🔝 Pred Max Claim", f"${max_claim:,.2f}", delta=f"${max_delta:,.2f} vs actual")
                st.caption(f"Actual total: ${actual_total_claims:,.0f} | Actual max: ${actual_max_claim:,.0f}")
            else:
                col4.metric("🔝 Max Claim", f"${max_claim:,.2f}")

            st.markdown("---")
            st.subheader("📊 Claims Analysis Dashboard")
            vis1, vis2 = st.columns(2)

            with vis1:
                age_claim = filtered_df.groupby("age")[pred_col].mean().reset_index()
                fig_age = px.line(
                    age_claim,
                    x="age",
                    y=pred_col,
                    markers=True,
                    title="Average Claim by Age",
                    labels={pred_col: "Predicted Claim"}
                )
                st.plotly_chart(fig_age, width="stretch")

                region_claim = filtered_df.groupby("region")[pred_col].mean().reset_index()
                fig_region = px.bar(
                    region_claim,
                    x="region",
                    y=pred_col,
                    color="region",
                    title="Average Claim by Region",
                    labels={pred_col: "Predicted Claim"}
                )
                st.plotly_chart(fig_region, width="stretch")

            with vis2:
                fig_bmi = px.scatter(
                    filtered_df,
                    x="bmi",
                    y=pred_col,
                    color="smoker",
                    title="BMI vs Claim by Smoker Status",
                    labels={pred_col: "Predicted Claim"}
                )
                st.plotly_chart(fig_bmi, width="stretch")

                smoker_counts = filtered_df["smoker"].value_counts()
                fig_smoker = px.pie(
                    values=smoker_counts.values,
                    names=smoker_counts.index,
                    title="Smoker Distribution",
                    hole=0.45,
                )
                st.plotly_chart(fig_smoker, width="stretch")

            heatmap_data = filtered_df.pivot_table(index="region", columns="smoker", values=pred_col, aggfunc="mean")
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale="Blues",
                colorbar_title="Avg Claims"
            ))
            fig_heatmap.update_layout(title="Average Claim by Region & Smoker")
            st.plotly_chart(fig_heatmap, width="stretch")

            csv = filtered_df.to_csv(index=False).encode()
            st.download_button("📥 Download Predictions CSV", csv, "predicted_claims.csv", "text/csv", key="batch_download")

with tab_fraud:
    st.subheader("Insurance Fraud Indicators")
    st.warning("Upload only CSV files that include actual claim values in a 'claim' column for fraud analytics.")

    fraud_file = st.file_uploader("Upload CSV for fraud analytics", type=["csv"], key="fraud_upload")

    if fraud_file is not None:
        fraud_df = pd.read_csv(fraud_file)
        required_cols = ["age", "bmi", "bloodpressure", "children", "gender", "smoker", "diabetic", "region"]
        missing_cols = [c for c in required_cols if c not in fraud_df.columns]

        if missing_cols:
            st.error(f"Missing required feature columns: {missing_cols}")
        elif "claim" not in fraud_df.columns:
            st.info("Fraud detection requires actual claim values.")
        else:
            if "Predicted_Claims" not in fraud_df.columns:
                fraud_df["Predicted_Claims"] = predict_claims_from_log_model(model, fraud_df[required_cols])

            fraud_df["claim"] = pd.to_numeric(fraud_df["claim"], errors="coerce")
            fraud_df["Predicted_Claims"] = pd.to_numeric(fraud_df["Predicted_Claims"], errors="coerce")
            fraud_df = fraud_df.dropna(subset=["claim", "Predicted_Claims"]).copy()

            fraud_df["fraud_ratio"] = fraud_df["claim"] / (fraud_df["Predicted_Claims"] + 1e-9)
            fraud_df["fraud_gap"] = fraud_df["claim"] - fraud_df["Predicted_Claims"]
            fraud_df["fraud_flag"] = fraud_df["claim"] > (3 * fraud_df["Predicted_Claims"])
            fraud_df["suspicious_fraud_gap"] = np.where(
                fraud_df["fraud_flag"],
                fraud_df["fraud_gap"].clip(lower=0),
                0,
            )
            fraud_df["fraud_level"] = np.select(
                [
                    fraud_df["fraud_ratio"] > 3,
                    fraud_df["fraud_ratio"].between(2, 3, inclusive="right"),
                ],
                ["High", "Medium"],
                default="Low",
            )

            k1, k2, k3 = st.columns(3)
            suspicious_count = int(fraud_df["fraud_flag"].sum())
            suspicious_rate = (suspicious_count / len(fraud_df) * 100) if len(fraud_df) else 0
            k1.metric("🚨 Suspicious Claims", f"{suspicious_count}")
            k2.metric("📌 Suspicious Rate", f"{suspicious_rate:.1f}%")
            k3.metric("💸 Total Excess Claim Gap", f"${fraud_df['fraud_gap'].clip(lower=0).sum():,.0f}")

            st.markdown("### Suspicious Claims by Region")
            region_risk = (
                fraud_df.groupby("region", as_index=False)
                .agg(
                    suspicious_claims=("fraud_flag", "sum"),
                    total_fraud_gap=("suspicious_fraud_gap", "sum"),
                )
                .sort_values("total_fraud_gap", ascending=False)
            )
            fig_region_risk = make_subplots(specs=[[{"secondary_y": True}]])
            fig_region_risk.add_trace(
                go.Bar(
                    x=region_risk["region"],
                    y=region_risk["suspicious_claims"],
                    name="Suspicious Count",
                    marker=dict(color="#93C5FD", line=dict(color="#1D4ED8", width=1)),
                    opacity=0.65,
                    text=region_risk["suspicious_claims"],
                    textposition="outside",
                    hovertemplate="Region: %{x}<br>Suspicious Count: %{y}<extra></extra>",
                ),
                secondary_y=False,
            )
            fig_region_risk.add_trace(
                go.Scatter(
                    x=region_risk["region"],
                    y=region_risk["total_fraud_gap"],
                    name="Total Fraud Gap ($)",
                    mode="lines+markers",
                    line=dict(color="#F97316", width=3),
                    marker=dict(size=10, color="#F97316"),
                    hovertemplate="Region: %{x}<br>Total Fraud Gap: $%{y:,.0f}<extra></extra>",
                ),
                secondary_y=True,
            )
            fig_region_risk.update_layout(
                title="Suspicious Claims by Region: Count vs Total Fraud Gap",
                template="plotly_white",
                legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
                hovermode="x unified",
                bargap=0.4,
                margin=dict(t=90, l=40, r=40, b=40),
            )
            fig_region_risk.update_yaxes(title_text="Suspicious Claim Count", secondary_y=False, rangemode="tozero")
            fig_region_risk.update_yaxes(
                title_text="Total Fraud Gap ($)",
                secondary_y=True,
                rangemode="tozero",
                tickprefix="$",
                separatethousands=True,
            )
            st.plotly_chart(fig_region_risk, width="stretch")

            fig = px.scatter(
                fraud_df,
                x="Predicted_Claims",
                y="claim",
                color="fraud_level",
                title="Predicted vs Actual Claims",
                hover_data=["fraud_ratio", "fraud_gap", "region"],
            )
            st.plotly_chart(fig, width="stretch")

            st.markdown("### Top Suspicious Claims Table")
            suspicious_df = fraud_df[fraud_df["fraud_flag"]].sort_values("fraud_ratio", ascending=False).head(20)
            if suspicious_df.empty:
                st.info("No claims met the rule: Actual Claim > 3 × Predicted Claim.")
            else:
                suspicious_export_df = fraud_df[fraud_df["fraud_flag"]].sort_values("fraud_ratio", ascending=False).copy()
                top_cols = [
                    c for c in ["PatientID", "region", "claim", "Predicted_Claims", "fraud_ratio", "fraud_gap", "fraud_level"]
                    if c in suspicious_df.columns
                ]
                st.dataframe(suspicious_df[top_cols], width="stretch", hide_index=True)
                suspicious_csv = suspicious_export_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Suspicious Claims (CSV)",
                    data=suspicious_csv,
                    file_name="suspicious_claims.csv",
                    mime="text/csv",
                    key="fraud_export_suspicious"
                )

            st.markdown("### Fraud Investigation Table")
            review_df = fraud_df[fraud_df["fraud_level"].isin(["High", "Medium"])].sort_values(
                ["fraud_level", "fraud_ratio"],
                ascending=[True, False]
            )
            review_cols = [
                c for c in [
                    "PatientID", "age", "bmi", "bloodpressure", "children", "gender", "smoker", "diabetic", "region",
                    "Predicted_Claims", "claim", "fraud_ratio", "fraud_gap", "fraud_level"
                ] if c in review_df.columns
            ]
            st.dataframe(review_df[review_cols], width="stretch", hide_index=True)
            review_csv = review_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Fraud Investigation Table (CSV)",
                data=review_csv,
                file_name="fraud_investigation_table.csv",
                mime="text/csv",
                key="fraud_export_investigation"
            )