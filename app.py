import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from knn_base import generate_and_train, generate_prediction_df, drop_highly_correlated

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("🩺 Health Analytics Dashboard")

st.sidebar.header("📊 Generate Dataset")
n_samples = st.sidebar.slider("Dataset Size", 0, 500, 284, 1)
n_features = st.sidebar.slider("Features", 0, 1000, 644, 1)
corr_threshold = st.sidebar.slider("Drop Corr >", 0.00, 1.00, 0.70, 0.01)

st.sidebar.header("🔧 Model Settings")
test_size = st.sidebar.slider("Test Size", 0.0, 1.0, 0.25, 0.01)
random_state = st.sidebar.slider("Random Seed", 0, 100, 42, 1)
n_neighbors = st.sidebar.slider("KNN Neighbors", 1, 25, 5, 1)
weights = st.sidebar.radio("Weights", ["uniform", "distance"])
metric = st.sidebar.radio("Metric", ["manhattan", "euclidean"])

if st.sidebar.button("Generate Data & Train Model", type="primary"):
    with st.spinner(f"Generating {n_samples} samples + model training..."):
        results = generate_and_train(
            n_samples=n_samples,
            n_features=n_features,
            test_size=test_size,
            random_state=random_state,
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            corr_threshold=corr_threshold
        )
        st.session_state.results = results
    st.sidebar.success("✅ Ready!")

if "results" in st.session_state:
    res = st.session_state["results"]

    with st.expander("Model Plots", expanded=False):
        if "train_scatter_bytes" in res:
            st.image(res["train_scatter_bytes"], caption="Train Predictions", width='stretch')
        if "test_scatter_bytes" in res:
            st.image(res["test_scatter_bytes"], caption="Test Predictions", width='stretch')
        if "train_residuals_bytes" in res:
            st.image(res["train_residuals_bytes"], caption="Train Residuals", width='stretch')
        if "test_residuals_bytes" in res:
            st.image(res["test_residuals_bytes"], caption="Test Residuals", width='stretch')
        if "train_distribution_bytes" in res:
            st.image(res["train_distribution_bytes"], caption="Train Distribution", width='stretch')
        if "test_distribution_bytes" in res:
            st.image(res["test_distribution_bytes"], caption="Test Distribution", width='stretch')
        if "feature_importance_bytes" in res:
            st.subheader("Feature Importance")
            st.image(res["feature_importance_bytes"], caption="Feature Importance", width='stretch')
        if "learning_curve_bytes" in res:
            st.image(res["learning_curve_bytes"], caption="Learning Curve", width='stretch')
        if "validation_curve_bytes" in res:
            st.image(res["validation_curve_bytes"], caption="Validation Curve", width='stretch')
        else:
            st.warning("Train plots not available.")
else:
    st.info("No generated model plots. Click 'Generate Data & Train Model'.")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("📋 Data Explorer")
    if 'results' in st.session_state:
        df = st.session_state.results['df']
        
        id_cols_region = ['State_Name', 'District_Name']
        default_cols = id_cols_region + df.columns[2:6].tolist() 
        feat_cols = st.multiselect(
            "Features to show",  
            df.columns.drop('Target_IMR'), 
            default=[col for col in default_cols if col in df.columns and col != 'Target_IMR']
        )
        target = 'Target_IMR'  
        filtered_df = df[feat_cols + [target]] if target not in feat_cols else df[feat_cols]

        st.dataframe(filtered_df, width='stretch')
        
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1: st.metric("Rows", len(df))
        with col_stats2: st.metric("Features", len(df.columns)-1)
        with col_stats3: st.metric("IMR Range", f"{df['Target_IMR'].min():.0f}-{df['Target_IMR'].max():.0f}")

with col2:
    st.header("🎯 Real-time Predictions")
    if 'results' in st.session_state:
        results = st.session_state.results
        knn, preprocessor, selector, feats = (results[k] for k in ['knn','preprocessor','selector','selected_features'])
        
        st.metric("Test R²", f"{results['test_metrics']['R2']:.4f}")
        st.metric("Test RMSE", f"{results['test_metrics']['RMSE']:.2f}")
        st.metric("Test MAE", f"{results['test_metrics']['MAE']:.2f}")

        st.subheader("Input Features")
        input_data = {}
        for feat in feats[:5]: 
            minv, maxv = df[feat].min(), df[feat].max()
            input_data[feat] = st.slider(f"{feat[:25]}...", minv, maxv, (minv+maxv)/2)
        
        if st.button("⚕ Predict IMR", type="secondary"):
            input_df = generate_prediction_df(input_data, df.columns[:-2].tolist(), ['State_Name', 'District_Name'], n_features)
            input_proc = selector.transform(preprocessor.transform(input_df))
            pred = knn.predict(input_proc)[0]
            st.success(f"**Predicted IMR: {pred:.0f}**")
            st.caption(f"Approximation Range: {df['Target_IMR'].min():.0f}-{df['Target_IMR'].max():.0f} (per 1000 live births)")
    else:
        st.info("👈 Generate data & model first!")

if 'results' in st.session_state:
    st.header("📊 Model Diagnostics")
    results = st.session_state.results
    
    col1_diag, col2_diag = st.columns(2)
    with col1_diag:
        st.write("**Top Selected Features**")
        st.write(results['selected_features'][:10])
    
    with col2_diag:
        dropped_count = len(results.get('corr_dropped', []))
        st.metric("Corr Dropped", dropped_count)
        st.caption(f"Threshhold: {results.get('corr_threshold', 0.85):.2f}")
        if dropped_count > 0:
            st.caption("📊 " + ", ".join(results['corr_dropped'][:3]))
        st.write("**Test Metrics**")
        metrics_df = pd.DataFrame([results['test_metrics']])
        st.dataframe(metrics_df.T, width='stretch')

st.markdown("---")
st.caption("For simulation and educational purposes, this model utilizes synthetic data.")
