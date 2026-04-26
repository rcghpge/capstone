# MIT License
# See LICENSE file in the project root or at https://opensource.org/license/mit
#
# Copyright (c) 2026 Robert Cocker
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
os.environ.setdefault("KERAS_BACKEND", "jax")
import keras
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from urllib.parse import urlencode
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from statsmodels.tools.tools import add_constant
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

st.set_page_config(page_title="Dashboard", layout="wide")

APP_TITLE = "📊 Health Analytics Dashboard"
st.title(APP_TITLE)
st.caption("Machine Learning Utilizing Key Health Indicators For Health Analytics • Preview")
st.subheader("Google Search")

with st.form("google_search_form", clear_on_submit=False, enter_to_submit=True):
    search_query = st.text_input(
        "Search Google",
        key="google_search_box",
        placeholder="Type a Google search query and press Enter"
    )
    submitted = st.form_submit_button("Run Google Search")

if submitted:
    query = search_query.strip()

    if not query:
        st.warning("Please enter a search query.")
    else:
        google_url = "https://www.google.com/search?" + urlencode({"q": query})

        st.markdown("### Search Preview")
        st.info(f"Query: {query}")
        st.code(google_url, language="text")

        with st.container(border=True):
            st.markdown(f"**Search term:** {query}")
            st.markdown("**Preview:** Open Google search results using the link below.")
            st.markdown(f"[Open Google Results in New Tab]({google_url})")
        
def build_dnn(input_dim: int):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
    	keras.layers.BatchNormalization(),
        keras.layers.Dense(512, activation='swish'),
    	keras.layers.Dropout(0.15),
    	keras.layers.BatchNormalization(),
    	keras.layers.Dense(256, activation='swish'),
    	keras.layers.Dropout(0.15),
    	keras.layers.BatchNormalization(),
        keras.layers.Dense(128, activation='swish'),
        keras.layers.Dropout(0.15),
    	keras.layers.BatchNormalization(),
        keras.layers.Dense(64, activation='swish'),
        keras.layers.Dropout(0.15),
    	keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation='swish'),
    	keras.layers.Dropout(0.075),
        keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.0009), loss="huber", metrics=["mae"])
    return model

def check_vif(df_num: pd.DataFrame):
    X = df_num.copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=0, how="any")
    X = X.loc[:, X.nunique(dropna=True) > 1]
    if X.shape[1] < 2 or len(X) < 5:
        return None
    try:
        X_const = add_constant(X, has_constant="add")
        vif_df = pd.DataFrame({
            "feature": X_const.columns,
            "VIF": [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])],
        })
        return vif_df.sort_values("VIF", ascending=False)
    except Exception:
        return None

def set_feature_bins(series: pd.Series):
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if non_null.nunique() >= 4 and len(non_null) >= 4:
            ranked = series.rank(method="first")
            try:
                return pd.qcut(ranked, q=4, duplicates="drop").astype(str)
            except Exception:
                return series.round(2).astype(str)
        return series.round(2).astype(str)
    top_counts = series.astype(str).value_counts()
    if len(top_counts) > 12:
        top_labels = set(top_counts.head(11).index.tolist())
        return series.astype(str).apply(lambda x: x if x in top_labels else "Other")
    return series.astype(str)

def plot_feature_dynamics(df, feature, y_col, color_col=None, title_suffix=""):
    s = df[feature]

    if pd.api.types.is_datetime64_any_dtype(s):
        plot_df = df.sort_values(feature).copy()
        return px.scatter(
            plot_df,
            x=feature,
            y=y_col,
            color=color_col,
            title=f"{feature} vs {y_col}{title_suffix}",
        )

    if pd.api.types.is_numeric_dtype(s):
        return px.scatter(
            df,
            x=feature,
            y=y_col,
            color=color_col,
            trendline="lowess",
            title=f"{feature} vs {y_col}{title_suffix}",
        )

    plot_df = df.copy()
    plot_df[feature] = plot_df[feature].astype(str)
    return px.box(
        plot_df,
        x=feature,
        y=y_col,
        color=feature if color_col is None else color_col,
        points="all",
        title=f"{feature} vs {y_col}{title_suffix}",
    )

model_options = ["JAX/Keras DNN", "RandomForest", "KNN Regression"]
if HAS_XGBOOST:
    model_options.insert(2, "XGBoost")

st.header("Upload A Dataset")
uploaded_file = st.file_uploader("Upload Dataset", type="csv")

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df_uploaded.head(10), width='stretch')

    if df_uploaded.shape[1] < 2:
        st.error("Dataset must contain at least one feature column and one target column.")
        st.stop()

    target_col = st.selectbox("Select Target Feature Column To Predict", options=df_uploaded.columns.tolist())
    feature_cols = st.multiselect(
        "Select Feature Columns",
        options=[c for c in df_uploaded.columns if c != target_col],
        default=[c for c in df_uploaded.columns if c != target_col],
    )
    model_family = st.selectbox("Model Family", model_options, index=model_options.index("JAX/Keras DNN"))
    test_size = st.slider("Test Set Size", min_value=0.00, max_value=1.0, value=0.25, step=0.05)
    random_state = st.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)

    if model_family == "KNN Regression":
        knn_neighbors = st.slider("KNN Number of Neighbors", min_value=2, max_value=25, value=5, step=1)
    else:
        knn_neighbors = 5

    if model_family == "XGBoost" and not HAS_XGBOOST:
        st.warning("XGBoost is not installed in this environment, so that option is unavailable.")

    if st.button("Run Inference On a Model", type="primary"):
        if not feature_cols:
            st.error("Please Select at Least 1 Feature Column.")
            st.stop()

        model_df = df_uploaded[feature_cols + [target_col]].copy()
        model_df = model_df.dropna(subset=[target_col])

        y = pd.to_numeric(model_df[target_col], errors="coerce")
        valid_mask = y.notna()
        X = model_df.loc[valid_mask, feature_cols].copy()
        y = y.loc[valid_mask].copy()

        if len(X) < 20:
            st.error("Not enough rows in the data after cleaning the target column. You need at least 20 rows.")
            st.stop()

        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=int(random_state),
        )

        X_train_p = preprocessor.fit_transform(X_train)
        X_test_p = preprocessor.transform(X_test)

        if model_family == "RandomForest":
            model = RandomForestRegressor(n_estimators=300, random_state=int(random_state), n_jobs=-1)
            model.fit(X_train_p, y_train)
            preds = model.predict(X_test_p)
        elif model_family == "XGBoost" and HAS_XGBOOST:
            model = XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=int(random_state),
                n_jobs=4,
            )
            model.fit(X_train_p, y_train)
            preds = model.predict(X_test_p)
        elif model_family == "KNN Regression":
            model = KNeighborsRegressor(n_neighbors=knn_neighbors, weights='uniform', metric='manhattan')
            model.fit(X_train_p, y_train)
            preds = model.predict(X_test_p)
        else:
            scaler_y = StandardScaler()
            y_train_s = scaler_y.fit_transform(np.asarray(y_train).reshape(-1, 1)).ravel()
            model = build_dnn(X_train_p.shape[1])
            early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
            model.fit(
                X_train_p,
                y_train_s,
                validation_split=0.10,
                epochs=500,
                batch_size=32,
                verbose=0,
                callbacks=[early],
            )
            preds = scaler_y.inverse_transform(model.predict(X_test_p, verbose=0).reshape(-1, 1)).ravel()

        r2_val = r2_score(y_test, preds)
        rmse_val = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae_val = mean_absolute_error(y_test, preds)

        results_df = pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": preds}, index=y_test.index)
        results_df["Residuals"] = results_df["Predicted"] - results_df["Actual"]
        results_df["Abs_Error"] = results_df["Residuals"].abs()

        feature_plot_df = X_test.copy()
        feature_plot_df["Actual"] = y_test.to_numpy()
        feature_plot_df["Predicted"] = preds
        feature_plot_df["Residuals"] = results_df["Residuals"].to_numpy()
        feature_plot_df["Abs_Error"] = results_df["Abs_Error"].to_numpy()

        st.session_state["results_df"] = results_df
        st.session_state["feature_plot_df"] = feature_plot_df
        st.session_state["feature_cols"] = feature_cols
        st.session_state["target_col"] = target_col
        st.session_state["metrics_summary"] = {
            "r2": r2_val,
            "rmse": rmse_val,
            "mae": mae_val,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "feature_count": len(feature_cols),
            "model_family": model_family,
        }
        st.session_state["numeric_train_df"] = X_train[numeric_cols].copy() if numeric_cols else pd.DataFrame()

if st.session_state.get("results_df") is not None:
    results_df = st.session_state["results_df"]
    feature_plot_df = st.session_state["feature_plot_df"]
    feature_cols = st.session_state["feature_cols"]
    target_col = st.session_state["target_col"]
    metrics = st.session_state["metrics_summary"]
    numeric_train_df = st.session_state["numeric_train_df"]

    st.subheader("Model Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", metrics["model_family"])
    c2.metric("Test R²", f"{metrics['r2']:.3f}")
    c3.metric("Test RMSE", f"{metrics['rmse']:.3f}")
    c4.metric("Test MAE", f"{metrics['mae']:.3f}")

    st.header("📋 Interactive Data Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "Features", "Performance", "VIF"])

    with tab1:
        a, b = st.columns(2)
        with a:
            fig_pred = px.scatter(
                results_df,
                x="Actual",
                y="Predicted",
                color="Abs_Error",
                title=f"Predictions vs Actual: {target_col}",
                color_continuous_scale=[[0, 'coral'], [0.5, 'white'], [1, 'steelblue']],  
            )
            minv = float(min(results_df["Actual"].min(), results_df["Predicted"].min()))
            maxv = float(max(results_df["Actual"].max(), results_df["Predicted"].max()))
            fig_pred.add_shape(type="line", x0=minv, y0=minv, x1=maxv, y1=maxv, line=dict(color="orange", dash="dash"))
            st.plotly_chart(fig_pred, width='stretch')
        with b:
            fig_resid = px.scatter(
                results_df,
                x="Predicted",
                y="Residuals",
                color="Abs_Error",
                title="Residuals Plot",
                color_continuous_scale=[[0, 'coral'], [0.5, 'white'], [1, 'steelblue']],
            )
            fig_resid.add_hline(y=0, line_dash="dash", line_color="orange")
            st.plotly_chart(fig_resid, width='stretch')
        st.subheader("Prediction Sample")
        st.dataframe(results_df.head(20), width='stretch')

    with tab2:
        selected_feature = st.selectbox(
            "Feature To Visualize",
            options=feature_cols,
            key="feature_plot_selector"
        )

        feature_series = feature_plot_df[selected_feature]
        is_numeric = pd.api.types.is_numeric_dtype(feature_series)
        is_datetime = pd.api.types.is_datetime64_any_dtype(feature_series)
        plot_df = feature_plot_df.copy()

        if is_datetime:
            plot_df[selected_feature] = pd.to_datetime(plot_df[selected_feature], errors="coerce")
            plot_df = plot_df.dropna(subset=[selected_feature]).sort_values(selected_feature)
        elif not is_numeric:
            plot_df[selected_feature] = plot_df[selected_feature].astype(str).fillna("Missing")

        c, d = st.columns(2)

        with c:
            if is_numeric or is_datetime:
                fig_feat = px.scatter(
                    plot_df,
                    x=selected_feature,
                    y="Predicted",
                    color="Actual",
                    trendline="lowess" if is_numeric else None,
                    title=f"{selected_feature} vs Predicted {target_col}",
                )
            else:
                fig_feat = px.box(
                    plot_df,
                    x=selected_feature,
                    y="Predicted",
                    color=selected_feature,
                    points="all",
                    title=f"{selected_feature} vs Predicted {target_col}",
                )
            st.plotly_chart(fig_feat, width='stretch')

        with d:
            if is_numeric:
                fig_hist = px.histogram(
                    plot_df,
                    x=selected_feature,
                    nbins=10,
                    title=f"{selected_feature} Distribution",
                )
            else:
                fig_hist = px.histogram(
                    plot_df,
                    x=selected_feature,
                    title=f"{selected_feature} Distribution",
                )
                fig_hist.update_xaxes(type="category")
            st.plotly_chart(fig_hist, width='stretch')

        e, f = st.columns(2)

        with e:
            if is_numeric or is_datetime:
                fig_actual = px.scatter(
                    plot_df,
                    x=selected_feature,
                    y="Actual",
                    color="Predicted",
                    trendline="lowess" if is_numeric else None,
                    title=f"{selected_feature} vs Actual {target_col}",
                )
            else:
                fig_actual = px.box(
                    plot_df,
                    x=selected_feature,
                    y="Actual",
                    color=selected_feature,
                    points="all",
                    title=f"{selected_feature} vs Actual {target_col}",
                )
            st.plotly_chart(fig_actual, width='stretch')

        with f:
            error_df = plot_df.copy()
            error_df["feature_bin"] = set_feature_bins(error_df[selected_feature])
            fig_error = px.box(
                error_df,
                x="feature_bin",
                y="Abs_Error",
                title=f"Absolute Error by {selected_feature}",
            )
            fig_error.update_xaxes(type="category")
            st.plotly_chart(fig_error, width='stretch')

    with tab3:
        perf_left, perf_right = st.columns(2)
        with perf_left:
            summary_df = pd.DataFrame([
                {"Metric": "Model", "Value": metrics["model_family"]},
                {"Metric": "R²", "Value": metrics["r2"]},
                {"Metric": "RMSE", "Value": metrics["rmse"]},
                {"Metric": "MAE", "Value": metrics["mae"]},
                {"Metric": "Train Rows", "Value": metrics["train_rows"]},
                {"Metric": "Test Rows", "Value": metrics["test_rows"]},
                {"Metric": "Feature Count", "Value": metrics["feature_count"]},
            ])
            st.dataframe(summary_df, width='stretch', hide_index=False)
        with perf_right:
            fig_error_dist = px.histogram(results_df, x="Residuals", nbins=20, title="Residual Distribution")
            fig_error_dist.add_vline(x=0, line_dash="dash", line_color="orange")
            st.plotly_chart(fig_error_dist, width='stretch')
        fig_rank = px.scatter(
            results_df.reset_index(drop=True).assign(Row=lambda d: d.index + 1),
            x="Row",
            y="Abs_Error",
            color="Abs_Error",
            title="Absolute Error by Test Row",
            color_continuous_scale="Turbo",
        )
        st.plotly_chart(fig_rank, width='stretch')

    with tab4:
        st.subheader("🔍 VIF Test For Multicollinearity")
        if numeric_train_df.shape[1] < 2:
            st.info("VIF requires at least two numeric feature columns.")
        else:
            vif_df_input = numeric_train_df.copy()
            vif_df_input = vif_df_input.replace([np.inf, -np.inf], np.nan)

            imputer = SimpleImputer(strategy="median")
            imputed_array = imputer.fit_transform(vif_df_input)

            vif_imputed = pd.DataFrame(imputed_array, columns=vif_df_input.columns[:imputed_array.shape[1]])

            scaler = StandardScaler()
            scaled_array = scaler.fit_transform(vif_imputed)
            vif_scaled = pd.DataFrame(scaled_array, columns=vif_imputed.columns)

            vif_data = check_vif(vif_scaled)
            if vif_data is None:
                st.info("VIF could not be computed reliably for the data after cleaning constant or missing columns.")
            else:
                st.dataframe(vif_data, width='stretch', hide_index=False)
                high_vif = vif_data[(vif_data["feature"] != "const") & (vif_data["VIF"] > 10)]
                if len(high_vif) > 0:
                    st.error(f"🚨 {len(high_vif)} High-VIF Features (>10): {', '.join(high_vif['feature'].tolist())}")
                    fig_vif = px.bar(high_vif.head(10), x="feature", y="VIF", title="Top Features By Multicollinearity")
                    st.plotly_chart(fig_vif, width='stretch')

else:
    st.info("Upload a dataset to begin")

st.markdown("---")
st.caption("Health Analytics Dashboard • For research and educational purposes only")
