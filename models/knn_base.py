# MIT License
# See LICENSE file in the project root or at https://opensource.org/license/mit
#
# Copyright (c) 2025 Landon Nguyen, Alex Nguyen, Robert Cocker
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
import io
import re
import json
import joblib
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_regression
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, learning_curve, validation_curve
"""
Example Usage: See Jupyter Notebooks for more information.

!python knn_base.py \
--data ../data/Key_indicator_districtwise.csv \
--target YY_Infant_Mortality_Rate_Imr_Total_Person \
--id-cols State_Name State_District_Name \
--test-size 0.25 --random-state 42 --outdir knn

"""
warnings.filterwarnings("ignore")
sns.set_palette("husl")
plt.style.use('default')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(datapath):
    df = pd.read_csv(datapath)
    print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df

def verify_columns(df, target, id_cols):
    print("\n🔍 Verifying Features:")
    print(f"✓  Target: '{target}' → {'✅ Found' if target in df.columns else '❌ Missing'}")
    
    for id_col in id_cols:
        status = '✅ Found' if id_col in df.columns else '❌ Missing'
        print(f"✓  ID: '{id_col}' → {status}")
    print(f"✓  Dropping selected features from the training data..")
    if target not in df.columns:
        print("\n📋 All columns with 'Infant'/'IMR':")
        for col in df.columns:
            if 'infant' in col.lower() or 'imr' in col.lower():
                print(f"   - '{col}'")
        raise KeyError(f"Target '{target}' not found!")

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    return ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

def get_raw_feature_names(preprocessor, original_cols):
    feat_names = list(preprocessor.get_feature_names_out())
    raw_mapping = {}
    
    for feat in feat_names:
        if feat.startswith('num__'):
            raw_mapping[feat] = feat[5:]
        elif feat.startswith('cat__'):
            raw_mapping[feat] = feat[5:].split('_', 1)[1] if '_' in feat[5:] else feat[5:]
    
    return [raw_mapping.get(name, name) for name in feat_names]

def plot_true_vs_pred(y_true, y_pred, outdir, subset_label, metrics):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_true)), y_true, color='steelblue', label='True', s=40, alpha=0.7)
    plt.plot(range(len(y_pred)), y_pred, color='coral', label='Predicted', linewidth=3, alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.title(f'{subset_label} Predictions')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    text_str = (f'R²: {metrics["R2"]:.3f}\n'
                f'Adj R²: {metrics["Adj R2"]:.3f}\n'
                f'RMSE: {metrics["RMSE"]:.3f}\n'
                f'MAE: {metrics["MAE"]:.3f}')
    plt.gca().text(0.02, 0.98, text_str, transform=plt.gca().transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(Path(outdir)/f'{subset_label.lower()}_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {subset_label.lower()}_scatter.png")

def plot_residuals(y_true, y_pred, outdir, split_name):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0,0].scatter(y_pred, residuals, alpha=0.7, s=40, color='steelblue')
    axes[0,0].axhline(0, color='coral', linestyle='--', linewidth=2)
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].set_title('Residuals vs Predicted')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].hist(residuals, bins=10, color='steelblue', alpha=0.7, edgecolor='coral')
    axes[0,1].set_xlabel('Residuals')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Residuals Distribution')
    axes[0,1].grid(True, alpha=0.5)
    
    stats.probplot(residuals, dist="norm", plot=axes[1,0])
    axes[1,0].get_lines()[0].set_markerfacecolor('steelblue')
    axes[1,0].get_lines()[0].set_markeredgecolor('coral')
    axes[1,0].set_title('Q-Q Plot (Normality)')
    
    axes[1,1].scatter(range(len(residuals)), residuals, alpha=0.6, s=20, color='steelblue')
    axes[1,1].axhline(0, color='coral', linestyle='--', linewidth=2)
    axes[1,1].set_xlabel('Index')
    axes[1,1].set_ylabel('Residuals')
    axes[1,1].set_title('Residuals vs Index')
    axes[1,1].grid(True, alpha=0.5)
    
    plt.suptitle(f'{split_name} Residuals Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(Path(outdir)/f'{split_name.lower()}_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {split_name.lower()}_residuals.png")

def plot_feature_importance(importances, feature_names, selector_support, outdir, top_n=20):
    def clean_name(name: str) -> str:
        for prefix in ("num__", "cat__"):
            if name.startswith(prefix):
                name = name[len(prefix):]
        # remove your domain prefix
        if name.startswith("AA_"):
            name = name[3:]
        return name

    clean_feature_names = [clean_name(n) for n in feature_names]

    top_n = min(top_n, len(importances))
    idx = np.argsort(importances)[-top_n:][::-1]
    colors = ["darkgreen" if selector_support[i] else "steelblue" for i in idx]

    plt.figure(figsize=(12, 10))
    plt.barh(range(top_n), importances[idx], color=colors, alpha=0.7)

    labels = []
    for i in idx:
        n = clean_feature_names[i]
        labels.append(n[:35] + "..." if len(n) > 35 else n)

    plt.yticks(range(top_n), labels)
    plt.xlabel("Feature Importance")
    plt.title("Model Feature Importance")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.5)
    plt.tight_layout()
    plt.savefig(Path(outdir)/"feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: feature_importance.png")

def plot_learning_curve(estimator, X_df, y, outdir, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    est = clone(estimator)    
    train_sizes, train_scores, val_scores = learning_curve(
        est, X_df, y,  
        train_sizes=train_sizes, cv=cv, scoring='r2',
        n_jobs=-1, shuffle=True, random_state=42,
        error_score='raise'  
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='steelblue', label='Training R²', lw=2)
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', color='coral', label='CV R²', lw=2)
    plt.fill_between(train_sizes, 
                     np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                     np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), 
                     alpha=0.2, color='steelblue')
    plt.fill_between(train_sizes, 
                     np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                     np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), 
                     alpha=0.2, color='coral')
    plt.xlabel('Training Set Size')
    plt.ylabel('R² Score')
    plt.title('KNN + RFECV Regression Model Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir/'learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: learning_curve.png")
    print(f"✓ Final CV R²: {np.mean(val_scores[-1]):.4f} ± {np.std(val_scores[-1]):.4f}")
    print(f"✓ Final Train R²: {np.mean(train_scores[-1]):.4f} ± {np.std(train_scores[-1]):.4f}")

def plot_validation_curve(estimator, X, y, param_name, param_range, outdir, cv=5, **kwargs):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    
    est = clone(estimator)    
    train_scores, val_scores = validation_curve(
        est, X, y, 
        param_name=param_name, 
        param_range=param_range,
        cv=cv, 
        scoring='r2', 
        n_jobs=-1
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, np.mean(train_scores, axis=1), 'o-', color='steelblue', label='Training R²', lw=2)
    plt.plot(param_range, np.mean(val_scores, axis=1), 'o-', color='coral', label='CV R²', lw=2)
    plt.fill_between(param_range, 
                     np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                     np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), 
                     alpha=0.2, color='steelblue')
    plt.fill_between(param_range, 
                     np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                     np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), 
                     alpha=0.2, color='coral')
    plt.xlabel(param_name.replace('_', ' ').title())
    plt.ylabel('R² Score')
    plt.title('KNN + RFECV Regression Model Validation Curve')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(outdir/'validation_curve.png', dpi=300, bbox_inches='tight'); plt.close()
    
    best_idx = np.argmax(np.mean(val_scores, axis=1))
    print(f"Saved: validation_curve.png | Best CV R²: {np.mean(val_scores[best_idx]):.4f} ±{np.std(val_scores[best_idx]):.4f} (n_neighbors={param_range[best_idx]})")

def plot_prediction_distribution(y_true, y_pred, outdir, split_name):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(y_true, bins=7, alpha=0.6, label='True', color='steelblue', density=True)
    plt.hist(y_pred, bins=7, alpha=0.6, label='Predicted', color='coral', density=True)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'{split_name} Distribution')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.6, s=40, color='steelblue')
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'coral', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs True')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(Path(outdir)/f'{split_name.lower()}_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {split_name.lower()}_distribution.png")

def plot_model_comparison(train_metrics, test_metrics, outdir):
    metrics_df = pd.DataFrame([train_metrics, test_metrics], index=['Train', 'Test'])
    x = np.arange(len(metrics_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, metrics_df['R2'], width, label='R²', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, metrics_df['Adj R2'], width, label='Adj R²', color='coral', alpha=0.8)
    ax.set_xlabel('Split')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance: Train vs Test')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df.index)
    ax.legend()
    ax.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.savefig(Path(outdir)/'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: model_comparisons.png")

def plot_jitter_true_vs_pred(y_true, y_pred, outdir, split_name, jitter_level=1e-6):
    x_jitter = np.random.normal(0, jitter_level, len(y_true))
    y_jitter = np.random.normal(0, jitter_level, len(y_pred))
    
    x_jittered = y_true + x_jitter
    y_jittered = y_pred + y_jitter
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(x_jittered, y_jittered, alpha=0.7, s=80, c=y_pred-y_true, 
                         cmap='RdYlBu_r', edgecolors='black', linewidth=0.8)
    
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'coral', lw=3, alpha=0.9, label='Perfect Prediction')
    plt.axvline(y_true.mean(), color='steelblue', linestyle='--', alpha=0.7, label=f'True Mean: {y_true.mean():.1f}')
    plt.axhline(y_pred.mean(), color='coral', linestyle='--', alpha=0.7, label=f'Pred Mean: {y_pred.mean():.1f}')
    plt.xlabel('True Values (Jittered)')
    plt.ylabel('Predicted Values (Jittered)')
    plt.title(f'{split_name} Jittered True vs Predicted\n(Jitter={jitter_level}, Color=Residual)')
    plt.colorbar(scatter, label='Prediction Error (Pred-True)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    textstr = (f'R²: {r2_score(y_true, y_pred):.3f}\n'
               f'RMSE: {rmse:.3f}\n'
               f'MAE: {mae:.3f}\n'
               f'N: {len(y_true):,}')
    plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(Path(outdir)/f'{split_name.lower()}_jitter_true_vs_pred.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {split_name.lower()}_jitter_true_vs_pred.png")

def generate_streamlit_plots(fig):
    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()

def plot_true_vs_pred_bytes(y_true, y_pred, metrics, split_label):
    if len(y_true) == 0 or len(y_pred) == 0:
        st.warning(f"No data for model {split_label} true vs pred plot.")
        return None
    
    n_points = len(y_true)
    
    fig, ax = plt.subplots(figsize=(12, 6))  
    
    ax.scatter(range(n_points), y_true, color="steelblue", label="True", s=40, alpha=0.7, edgecolors="navy")
    ax.plot(range(n_points), y_pred, color="coral", label="Predicted", linewidth=2, alpha=0.8)
    
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Target_IMR")
    ax.set_title(f"Model {split_label} Predictions")
    
    textstr = ""
    if metrics:
        keys = ['R2', 'Adj R2', 'RMSE', 'MAE']
        for key in keys:
            if key in metrics:
                textstr += f"{key}: {metrics[key]:.3f}\n"
    else:
        textstr = "Metrics unavailable"
    
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    ax.legend()
    ax.grid(True, alpha=0.5)
    plt.tight_layout()
    return generate_streamlit_plots(fig)

def plot_residuals_bytes(y_true, y_pred, split_label):
    if len(y_true) == 0 or len(y_pred) == 0:
        st.warning(f"No data for model {split_label} residuals plot.")
        return None
    
    residuals = y_true - y_pred
    n_points = len(residuals)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].scatter(y_pred, residuals, alpha=0.7, s=40, color="steelblue")
    axes[0, 0].axhline(0, color="coral", linestyle="--", linewidth=2)
    axes[0, 0].set_xlabel("Predicted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Model Residuals vs Predicted")
    axes[0, 0].grid(True, alpha=0.5)
    
    axes[0, 1].hist(residuals, bins=10, color="steelblue", alpha=0.7, edgecolor="coral")
    axes[0, 1].set_xlabel("Residuals")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title(f"Model Residuals Distribution")
    axes[0, 1].grid(True, alpha=0.5)
    
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].get_lines()[0].set_markerfacecolor("steelblue")
    axes[1, 0].get_lines()[0].set_markeredgecolor("coral")
    axes[1, 0].get_lines()[1].set_color("coral")  
    axes[1, 0].set_title("Model Q-Q Plot (Normality)")
    axes[1, 0].grid(True, alpha=0.5)
    
    axes[1, 1].scatter(range(n_points), residuals, alpha=0.6, s=20, color="steelblue")
    axes[1, 1].axhline(0, color="coral", linestyle="--", linewidth=2)
    axes[1, 1].set_xlabel("Data Index")
    axes[1, 1].set_ylabel("Residuals")
    axes[1, 1].set_title("Residuals vs Index")
    axes[1, 1].grid(True, alpha=0.5)
    
    plt.suptitle(f"Model {split_label} Residuals Analysis", fontsize=14)
    plt.tight_layout()
    return generate_streamlit_plots(fig)

def plot_prediction_distribution_bytes(y_true, y_pred, split_label):
    if len(y_true) == 0 or len(y_pred) == 0:
        st.warning(f"No data for model {split_label} prediction plot.")
        return None
    
    n_points = len(y_true)
    
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = plt.subplot(1, 2, 1)
    ax1.hist(y_true, bins=7, alpha=0.7, label="True", color="steelblue", edgecolor="steelblue", linewidth=1, density=True)
    ax1.hist(y_pred, bins=7, alpha=0.7, label="Predicted", color="coral", edgecolor="steelblue", linewidth=1, density=True)
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Model {split_label} Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.5)
    
    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(y_true, y_pred, alpha=0.7, s=40, color="steelblue")
    minval, maxval = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax2.plot([minval, maxval], [minval, maxval], "coral", lw=2)
    ax2.set_xlabel("True Values")
    ax2.set_ylabel("Predicted Values")
    ax2.set_title(f"Model Predicted vs True")
    ax2.legend()
    ax2.grid(True, alpha=0.5)
    plt.tight_layout()
    return generate_streamlit_plots(fig)

def plot_feature_importance_bytes(importances, feature_names, selector_support, top_n=20):
    def clean_name(name):
        name = str(name).replace("_", " ")
        for prefix in ["num__", "cat__"]:
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name.startswith("AA"):
            name = name[2:]
        return name
    clean_feature_names = [clean_name(n) for n in feature_names]
    
    top_n = min(top_n, len(importances))
    idx = np.argsort(importances)[-top_n:][::-1]  
    colors = ["steelblue" if selector_support[i] else "coral" for i in idx]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(range(top_n), importances[idx], color=colors, alpha=0.7)
    
    labels = []
    for i in idx:
        n = clean_feature_names[i]
        labels.append(n[:35] + "..." if len(n) > 35 else n)
    
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Model Feature Importance")
    ax.invert_yaxis()  
    ax.grid(axis="x", alpha=0.5)
    plt.tight_layout()
    return generate_streamlit_plots(fig)

def plot_learning_curve_bytes(estimator, X_train, y_train, cv=5, n_samples=None, n_neighbors=5, test_size_pct=0.2):
    n_samples_total = len(X_train) if X_train is not None else 0
    
    safe_cv_splits = min(5, max(2, n_samples_total//4))
    safe_n_jobs = 1 if n_samples_total < 20 else -1
    min_train_size = max(2, int(0.2 * n_samples_total * (1 - test_size_pct))) 
    safe_n_points = min(10, max(3, min_train_size//5))
    train_sizes = np.linspace(max(0.2, 2/n_samples_total), 1.0, safe_n_points)
    
    if n_samples_total < 4 or min_train_size < 1:
        st.warning("Model learning curve skipped: dataset too small (<4 train samples).")
        return None 
    
    try:
        est = clone(estimator)
        orig_neighbors = getattr(est, 'n_neighbors', n_neighbors)
        safe_neighbors = min(orig_neighbors, max(1, min_train_size//2))
        if hasattr(est, 'n_neighbors'):
            est.n_neighbors = safe_neighbors
        
        cv_lc = KFold(n_splits=safe_cv_splits, shuffle=True, random_state=42)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            est, X_train, y_train,
            train_sizes=train_sizes, cv=cv_lc,
            scoring="r2", n_jobs=safe_n_jobs,
            shuffle=True, random_state=42,
            error_score=np.nan  
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_sizes_abs, np.mean(train_scores, axis=1), "o-", color="steelblue", label="Training R²", lw=2)
        ax.plot(train_sizes_abs, np.mean(val_scores, axis=1), "o-", color="coral", label="CV R²", lw=2)
        ax.fill_between(train_sizes_abs, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                        np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.5, color="steelblue")
        ax.fill_between(train_sizes_abs, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                        np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.5, color="coral")
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("R² Score")
        ax.set_title(f"Model Learning Curve")
        ax.legend()
        ax.grid(True, alpha=0.5)
        plt.tight_layout()
        return generate_streamlit_plots(fig)
    
    except Exception as e:
        st.error(f"Model learning curve failed: {str(e)[:100]}... Skipping.")
        return None

def plot_validation_curve_bytes(estimator, X_train, y_train, param_name, param_range, cv=5, n_samples=None, test_size_pct=0.2):
    n_samples_total = len(X_train) if X_train is not None else 0

    safe_cv_splits = min(5, max(2, n_samples_total//4))
    safe_n_jobs = 1 if n_samples_total < 20 else -1
    min_fold_size = n_samples_total/safe_cv_splits
    safe_range_len = min(8, max(3, n_samples_total//10))
    
    orig_range = np.array(param_range)
    safe_max_param = max(1, int(min_fold_size//2))
    safe_param_range = np.clip(orig_range, 1, safe_max_param)
    if len(safe_param_range) > safe_range_len or len(np.unique(safe_param_range)) < 3:
        safe_param_range = np.linspace(1, min(safe_max_param, orig_range.max()), safe_range_len).astype(int)
    
    if n_samples_total < 4 or len(safe_param_range) < 2:
        st.warning(f"Model validation curve skipped: too small (n={n_samples_total}, safe_range={safe_param_range}).")
        return None
    
    try:
        est = clone(estimator)
        cv_vc = KFold(n_splits=safe_cv_splits, shuffle=True, random_state=42)
        
        train_scores, val_scores = validation_curve(
            est, X_train, y_train,
            param_name=param_name, param_range=safe_param_range.tolist(),
            cv=cv_vc, scoring="r2", n_jobs=safe_n_jobs,
            error_score=np.nan  
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(safe_param_range, np.mean(train_scores, axis=1), "o-", color="steelblue", label="Training R²", lw=2)
        ax.plot(safe_param_range, np.mean(val_scores, axis=1), "o-", color="coral", label="CV R²", lw=2)
        ax.fill_between(safe_param_range, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                        np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.5, color="steelblue")
        ax.fill_between(safe_param_range, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                        np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.5, color="coral")
        ax.set_xlabel(param_name.replace("_", " ").title())
        ax.set_ylabel("R² Score")
        ax.set_title(f"Model Validation Curve")
        ax.legend()
        ax.grid(True, alpha=0.5)
        plt.tight_layout()
        return generate_streamlit_plots(fig)
    
    except Exception as e:
        st.error(f"Model validation curve failed ({param_name}): {str(e)[:100]}... Skipping.")
        return None

def calculate_adjusted_r2(r2_score, n_samples, n_features):
    if n_samples <= n_features + 1:
        return np.nan
    return 1 - (1 - r2_score) * (n_samples - 1)/(n_samples - n_features - 1)

def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    
    print("KNN Regression Model")
    print(f"📁 Output: {outdir}")
    print("="*80)
    
    df = load_data(args.data)
    verify_columns(df, args.target, args.id_cols)
    id_cols_found = [col for col in args.id_cols if col in df.columns]
    X = df.drop(columns=[args.target] + id_cols_found)
    y = df[args.target]
    
    print(f"✅ Features: {X.shape[1]} | Target range: {y.min():.1f}-{y.max():.1f}")
    
    mask = ~df.duplicated(subset=X.columns.tolist())
    df_deduped = df[mask]
    X = df_deduped.drop(columns=[args.target] + id_cols_found)
    y = df_deduped[args.target]
    print(f"✅ Deduplicated: {len(X):,} samples")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, shuffle=True
    )
    
    preprocessor = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    raw_feature_names = get_raw_feature_names(preprocessor, X_train.columns.tolist())
    
    print(f"✅ Processed: {X_train_proc.shape}")
    print("\n🎯 RFECV Feature Selection... Running a pass on the data")
    cv = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    
    selector = RFECV(rf, step=0.1, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    selector.fit(X_train_proc, y_train)
    
    print(f"✅ Selected: {selector.n_features_} features")
    print_selected_features_raw(selector, raw_feature_names)
    
    X_train_sel = selector.transform(X_train_proc)
    X_test_sel = selector.transform(X_test_proc)
    
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='manhattan')
    knn.fit(X_train_sel, y_train)
    
    y_train_pred = knn.predict(X_train_sel)
    y_test_pred = knn.predict(X_test_sel)
    
    n_train, p = len(y_train), X_train_sel.shape[1]
    n_test = len(y_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    train_adj_r2 = calculate_adjusted_r2(train_r2, n_train, p)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    test_r2 = r2_score(y_test, y_test_pred)
    test_adj_r2 = calculate_adjusted_r2(test_r2, n_test, p)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_metrics = {'R2': train_r2, 'Adj R2': train_adj_r2, 'RMSE': train_rmse, 'MAE': train_mae}
    test_metrics = {'R2': test_r2, 'Adj R2': test_adj_r2, 'RMSE': test_rmse, 'MAE': test_mae}
    
    print(f"\n📊 KNN Regression Model Inference Metrics:")
    print(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    print(f"Train RMSE: {train_rmse:.3f} | Test RMSE: {test_rmse:.3f}")
    
    joblib.dump(knn, outdir/'knn_model.joblib')
    joblib.dump(preprocessor, outdir/'preprocessor.joblib')
    joblib.dump(selector, outdir/'rfecv_selector.joblib')
    
    print("\n📈 Generating KNN model inference plots...")
    print("-" * 50)

    raw_feature_names = get_raw_feature_names(preprocessor, X_train.columns.tolist())
    plot_true_vs_pred(y_train, y_train_pred, outdir, 'Train', train_metrics)
    plot_true_vs_pred(y_test, y_test_pred, outdir, 'Test', test_metrics)
    plot_residuals(y_train, y_train_pred, outdir, 'Train')
    plot_residuals(y_test, y_test_pred, outdir, 'Test')
    plot_feature_importance(selector.estimator_.feature_importances_, raw_feature_names, selector.support_, outdir)
    plot_prediction_distribution(y_train, y_train_pred, outdir, 'Train')
    plot_prediction_distribution(y_test, y_test_pred, outdir, 'Test')
    plot_model_comparison(train_metrics, test_metrics, outdir)
    
    plot_jitter_true_vs_pred(y_train, y_train_pred, outdir, 'Train', jitter_level=0.3)
    plot_jitter_true_vs_pred(y_test, y_test_pred, outdir, 'Test', jitter_level=0.3)
    
    selected_features = [raw_feature_names[i] for i, sel in enumerate(selector.support_) if sel]
    pd.Series(selected_features).to_csv(outdir/'selected_features_raw.csv', index=False)
    
    print("\n" + "="*70)
    print("✅ KNN Base Model Results:")
    print("="*70)
    print(f"🎯 Test: R²={test_r2:.4f} | Adj R²={test_adj_r2:.4f} | RMSE={test_rmse:.4f} | MAE={test_mae:.4f}")
    print(f"📊 Features: {selector.n_features_}/{len(raw_feature_names)}")
    print(f"📁 Outputs: {outdir}")
    print("="*70)

def generate_prediction_df(input_dict, feature_names, id_cols, n_features):
    input_data = dict(input_dict)  

    for i in range(n_features):
        col = f'Health_Indicator_{i}'
        if col not in input_data:
            input_data[col] = 0 
    for col in id_cols:
        if col not in input_data:
            input_data[col] = 'Dummy'
    return pd.DataFrame([input_data])

def print_selected_features_raw(selector_or_support, raw_feature_names, top_n=20):
    print("\n" + "="*80)
    print("🎯 Top Selected Features - Raw Names:")
    print("="*80)
    
    if hasattr(selector_or_support, 'support_'):
        selected_mask = selector_or_support.support_
    else:  # numpy array/boolean mask
        selected_mask = selector_or_support
    
    selected_raw = [raw_feature_names[i] for i, selected in enumerate(selected_mask) if selected]
    
    for i, feat in enumerate(selected_raw[:top_n]):
        print(f"  {i+1:2d}. '{feat}'")
    
    total = len(selected_raw)
    print(f"\n📊 Selected: {total}/{len(raw_feature_names)} ({100*total/len(raw_feature_names):.1f}%)")
    print("="*80)

def drop_highly_correlated(X, threshold=0.70):
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    print(f"✅ Dropped {len(to_drop)} correlated features (thresh={threshold}):")
    for col in to_drop[:8]:
        print(f"  - {col}")
    if len(to_drop) > 8:
        print(f"  ... +{len(to_drop)-8} more")
    
    X_filtered = X.drop(columns=to_drop)
    print(f"✅ Features: {X.shape[1]} → {X_filtered.shape[1]}")
    return X_filtered, to_drop

def generate_and_train(
    n_samples=250, n_features=64, test_size=0.25, random_state=42,
    n_neighbors=5, weights='distance', metric='manhattan', corr_threshold=0.85
):
    print(f"Generating {n_samples} samples with {n_features} features...")
    np.random.seed(random_state)

    n_base = 8
    base_health = np.random.uniform(20, 80, n_samples)
    factors = {
        'income': base_health * 0.7 + np.random.normal(0, 8, n_samples),
        'education': base_health * 0.75 + np.random.normal(0, 6, n_samples),
        'sanitation': base_health * 0.8 + np.random.normal(0, 7, n_samples),
        'hospitals': base_health * 0.65 + np.random.normal(0, 9, n_samples),
        'vaccines': base_health * 0.85 + np.random.normal(0, 5, n_samples),
        'nutrition': base_health * 0.9 + np.random.normal(0, 4, n_samples),
        'water': base_health * 0.82 + np.random.normal(0, 6, n_samples),
        'roads': base_health * 0.6 + np.random.normal(0, 10, n_samples),
    }

    base_df = pd.DataFrame({k: v for k, v in factors.items()})
    corr_matrix = base_df.corr().abs()
    print(f"✅ Base factors corr max: {corr_matrix.values.max():.3f}")

    n_blocks = (n_features + n_base - 1) // n_base
    X_raw = np.tile(base_df.values, (1, n_blocks))[:, :n_features]
    noise_scale = np.linspace(1.0, 0.2, n_features)
    X_raw += np.random.normal(0, noise_scale * 4, X_raw.shape)

    imr_base = 120 - base_df.mean(axis=1) * 1.5
    y_raw = np.clip(imr_base + np.random.normal(0, 4, n_samples), 15, 110)

    feature_names = [f'Health_Indicator_{i}' for i in range(n_features)]
    df = pd.DataFrame(X_raw, columns=feature_names)
    df['Target_IMR'] = y_raw 
    df['State_Name'] = np.random.choice(['State_A', 'State_B', 'State_C'], n_samples)
    df['District_Name'] = [f'Dist_{i}' for i in range(n_samples)]

    id_cols = ['State_Name', 'District_Name']
    feature_cols = feature_names
    df = df[id_cols + feature_cols + ['Target_IMR']]

    print(f"Generated df: {df.shape} | Target range {df['Target_IMR'].min():.1f}-{df['Target_IMR'].max():.1f}")

    class Args:
        data_path = None
        target = 'Target_IMR'
        id_cols = ['State_Name', 'District_Name']

    args = Args()
    args.test_size = test_size
    args.random_state = random_state

    verify_columns(df, args.target, args.id_cols)
    id_cols_found = [col for col in args.id_cols if col in df.columns]
    X = df.drop(columns=[args.target] + id_cols_found)
    y = df[args.target]
    print(f"✅ Features: {X.shape[1]} | Target range: {y.min():.1f}-{y.max():.1f}")

    mask = ~df.duplicated(subset=X.columns.tolist())
    df_deduped = df[mask]
    X = df_deduped.drop(columns=[args.target] + id_cols_found)
    y = df_deduped[args.target]
    print(f"✅ Deduplicated: {len(X):,} samples")

    print("\n🧹 Removing multicollinear features...")
    corr_dropped = []
    if corr_threshold < 0.99:
        X, corr_dropped = drop_highly_correlated(X, threshold=corr_threshold)
    print("-" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, shuffle=True
    )

    preprocessor = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    raw_feature_names = get_raw_feature_names(preprocessor, X_train.columns.tolist())
    print(f"✅ Processed: {X_train_proc.shape}")

    cv = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

    n_features = X_train_proc.shape[1]
    print(f"X_train_proc shape: {X_train_proc.shape}")

    selector = None
    selector_support = np.ones(n_features, dtype=bool)  
    n_selected = n_features

    if n_features < 2:
        st.warning(f"Skipping RFECV: only {n_features} feature(s) after processing (need ≥2). Using all.")
        X_train_sel = X_train_proc
        X_test_sel = X_test_proc
        selector_support = np.ones(n_features, dtype=bool) 
        n_selected = n_features

    elif n_features == 0:
        st.error("No features after processing! Check data/corr_threshold.")
        st.stop()

    else:
        min_features = max(1, min(5, n_features//10, n_features))  
        selector = RFECV(
            rf, step=0.1, cv=cv, scoring='neg_root_mean_squared_error',
            min_features_to_select=min_features, n_jobs=-1
        )

        n_splits = cv.n_splits if hasattr(cv, 'n_splits') else 5
        samples_per_fold = len(X_train_proc)/n_splits
        if samples_per_fold < 3: 
            st.error(f"Too few samples for RFECV CV (samples/fold ~{samples_per_fold:.1f} < 3). Reduce n_splits or add data.")
            print_selected_features_raw(selector_support, raw_feature_names)
            st.stop()
        
        selector.fit(X_train_proc, y_train)
        X_train_sel = selector.transform(X_train_proc)
        X_test_sel = selector.transform(X_test_proc)

        selector_support = selector.support_
        n_selected = selector.n_features_
        print(f"✅ Selected: {n_selected} features")
        if selector is not None:
            print_selected_features_raw(selector, raw_feature_names)
        else:
            print_selected_features_raw(selector_support, raw_feature_names)

    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, metric=metric)
    if len(X_train_sel) < n_neighbors:
        st.error(f"Need at least {n_neighbors} samples; got {len(X_train_sel)}")
        st.stop()

    knn.fit(X_train_sel, y_train)
    y_train_pred = knn.predict(X_train_sel)
    y_test_pred = knn.predict(X_test_sel)

    n_train, p = len(y_train), X_train_sel.shape[1]
    n_test = len(y_test)
    train_r2 = r2_score(y_train, y_train_pred)
    train_adj_r2 = calculate_adjusted_r2(train_r2, n_train, p)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)

    test_r2 = r2_score(y_test, y_test_pred)
    test_adj_r2 = calculate_adjusted_r2(test_r2, n_test, p)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_metrics = {'R2': train_r2, 'Adj R2': train_adj_r2, 'RMSE': train_rmse, 'MAE': train_mae}
    test_metrics = {'R2': test_r2, 'Adj R2': test_adj_r2, 'RMSE': test_rmse, 'MAE': test_mae}

    print(f"\n📊 KNN Regression Model Inference Metrics:")
    print(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    print(f"Train RMSE: {train_rmse:.3f} | Test RMSE: {test_rmse:.3f}")

    print("\n📈 Generating model plots...")
    rf.fit(X_train_proc, y_train)
    importances = rf.feature_importances_
    selected_features = [raw_feature_names[i] for i, sel in enumerate(selector_support) if sel]
    print(f"Selected features: {len(selected_features)}): {selected_features}")

    return {
        "df": df,
        "corr_dropped": corr_dropped,
        "corr_threshold": corr_threshold,
        "knn": knn,
        "preprocessor": preprocessor,
        "selector": selector,
        "selected_features": selected_features,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_scatter_bytes": plot_true_vs_pred_bytes(y_train, y_train_pred, train_metrics, "Train"),
        "test_scatter_bytes": plot_true_vs_pred_bytes(y_test, y_test_pred, test_metrics, "Test"),
        "train_residuals_bytes": plot_residuals_bytes(y_train, y_train_pred, "Train"),
        "test_residuals_bytes": plot_residuals_bytes(y_test, y_test_pred, "Test"),
        "train_distribution_bytes": plot_prediction_distribution_bytes(y_train, y_train_pred, "Train"),
        "test_distribution_bytes": plot_prediction_distribution_bytes(y_test, y_test_pred, "Test"),
        "feature_importance_bytes": plot_feature_importance_bytes(importances, raw_feature_names, selector_support),
        "learning_curve_bytes": plot_learning_curve_bytes(knn, X_train_sel, y_train, cv=5),
        "validation_curve_bytes": plot_validation_curve_bytes(knn, X_train_sel, y_train, "n_neighbors", list(range(1, min(26, len(X_train_sel)+1), 2))),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN regression base model. Machine learning utilizing key health indicators for infant mortality rate  prediction.")
    parser.add_argument("--data", required=True, help="CSV dataset file")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--id-cols", nargs="*", default=[], help="ID columns")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--outdir", default="artifacts/knn")
    args = parser.parse_args()
    main(args)
