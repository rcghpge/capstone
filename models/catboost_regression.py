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
import warnings
warnings.filterwarnings('ignore')
import re
import sys
import time
import json
import joblib
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
from catboost import CatBoostRegressor, Pool
from scipy import stats

sns.set_palette('husl')
plt.style.use('default')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler('catboost_regression.log'), logging.StreamHandler()]
)

def calculate_adjusted_r2(r2_score, n_samples, n_features):
    if n_samples <= n_features + 1:
        return np.nan
    return 1 - (1 - r2_score) * (n_samples - 1)/(n_samples - n_features - 1)
    
def load_data(datapath):
    if datapath.endswith('.csv'):
        return pd.read_csv(datapath)
    elif datapath.endswith('.parquet'):
        return pd.read_parquet(datapath)
    raise ValueError("Unsupported file type. Use .csv or .parquet")

def find_target_column(df, target_name):
    target_candidates = [col for col in df.columns if target_name.lower() in col.lower()]
    return target_candidates[0] if target_candidates else target_name.lower()

def find_state_column(df, id_cols):
    candidates = []
    for col in id_cols:
        matching_cols = [c for c in df.columns if col.lower() in c.lower()]
        candidates.extend(matching_cols)
    
    state_patterns = ['state', 'region', 'province', 'county', 'district']
    for pattern in state_patterns:
        matches = df.columns[df.columns.str.lower().str.contains(pattern, na=False)]
        candidates.extend(matches.tolist())
    
    for col in candidates:
        if col in df.columns and df[col].nunique() > 1:
            return col
    return None

def get_feature_names(preprocessor):
    feature_names = list(preprocessor.get_feature_names_out())
    feature_names = [name.split('__', 1)[1] if '__' in name else name for name in feature_names]
    feature_names = [name[3:] if len(name) > 3 and name[2] == '_' else name for name in feature_names]
    return feature_names

def drop_feature_correlations(X_train, X_test, y_train, feature_names, drop_pct):
    drop_pct = float(max(0.0, min(100.0, drop_pct)))
    if drop_pct == 0:
        return X_train, X_test, feature_names
    
    y_arr = np.asarray(y_train).ravel()
    n_features = X_train.shape[1]
    if n_features <= 1:
        return X_train, X_test, feature_names
    
    corrs = np.corrcoef(X_train.T, y_arr)[-1, :-1]
    abs_corrs = np.abs(corrs)
    n_drop = int(round(n_features * drop_pct/100.0))
    
    if n_drop == 0 or n_drop >= n_features:
        return X_train, X_test, feature_names
    
    keep_idx = np.argsort(abs_corrs)[:n_features-n_drop]
    X_train_new = X_train[:, keep_idx]
    X_test_new = X_test[:, keep_idx]
    feature_names_new = [feature_names[i] for i in keep_idx]
    
    print(
        f"✓ Dropped {n_drop}/{n_features} features " 
        f"({drop_pct:.2f}%) by |corr| with target; " 
        f"{len(feature_names_new)} remain."
    )
    return X_train_new, X_test_new, feature_names_new

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

def plot_statewise_histogram(df, valuecol, statecol, out_dir):
    plt.figure(figsize=(14, 8))
    state_means = df.groupby(statecol)[valuecol].mean().sort_values()
    states_order = state_means.index.tolist()
    plot_data = df[df[statecol].isin(states_order)]
    palette = sns.color_palette("husl", n_colors=len(states_order))

    sns.histplot(
        data=plot_data,
        x=valuecol,
        hue=statecol,
        hue_order=states_order,
        element="step",
        stat="count",
        common_norm=False,
        palette=palette,
        alpha=0.5,
        multiple="layer", 
    )

    plt.title("Infant Mortality Rate by State")
    plt.xlabel(valuecol)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.grid(True, alpha=0.5)
    plt.savefig(Path(out_dir)/"statewise_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ statewise_histogram.png")

def plot_statewise_facets(df, value_col, state_col, out_dir):
    top_states = df[state_col].value_counts().head(9).index
    plot_data = df[df[state_col].isin(top_states)]

    g = sns.FacetGrid(plot_data, col=state_col, col_wrap=3, height=3, aspect=1.3, sharex=False, sharey=False)
    g.map_dataframe(sns.histplot, x=value_col, bins=9, color="steelblue", alpha=0.7)
    
    for ax in g.axes.flatten():
        if ax is not None:
            ax.grid(True, alpha=0.5)

    g.set_titles("{col_name}")
    g.set_axis_labels(value_col, "Count")
    g.fig.suptitle(f"{value_col} by {state_col}", y=1.02)
    plt.savefig(Path(out_dir)/"statewise_facets.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ statewise_facets.png")

def print_dataset_stats(df, target_col):
    print("\n" + "="*80)
    print("📈 Raw Dataset Summary")
    print('='*80)
    total_samples, total_features = df.shape
    print(f"📊 Dataset Shape: {total_samples:,} samples, {total_features:,} features")
    print(f"🎯 Target Column: {target_col}")
    
    total_missing = df.isnull().sum().sum()
    missing_pct = total_missing/(total_samples * total_features) * 100
    print(f"🔍 Total Missing Null/NaN Values: {total_missing:,} ({missing_pct:.2f}%)")
    
    target_missing = df[target_col].isnull().sum()
    target_missing_pct = target_missing / total_samples * 100
    print(f"🎯 Target Missing: {target_missing:,}/{total_samples:,} ({target_missing_pct:.1f}%)")
    
    if target_missing > 0:
        print(f"⚠️ Warning: Target has missing values!")
    
    feature_missing = df.drop(columns=[target_col]).isnull().sum()
    missing_features = feature_missing[feature_missing > 0].sort_values(ascending=False)
    
    if len(missing_features) > 0:
        print("\n📋 Top 10 Features with Missing Null/NaN Values")
        print("-" * 60)
        for feature, count in missing_features.head(10).items():
            pct = count/total_samples * 100
            print(f"{str(feature):40s} | {count:6,} ({pct:5.1f}%)")
        print(f"\n📊 Total features with missing values: {len(missing_features)}/{total_features - 1}")
    else:
        print("\n✅ No missing values in features!")
    
    dtype_counts = df.dtypes.value_counts()
    print("\n🔧 Data Types:")
    for dtype, count in dtype_counts.items():
        dtype_str = str(dtype)[:14]
        print(f"ℹ️ {dtype_str:10s} | {count:3d} columns")
    print('='*80)

def plot_feature_importance(importances, feature_names, selector_support, out_dir, top_n=25):
    top_n = min(top_n, len(importances))
    idx = np.argsort(importances)[-top_n:][::-1]
    colors = ['steelblue' if selector_support[i] else 'coral' for i in idx]
    
    plt.figure(figsize=(12, max(8, top_n*0.4)))  # Dynamic height
    bars = plt.barh(range(top_n), importances[idx], color=colors, alpha=0.7)
    plt.yticks(range(top_n), 
               [feature_names[i][:35] + '...' if len(feature_names[i]) > 35 else feature_names[i] 
                for i in idx])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest RFECV Feature Importance (Inputs to CatBoost Model)')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.5)
    plt.tight_layout()
    plt.savefig(Path(out_dir)/'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ feature_importance.png")

def plot_feature_target_correlations(X_processed, y_train, feature_names, selector_support, out_dir, top_n=25, chunk_size=5000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X_processed)
    y = np.asarray(y_train).ravel()
    n_samples, n_features = X.shape
    if len(feature_names) != n_features:
        feature_names = feature_names[:n_features]
    if len(selector_support) != n_features:
        selector_support = selector_support[:n_features]

    top_n = min(top_n, n_features)
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        print("Target has zero variance; skipping correlation plot.")
        return
        
    best = []
    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        X_chunk = X[:, start:end]

        Xm = X_chunk - X_chunk.mean(axis=0, keepdims=True)
        ym = y - y_mean

        num = (Xm * ym[:, None]).sum(axis=0)
        std_x = Xm.std(axis=0, ddof=0)
        denom = std_x * y_std * n_samples
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_chunk = np.where(denom != 0, num/denom, 0.0)

        abs_corr_chunk = np.abs(corr_chunk)
        for local_i, (ac, c) in enumerate(zip(abs_corr_chunk, corr_chunk)):
            g_i = start + local_i
            if len(best) < top_n:
                best.append((ac, c, g_i))
                if len(best) == top_n:
                    best.sort(key=lambda x: x[0])
            else:
                if ac > best[0][0]:
                    best[0] = (ac, c, g_i)
                    best.sort(key=lambda x: x[0])

    if not best:
        print("No correlations computed; skipping correlation plot.")
        return

    best.sort(key=lambda x: x[0], reverse=True)
    top_corrs = np.array([b[1] for b in best])
    top_idx = np.array([b[2] for b in best], dtype=int)
    top_features = [feature_names[i] for i in top_idx]
    is_selected = [bool(selector_support[i]) for i in top_idx]

    colors = ["steelblue" if sel else ("coral" if corr > 0 else "darkgreen") for corr, sel in zip(top_corrs, is_selected)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[3, 1])
    bars1 = ax1.barh(range(len(top_corrs)), top_corrs, color=colors, alpha=0.7, height=0.7)
    ax1.set_yticks(range(len(top_corrs)))
    ax1.set_yticklabels([f[:35] + "..." if len(f) > 35 else f for f in top_features], fontsize=10)
    ax1.set_xlabel("Correlation with Target", fontsize=12, weight="bold")
    ax1.set_title("Top CatBoost Feature-Target Correlations Selected", fontsize=14, weight="bold")
    ax1.grid(axis="x", alpha=0.5)
    ax1.axvline(0, color="black", linestyle="-", alpha=0.5)

    for bar, corr in zip(bars1, top_corrs):
        width = bar.get_width()
        ax1.text(
            width + (0.01 if width >= 0 else -0.03),
            bar.get_y() + bar.get_height()/2,
            f"{corr:.3f}",
            ha="left" if width >= 0 else "right",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    selected_count = sum(is_selected)
    ax1.text(
        0.02,
        0.98,
        f"RFECV Selected: {selected_count}/{len(top_corrs)}",
        transform=ax1.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
        fontsize=11,
    )

    selected_corrs = [c for c, s in zip(top_corrs, is_selected) if s]
    non_selected_corrs = [c for c, s in zip(top_corrs, is_selected) if not s]
    ax2.hist(
        [selected_corrs, non_selected_corrs],
        bins=10,
        alpha=0.7,
        label=["Selected", "Not Selected"],
        color=["steelblue", "coral"],
        density=True,
    )
    ax2.set_xlabel("Correlation Coefficient")
    ax2.set_ylabel("Density")
    ax2.set_title("Correlation Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig(Path(out_dir)/"feature_target_correlations.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ feature_target_correlations.png")

def plot_selected_feature_corrs(X_selected, y_train, selected_feature_names, out_dir, top_n=25, chunk_size=5000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X_selected)
    y = np.asarray(y_train).ravel()
    n_samples, n_features = X.shape

    if n_features == 0:
        print("No selected features; skipping model-input correlation plot.")
        return

    if len(selected_feature_names) != n_features:
        selected_feature_names = selected_feature_names[:n_features]

    top_n = min(top_n, n_features)
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        print("Target has zero variance; skipping model-input correlation plot.")
        return

    best = []
    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        X_chunk = X[:, start:end]

        Xm = X_chunk - X_chunk.mean(axis=0, keepdims=True)
        ym = y - y_mean

        num = (Xm * ym[:, None]).sum(axis=0)
        std_x = Xm.std(axis=0, ddof=0)
        denom = std_x * y_std * n_samples
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_chunk = np.where(denom != 0, num/denom, 0.0)

        abs_corr_chunk = np.abs(corr_chunk)
        for local_i, (ac, c) in enumerate(zip(abs_corr_chunk, corr_chunk)):
            g_i = start + local_i
            if len(best) < top_n:
                best.append((ac, c, g_i))
                if len(best) == top_n:
                    best.sort(key=lambda x: x[0])
            else:
                if ac > best[0][0]:
                    best[0] = (ac, c, g_i)
                    best.sort(key=lambda x: x[0])

    if not best:
        print("No correlations computed; skipping model-input correlation plot.")
        return

    best.sort(key=lambda x: x[0], reverse=True)
    top_corrs = np.array([b[1] for b in best])
    top_idx = np.array([b[2] for b in best], dtype=int)
    top_features = [selected_feature_names[i] for i in top_idx]

    is_selected = [True] * len(top_corrs)
    colors = ["steelblue" if sel else ("coral" if corr > 0 else "darkgreen")
              for corr, sel in zip(top_corrs, is_selected)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[3, 1])

    bars1 = ax1.barh(range(len(top_corrs)), top_corrs, color=colors, alpha=0.7, height=0.7)
    ax1.set_yticks(range(len(top_corrs)))
    ax1.set_yticklabels(
        [f[:35] + "..." if len(f) > 35 else f for f in top_features],
        fontsize=10
    )
    ax1.set_xlabel("Correlation with Target", fontsize=12, weight="bold")
    ax1.set_title(
        f"Top CatBoost Model-Input Feature-Target Correlations (Selected {n_features})",
        fontsize=14,
        weight="bold",
    )
    ax1.grid(axis="x", alpha=0.5)
    ax1.axvline(0, color="black", linestyle="-", alpha=0.5)

    for bar, corr in zip(bars1, top_corrs):
        width = bar.get_width()
        ax1.text(
            width + (0.01 if width >= 0 else -0.03),
            bar.get_y() + bar.get_height() / 2,
            f"{corr:.3f}",
            ha="left" if width >= 0 else "right",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    selected_count = len(top_corrs)
    ax1.text(
        0.02,
        0.98,
        f"CatBoost Model Input (RFECV): {selected_count}/{n_features}",
        transform=ax1.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
        fontsize=11,
    )

    selected_corrs = list(top_corrs)
    ax2.hist(
        [selected_corrs],
        bins=10,
        alpha=0.7,
        label=["Selected (Model Inputs)"],
        color=["steelblue"],
        density=True,
    )
    ax2.set_xlabel("Correlation Coefficient")
    ax2.set_ylabel("Density")
    ax2.set_title("Model-Input Correlation Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig(Path(out_dir)/"model_input_feature_target_correlations.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ model_input_feature_target_correlations.png")

def plot_mae_learning_curve(cat_model, train_pool, test_pool, out_dir):
    out_dir = Path(out_dir)
    
    train_mae = cat_model.eval_metrics(train_pool, metrics=['MAE'], plot=False)['MAE']
    test_mae = cat_model.eval_metrics(test_pool, metrics=['MAE'], plot=False)['MAE']
    
    epochs = list(range(len(train_mae)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_mae, label='Train MAE', color='steelblue')
    plt.plot(epochs, test_mae, label='Test MAE', color='coral')
    plt.xlabel('Iterations')
    plt.ylabel('MAE')
    plt.title('MAE Learning Curve')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir/'mae_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ mae_learning_curve.png")

def plot_rmse_learning_curve(cat_model, train_pool, test_pool, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_rmse = cat_model.eval_metrics(train_pool, metrics=['RMSE'], plot=False)['RMSE']
    test_rmse = cat_model.eval_metrics(test_pool, metrics=['RMSE'], plot=False)['RMSE']
    
    epochs = list(range(len(train_rmse)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_rmse, label='Train RMSE', color='steelblue')
    plt.plot(epochs, test_rmse, label='Test RMSE', color='coral')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.title('RMSE Learning Curve')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir/'rmse_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ rmse_learning_curve.png")
  
def plot_loss_curve(cat_model, out_dr, metric='RMSE'):
    out_dr = Path(out_dr)
    out_dr.mkdir(parents=True, exist_ok=True)
    results = cat_model.get_evals_result()
    if not results or 'learn' not in results:
        return
    key = metric.lower()
    if key not in results['learn']:
        return
    epochs = range(len(results['learn'][key]))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, results['learn'][key], label=f'Train {metric}', color='steelblue')
    if 'validation' in results and key in results['validation']:
        plt.plot(epochs, results['validation'][key], label=f'Val {metric}', color='coral')
    plt.xlabel('Iteration')
    plt.ylabel(metric)
    plt.title(f'CatBoost {metric} Loss Curve')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    fname = out_dr / f'{key}_loss_curve.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(fname.name)

def plot_residuals(y_true, y_pred, out_dir, split_name):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0,0].scatter(y_pred, residuals, alpha=0.7, s=40, color='steelblue')
    axes[0,0].axhline(0, color='coral', linestyle='--', linewidth=2)
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].set_title('Residuals vs Predicted')
    axes[0,0].grid(True, alpha=0.5)
    
    axes[0,1].hist(residuals, bins=10, color='steelblue', alpha=0.7, edgecolor='coral')
    axes[0,1].set_xlabel('Residuals')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Residuals Distribution')
    axes[0,1].grid(True, alpha=0.5)
    
    stats.probplot(residuals, dist="norm", plot=axes[1,0])
    axes[1,0].get_lines()[0].set_markerfacecolor('steelblue')
    axes[1,0].get_lines()[0].set_markeredgecolor('coral')
    axes[1,0].set_title('Q-Q Plot (Normality)')
    
    axes[1,1].scatter(range(len(residuals)), residuals, alpha=0.7, s=20, color='steelblue')
    axes[1,1].axhline(0, color='coral', linestyle='--', linewidth=2)
    axes[1,1].set_xlabel('Predicted')
    axes[1,1].set_ylabel('Residuals')
    axes[1,1].set_title('Residuals vs Predicted')
    axes[1,1].grid(True, alpha=0.5)
    
    plt.suptitle(f'{split_name} Residuals Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(Path(out_dir)/f'{split_name.lower()}_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {split_name.lower()}_residuals.png")

def plot_true_vs_pred(y_true, y_pred, out_dir, subset_label, metrics):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_true)), y_true, color='steelblue', label='True', s=35, alpha=0.7)
    plt.plot(range(len(y_pred)), y_pred, color='coral', label='Predicted', linewidth=2, alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.title(f'{subset_label} Predictions')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    textstr = f'R²: {metrics["R2"]:.3f}\nAdj R²: {metrics["AdjR2"]:.3f}\nRMSE: {metrics["RMSE"]:.3f}\nMAE: {metrics["MAE"]:.3f}'
    plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(Path(out_dir)/f'{subset_label.lower()}_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {subset_label.lower()}_scatter.png")
    
def print_outlier_analysis(y_true, y_pred, split_name, out_dir, df_deduped=None, orig_indices=None, state_col='State_Name', district_col='State_District_Name'):
    print(f"\n🔍 {split_name} Outlier Analysis")
    print("-"*80)
    
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)
    
    Q1, Q3 = np.percentile(residuals, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    stat_outliers = np.abs(residuals) > np.max(np.abs([lower_bound, upper_bound]))
    stat_outlier_pct = stat_outliers.sum()/len(residuals) * 100
    
    print(f"📊 Statistical Outliers (IQR ±1.5): {stat_outliers.sum():3d}/{len(residuals):3d} ({stat_outlier_pct:4.1f}%)")
    
    worst_idx = np.argsort(abs_residuals)[-5:][::-1]
    print(f"\n🚨 Top 5 CatBoost Regression Model Inference Worst Predictions (Residuals Errors):")
    print(f" {'#':3s} | {'State':15s} | {'District':20s} | {'True':6s} | {'Pred':6s} | {'Error':6s}")
    print("-" * 80)
    
    for i, rel_idx in enumerate(worst_idx):
        orig_idx = int(orig_indices[rel_idx]) if orig_indices is not None else rel_idx
        
        state = "N/A"
        district = "N/A"
        if df_deduped is not None:
            try:
                if 0 <= orig_idx < len(df_deduped):
                    if state_col in df_deduped.columns:
                        state = str(df_deduped.iloc[orig_idx].get(state_col, "N/A"))[:14]
                    if district_col in df_deduped.columns:
                        district = str(df_deduped.iloc[orig_idx].get(district_col, "N/A"))[:19]
            except Exception:
                state, district = "LookupError", "LookupError"
        
        print(f"{i+1:2d}  | {state:15s} | {district:20s} | "
              f"{y_true[rel_idx]:6.1f} | {y_pred[rel_idx]:6.1f} | {abs_residuals[rel_idx]:6.1f}")
    
    outliers_df = pd.DataFrame({
        'relative_idx': np.arange(len(residuals)),
        'original_index': orig_indices if orig_indices is not None else np.arange(len(residuals)),
        'true': y_true, 'pred': y_pred, 'residual': residuals,
        'is_outlier': stat_outliers,
        'abs_error': abs_residuals
    })
    
    if df_deduped is not None:
        states = []
        districts = []
        for i, orig_idx in enumerate(outliers_df['original_index']):
            try:
                idx = int(orig_idx)
                if 0 <= idx < len(df_deduped):
                    states.append(str(df_deduped.iloc[idx].get(state_col, "N/A")))
                    districts.append(str(df_deduped.iloc[idx].get(district_col, "N/A")))
                else:
                    states.append("OutOfBounds")
                    districts.append("OutOfBounds")
            except:
                states.append("Error")
                districts.append("Error")
        outliers_df['state'] = states
        outliers_df['district'] = districts
    outliers_df.to_csv(Path(out_dir)/f'{split_name.lower()}_outliers_summary.csv', index=False)
    print(f"💾 Saved: {split_name.lower()}_outliers_summary.csv")

def save_metrics(train_metrics, test_metrics, out_dir):
    results = pd.DataFrame([train_metrics, test_metrics], index=['Train', 'Test'])
    results.index.name = 'Split'
    results.to_csv(Path(out_dir)/'metrics.csv', float_format='%.4f')
    print("✓ metrics.csv")
    print("-" * 60)
    print(results.round(4)) 
    
def main(args):
    print("=" * 70)
    print("CatBoost Regression Model")
    print("=" * 70)

    df = load_data(args.data)
    original_target = args.target
    df.columns = [re.sub(r'[A-Z]{2,}', lambda m: m.group(0).lower(), col) for col in df.columns]
    df.columns = [col[3:] if len(col) > 3 and col[2] == '_' else col for col in df.columns]
    args.target = find_target_column(df, original_target)

    print(f"Dataset: {df.shape}")
    print(f"Target: {original_target} -> {args.target}")

    id_cols_fixed = [re.sub(r'[A-Z]{2,}', lambda m: m.group(0).lower(), col) for col in args.id_cols]
    df_deduped = df.drop_duplicates()
    print_dataset_stats(df_deduped, args.target)

    if df_deduped[args.target].isnull().any():
        df_deduped = df_deduped.dropna(subset=[args.target])
        print(f"After target cleanup {df_deduped.shape}")

    if args.target not in df_deduped.columns:
        raise ValueError(f"Target {args.target} not found")

    X = df_deduped.drop(columns=[args.target] + [col for col in id_cols_fixed if col in df_deduped.columns])
    y = df_deduped[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, shuffle=True
    )
    train_indices = X_train.index.values
    test_indices = X_test.index.values

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_col = find_state_column(df_deduped, id_cols_fixed)
    print(f"State Column: {state_col or None}")
    if state_col and state_col in df_deduped.columns:
        plot_statewise_histogram(df_deduped, args.target, state_col, out_dir)
        plot_statewise_facets(df_deduped, args.target, state_col, out_dir)

    print("Building preprocessor...")
    preprocessor = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)

    X_train_proc, X_test_proc, feature_names = drop_feature_correlations(X_train_proc, X_test_proc, y_train, feature_names, args.correlation)
    num_features = X_train_proc.shape[1]
    print(f"Features after corr drop: {num_features}")

    print("RFECV Feature Selector")
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    selector = RFECV(estimator=rf, step=0.1, cv=cv, scoring='neg_mean_absolute_error', min_features_to_select=5, verbose=0, n_jobs=-1,)
    selector.fit(X_train_proc, y_train)
    support_mask = selector.support_

    X_train_selected = X_train_proc[:, support_mask]
    X_test_selected = X_test_proc[:, support_mask]
    selected_feature_names = [f for f, keep in zip(feature_names, support_mask) if keep]
    n_selected = X_train_selected.shape[1]
    print(f"RFECV Selected {n_selected} features ({n_selected / num_features * 100:.1f}%)")

    print("Training CatBoost Regression Model...")
    cat_model = CatBoostRegressor(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.random_state,
        verbose=False,
        early_stopping_rounds=args.early_stopping_rounds,
        loss_function='RMSE',
        thread_count=-1,
        task_type='CPU'
    )    

    train_pool = Pool(X_train_selected, y_train)
    test_pool = Pool(X_test_selected, y_test)

    cat_model.fit(
        train_pool, 
        eval_set=test_pool,
        use_best_model=True,
        plot=False
    )

    train_metrics = cat_model.eval_metrics(
        train_pool, 
        metrics=['RMSE', 'MAE'], 
        plot=False
    )
    train_history_rmse = train_metrics['RMSE']   
    train_history_mae = train_metrics['MAE']

    test_metrics = cat_model.eval_metrics(
        test_pool, 
        metrics=['RMSE', 'MAE'], 
        plot=False
    )
    test_history_rmse = test_metrics['RMSE']
    test_history_mae = test_metrics['MAE']
    
    y_train_pred = cat_model.predict(X_train_selected)
    y_test_pred = cat_model.predict(X_test_selected)

    n_train, p = len(y_train), X_train_selected.shape[1]
    n_test = len(y_test)

    train_r2 = r2_score(y_train, y_train_pred)
    train_adj_r2 = calculate_adjusted_r2(train_r2, n_train, p)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)

    test_r2 = r2_score(y_test, y_test_pred)
    test_adj_r2 = calculate_adjusted_r2(test_r2, n_test, p)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_metrics = {"R2": train_r2, "AdjR2": train_adj_r2, "RMSE": train_rmse, "MAE": train_mae}
    test_metrics = {"R2": test_r2, "AdjR2": test_adj_r2, "RMSE": test_rmse, "MAE": test_mae}

    joblib.dump(cat_model, out_dir/"catboost_model.joblib")
    joblib.dump(preprocessor, out_dir/"preprocessor.joblib")
    joblib.dump(selector, out_dir/"rfecv_selector.joblib")
    save_metrics(train_metrics, test_metrics, out_dir)

    print("Generating plots...")
    plot_residuals(y_train, y_train_pred, out_dir, "Train")
    plot_residuals(y_test, y_test_pred, out_dir, "Test")
    plot_true_vs_pred(y_train, y_train_pred, out_dir, "Train", train_metrics)
    plot_true_vs_pred(y_test, y_test_pred, out_dir, "Test", test_metrics)
    importances = cat_model.get_feature_importance()
    plot_feature_importance(importances, selected_feature_names, np.ones(n_selected, bool), out_dir, args.top_n_plot)
    plot_feature_target_correlations(X_train_proc, y_train, feature_names, support_mask, out_dir, args.top_n_plot, args.chunk_size)
    plot_selected_feature_corrs(X_train_selected, y_train, selected_feature_names, out_dir, args.top_n_plot, args.chunk_size)
    plot_loss_curve(cat_model, out_dir, "RMSE")
    plot_loss_curve(cat_model, out_dir, "MAE")
    cv = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
    cv_scores = []
    for train_idx, val_idx in cv.split(X_train_selected):
        X_tr, X_val = X_train_selected[train_idx], X_train_selected[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
        cv_model = clone(cat_model)
        cv_model.set_params(iterations=500, early_stopping_rounds=50)
        tr_pool = Pool(X_tr, y_tr)
        val_pool = Pool(X_val, y_val)
        cv_model.fit(tr_pool, eval_set=val_pool, verbose=False)
        val_pred = cv_model.predict(val_pool)
        cv_scores.append(r2_score(y_val, val_pred))
    print(f"5-fold CV R²: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    plot_mae_learning_curve(cat_model, train_pool, test_pool, out_dir)
    plot_rmse_learning_curve(cat_model, train_pool, test_pool, out_dir)
    print_outlier_analysis(y_train, y_train_pred, "Train", out_dir, df_deduped, train_indices)
    print_outlier_analysis(y_test, y_test_pred, "Test", out_dir, df_deduped, test_indices)

    print("=" * 70)
    print("CatBoost Model Results")
    print("=" * 70)
    print(f"Test R2 {test_r2:.4f} Adj R2 {test_adj_r2:.4f} RMSE {test_rmse:.4f} MAE {test_mae:.4f}")
    print(f"Features {n_selected}/{len(feature_names)} Pre/Post RFECV")
    print(f"Outputs {out_dir}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CatBoost Regression Model - Machine learning for infant mortality rate prediction.")
    parser.add_argument("--data", required=True, help="Path to dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--id-cols", nargs="+", default=[], help="ID columns to exclude")
    parser.add_argument("--cv-folds", type=int, default=7, help="KFold CV folds")
    parser.add_argument("--correlation", type=float, default=71.0, help="Drop correlated features by pct")
    parser.add_argument("--top-n-plot", type=int, default=20, help="Top N for plots")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Correlation chunk size")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out_dir", default="../notebooks/model-tests/catboost-test-1")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--l2-leaf-reg", type=float, default=70.0)
    parser.add_argument("--min-child-weight", type=float, default=50.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=300)
    args = parser.parse_args()
    main(args)