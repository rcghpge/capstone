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
warnings.filterwarnings("ignore")
import re
import sys
import time
import json
import joblib
import logging
import argparse
import numpy as np
import pandas as pd
from PyALE import ale
import seaborn as sns
from scipy import stats
from pathlib import Path
from rulefit import RuleFit
from sklearn.base import clone
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from contextlib import contextmanager
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.utils.parallel import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, learning_curve, cross_val_score
"""
Example usage: See Jupyter Notebooks for more information.

# Basic model run
!python xgboost_regression.py \
--data ../data/Key_indicator_districtwise.csv \
--target "Infant_Mortality_Rate_Imr_Total_Person" \
--id-cols "State_Name" "State_District_Name" \
--correlation 70 \
--test-size 0.15 \
--random-state 42 \
--outdir artifacts/xgboost

# Hyperparameter Tuning
!python xgboost_regression.py \
--data ../data/Key_indicator_districtwise.csv \
--target "Infant_Mortality_Rate_Imr_Total_Person" \
--id-cols "State_Name" "State_District_Name" \
--correlation 35 --test-size 0.20 --random-state 42 \
--eta 0.15 --n-estimators 3000 --early-stopping-rounds 100 --max-depth 4 \
--colsample-bytree 0.7 --subsample 0.7 --reg-lambda 2 --reg-alpha 1 \
--min-child-weight 25 --top-n-plot 10 --chunk-size 50000 --outdir artifacts/xgboost
"""
sns.set_palette('husl')
plt.style.use('default')

logging.basicConfig(
    level=logging.DEBUG if '--debug' in sys.argv else logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('xgboost_regression.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

TITLE = "XGBoost Regression Model"
@contextmanager
def spinner_progress(total_steps=64, spinners='|/-\r'):
    start = time.time()
    itercount = 0
    def update(n_features):
        nonlocal itercount
        itercount += 1
        elapsed = time.time() - start
        eta = elapsed * (total_steps - itercount)/itercount * 60 if itercount > 0 else 0
        pct = min(100, itercount/total_steps * 100)
        spinner = spinners[itercount % 4]
        sys.stdout.write(f'\r{spinner} [{itercount:3d}/{total_steps}] {n_features:4d} feats | ETA {eta:.0f}m ({pct:3.0f}%)')
        sys.stdout.flush()
    yield update
    elapsed = time.time() - start
    print(f'\n✅ RFECV Feature Selection Complete! {elapsed/60:.1f}m total')

def load_data(datapath):
    if datapath.endswith('.csv'):
        return pd.read_csv(datapath)
    elif datapath.endswith('.parquet'):
        return pd.read_parquet(datapath)
    raise ValueError("Unsupported file type. Use .csv or .parquet")

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

def print_pre_rfecv_stats(X_processed, y_train, feature_names, num_features):
    print("\n" + "="*80)
    print("📈 Pre-RFECV Feature Selection Summary (Post-Preprocessing + Feature Correlation Drop)")
    print('='*80)
    
    n_samples, n_features = X_processed.shape
    print(f"📊 Processed Dataset: {n_samples:,} samples, {n_features:,} features")
    print(f"🎯 Target Samples: {len(y_train):,}")
    print(f"📋 Feature Name Count: {len(feature_names):,}")
    
    total_missing = np.isnan(X_processed).sum()
    missing_pct = total_missing/(n_samples * n_features) * 100
    print(f"🔍 Post-Preprocessing Missing Null/NaN Values: {total_missing:,} ({missing_pct:.2f}%)")
    
    if total_missing == 0:
        print("✅ No missing Null/NaN values after preprocessing!")
    else:
        print("⚠️ Warning: Missing Null/NaN values persist after preprocessing!")
    
    y_train = np.asarray(y_train).ravel()
    target_missing = np.isnan(y_train).sum()
    print(f"🎯 Target Missing Null/NaN Values: {target_missing:,}/{len(y_train):,} ({target_missing/len(y_train)*100:.1f}%)")
    
    numeric_feats = sum(1 for name in feature_names if any(c.isdigit() or c in '.-' for c in name))
    ohe_features = sum(1 for name in feature_names if '__' in name)
    
    print("\n🔧 Feature Breakdown:")
    print(f"📊 Numerical Features: {numeric_feats}/{len(feature_names)}")
    print(f"🅰️ Categorical (Post-OHE): {len(feature_names) - numeric_feats}")
    print(f"🔄 OHE Features Generated: {ohe_features}")
    
    if n_features > 0:
        corrs = np.corrcoef(X_processed.T, y_train)[-1, :-1]
        abs_corrs = np.abs(corrs)
        top5_idx = np.argsort(abs_corrs)[-5:][::-1]
        print("\n📊 Top 5 |corr| Features With Target Variable")
        print("-" * 60)
        for i in top5_idx:
            print(f"{str(feature_names[i])[:40]:40s} | {abs_corrs[i]:.4f}")
    print('='*80)

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

def calculate_adjusted_r2(r2_score, n_samples, n_features):
    if n_samples <= n_features + 1:
        return np.nan
    return 1 - (1 - r2_score) * (n_samples - 1)/(n_samples - n_features - 1)

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', RobustScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer([('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])

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
    plt.title('Random Forest RFECV Feature Importance (Inputs to XGBoost Model)')
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
    ax1.set_title("Top XGBoost Feature-Target Correlations Selected", fontsize=14, weight="bold")
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
        f"Top XGBoost Model-Input Feature-Target Correlations (Selected {n_features})",
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
        f"XGBoost Model Input (RFECV): {selected_count}/{n_features}",
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

def plot_learning_curve(estimator, X_df, y, out_dir, cv=5):
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    est = clone(estimator)
    est.early_stopping_rounds = None 
    est.eval_metric = 'rmse' 
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, val_scores = learning_curve(
        est, X_df, y, train_sizes=train_sizes, cv=cv, scoring='r2', 
        n_jobs=1, 
        shuffle=True, random_state=42, error_score='raise'
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
    plt.title('XGBoost + RFECV Regression Model Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir/'learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ learning_curve.png")
    
    print(f"✓ Final CV R²: {np.mean(val_scores[-1]):.4f} ± {np.std(val_scores[-1]):.4f}")
    print(f"✓ Final Train R²: {np.mean(train_scores[-1]):.4f} ± {np.std(train_scores[-1]):.4f}")

def plot_loss_curve(xgb_model, out_dir, metric='rmse'):
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = xgb_model.evals_result()
    if not results:
        print("No evals_result found on model; did you pass eval_set and eval_metric?")
        return

    train_key = 'validation_0'
    metrics_dict = results.get(train_key, {})
    if metric not in metrics_dict:
        print(f"Metric '{metric}' not found in evals_result for {train_key}. "
              f"Available: {list(metrics_dict.keys())}")
        return

    train_metric = metrics_dict[metric]
    epochs = range(len(train_metric))

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        train_metric,              
        label=f'{train_key} {metric.upper()}',
        color='steelblue'            
    )

    test_key = 'validation_1'
    if test_key in results and metric in results[test_key]:
        test_metric = results[test_key][metric]
        plt.plot(
            epochs,
            test_metric,                
            label=f'{test_key} {metric.upper()}',
            color='coral'            
        )

    plt.xlabel('Boosting round')
    plt.ylabel(metric.upper())
    plt.title(f'XGBoost + RFECV Regression Model {metric.upper()} Loss Curve')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()

    fname = out_dir/f'{metric.lower()}_loss_curve.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {fname.name}")

def plot_ale_all_features(model, X_train_selected, selected_feature_names, out_dir, grid_size=30, max_features=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X_df = pd.DataFrame(X_train_selected, columns=selected_feature_names)
    n_features = X_df.shape[1]
    if max_features is not None:
        n_features = min(n_features, max_features)
    
    logging.getLogger('PyALE._ALE_generic').setLevel(logging.WARNING)
    
    success_count = 0
    for i in range(n_features):
        feat_name = selected_feature_names[i]
        try:
            plt.figure(figsize=(8, 6))
            exp = ale(X=X_df, model=model, feature=[feat_name], grid_size=grid_size) 
            plt.title(f'ALE {feat_name}')
            plt.xlabel(feat_name)
            plt.ylabel('ALE')
            plt.grid(True, alpha=0.5)
            fname = out_dir/f'ale_{feat_name.replace(" ", "_").replace("/", "_").replace(",", "_")}.png'
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            success_count += 1
        except Exception as e:
            print(f"Failed to plot ALE for {feat_name}: {e}")
    
    print(f"✓ Generated {success_count}/{n_features} ALE plots successfully.")

def plot_rulefit(X_train_selected, y_train, selected_feature_names, out_dir, n_top_rules=30, min_importance=0.01):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_np = np.asarray(X_train_selected) 
    y_np = np.asarray(y_train).ravel()
    
    rf = RuleFit(max_rules=n_top_rules, random_state=42)
    rf.fit(X_np, y_np)

    rules = rf.get_rules()
    rules_df = rules[rules['importance'] > min_importance].copy()
    rules_df = rules_df.sort_values('importance', ascending=False).head(n_top_rules)
    
    def enrich_rules(rules_df, feature_names):
        enriched = []
        for _, row in rules_df.iterrows():
            rule_str = row['rule']
            for i, fname in enumerate(feature_names):
                rule_str = rule_str.replace(f'x[{i}]', fname)
            enriched.append({**row.to_dict(), 'rule_enriched': rule_str})
        return pd.DataFrame(enriched)
    
    rules_enriched = enrich_rules(rules_df, selected_feature_names)
    rules_csv = out_dir/"rulefit_rules.csv"
    rules_enriched[['rule_enriched', 'type', 'coef', 'support', 'importance']].to_csv(rules_csv, index=False)
    print(f"✓ {rules_csv.name}")

    if len(rules_df) > 0:
        plt.figure(figsize=(12, max(8, 0.4 * min(15, len(rules_df)))))
        top_rules = rules_df.head(15)
        colors = ['steelblue' if t == 'linear' else 'coral' for t in top_rules['type']]
        plt.barh(range(len(top_rules)), top_rules['importance'], color=colors)
        plt.yticks(range(len(top_rules)), [r[:40] + '...' if len(r)>40 else r for r in top_rules['rule']])
        plt.gca().invert_yaxis()
        plt.xlabel('Importance')
        plt.title('Top RuleFit Rules (XGBoost Trees → Interpretable Rules)')
        plt.tight_layout()
        rules_png = out_dir/"rulefit_importance.png"
        plt.savefig(rules_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ {rules_png.name}")
    else:
        print("No rules above importance threshold.")

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
    print(f"\n🚨 Top 5 KNN Regression Model Inference Worst Predictions (Residuals Errors):")
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
    print(f"{TITLE}")
    print('=' * 70)
    df = load_data(args.data)
    original_target = args.target
    df.columns = [re.sub(r'[A-Z]{2,}', lambda m: m.group(0).lower(), col) for col in df.columns]
    df.columns = [col[3:] if len(col) > 3 and col[2] == '_' else col for col in df.columns]
    args.target = find_target_column(df, original_target)

    print(f"📊 Dataset: {df.shape}")
    print(f"🎯 Target: {original_target} -> {args.target}")
    id_cols_fixed = [re.sub(r'[A-Z]{2,}', lambda m: m.group(0).lower(), col) for col in args.id_cols]

    len_df = len(df)
    mask = ~df.duplicated()
    df_deduped = df[mask].copy()
    print(f"🧹 Deduplicating: {len_df:,} -> {len(df_deduped):,} samples")
    print_dataset_stats(df_deduped, args.target)

    if df_deduped[args.target].isnull().any():
        print("❌ ERROR: Target has missing values. Dropping incomplete rows.")
        df_deduped = df_deduped.dropna(subset=[args.target])
        print(f"📊 After target cleanup: {df_deduped.shape}")
    print(f"✅ Raw dataset for preprocessing ({df_deduped.shape})")

    if args.target not in df_deduped.columns:
        raise ValueError(f"Target '{args.target}' not found")

    X = df_deduped.drop(columns=[args.target])
    for col in id_cols_fixed:
        if col in X.columns:
            X = X.drop(columns=[col])
    y = df_deduped[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, shuffle=True
    )
    train_indices = X_train.index.values
    test_indices = X_test.index.values

    out_dir = Path(args.outdir)
    out_dir.mkdir(exist_ok=True, parents=True)

    state_col = find_state_column(df_deduped, id_cols_fixed)
    print(f"🗺️ State Column: {state_col or 'None'}")
    if state_col and state_col in df_deduped.columns:
        plot_statewise_histogram(df_deduped, args.target, state_col, out_dir)
        plot_statewise_facets(df_deduped, args.target, state_col, out_dir)
 
    print("\n🔧 Building preprocessor...")
    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)
    feature_mapping = pd.DataFrame({
        'pre_rfecv_index': range(len(feature_names)),
        'pre_rfecv_name': feature_names,
        'original_approx': [name.split('__')[-1].replace('_imputer', '').replace('_scaler', '') if '__' in name else name for name in feature_names]
    })
    feature_mapping.to_csv(out_dir/'feature_mapping_pre_rfecv.csv', index=False)
    print(f"✓ Saved pre-RFECV mapping: {len(feature_names)} features")

    X_train_processed, X_test_processed, feature_names = drop_feature_correlations(
        X_train_processed, X_test_processed, y_train, feature_names, args.correlation
    )

    num_features = X_train_processed.shape[1]
    target_min, target_max = y.min(), y.max()
    print(f"✓ Features: {num_features:,} | Target Range: {target_min:.1f}-{target_max:.1f}")
    print_pre_rfecv_stats(X_train_processed, y_train, feature_names, num_features)

    print("🔍 RFECV Feature Selector")
    print("🔍 Running a pass on the data..")
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    total_features = X_train_processed.shape[1]

    def quiet_worker():
        import warnings
        from joblib import Parallel, delayed
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
    quiet_worker()
    
    with spinner_progress(min(64, max(1, total_features//10))):
        selector = RFECV(estimator=rf, step=10, cv=cv, scoring='neg_mean_absolute_error', min_features_to_select=5, verbose=0, n_jobs=-1)
        print("🔍 X_train Shape:", X_train.shape)
        print("🔍 X_train_processed Shape:", X_train_processed.shape)
        print("🔍 First 5 Original Features:", list(X_train.columns[:5]))
        print(f"✅ RFECV Input: {X_train_processed.shape[1]} features")
        print(f"✅ Original Numeric Columns: {len(X_train.select_dtypes(include=np.number).columns)} features")
        
        selector.fit(X_train_processed, y_train)
        print(f"✅ RFECV: {X_train_processed.shape[1]} → {selector.n_features_} features selected")
    
    support_mask = selector.support_
    X_train_selected = X_train_processed[:, support_mask]
    X_test_selected  = X_test_processed[:, support_mask]
    selected_feature_names = [f for f, keep in zip(feature_names, support_mask) if keep]
    n_selected = X_train_selected.shape[1]

    print(f"✅ RFECV Selected: {n_selected} features "
          f"({n_selected/X_train_processed.shape[1] * 100:.1f}%)")
    print(f"✅ Features: {n_selected}/{len(feature_names)}")
    print(f"✅ RFECV Ranking score: {selector.ranking_[np.argsort(selector.ranking_)[:selector.n_features_]].mean():.3f}")

    print("\n📊 Training XGBoost Model...")
    xgb_model = XGBRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.eta,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
        objective='reg:squarederror',
        eval_metric=['rmse', 'mae'],
        random_state=args.random_state,
        verbosity=0,
        n_jobs=-1,
        early_stopping_rounds=args.early_stopping_rounds,
        tree_method="hist",
        device="cuda"
    )

    xgb_model.fit(
        X_train_selected,
        y_train,
        eval_set=[(X_train_selected, y_train), (X_test_selected, y_test)],
        verbose=False
    )
    print(f"✓ RFECV selected {selector.n_features_} features")
    print(f"✓ RFECV rank 1 indices: {np.where(selector.ranking_ == 1)[0][:10]}") 
    
    y_train_pred = xgb_model.predict(X_train_selected)
    y_test_pred = xgb_model.predict(X_test_selected)

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

    train_metrics = {'R2': train_r2, 'AdjR2': train_adj_r2, 'RMSE': train_rmse, 'MAE': train_mae}
    test_metrics = {'R2': test_r2, 'AdjR2': test_adj_r2, 'RMSE': test_rmse, 'MAE': test_mae}

    joblib.dump(xgb_model, out_dir/'xgb_model.joblib')
    joblib.dump(preprocessor, out_dir/'preprocessor.joblib')
    joblib.dump(selector, out_dir/'rfecv_selector.joblib')
    save_metrics(train_metrics, test_metrics, out_dir)

    print("\n📊 Generating model inference plots...")
    plot_residuals(y_train, y_train_pred, out_dir, 'Train')
    plot_residuals(y_test, y_test_pred, out_dir, 'Test')
    plot_true_vs_pred(y_train, y_train_pred, out_dir, 'Train', train_metrics)
    plot_true_vs_pred(y_test, y_test_pred, out_dir, 'Test', test_metrics)
    plot_feature_importance(selector.estimator_.feature_importances_, feature_names, selector.support_, out_dir, top_n=args.top_n_plot)
    plot_feature_target_correlations(X_train_processed, y_train, feature_names, selector.support_, out_dir, top_n=args.top_n_plot, chunk_size=args.chunk_size)
    plot_selected_feature_corrs(X_train_selected, y_train, selected_feature_names, out_dir, top_n=args.top_n_plot, chunk_size=args.chunk_size)

    print("\n📊 Generating model assessment plots...")
    plot_loss_curve(xgb_model, out_dir, metric='mae')
    plot_loss_curve(xgb_model, out_dir, metric='rmse')
    cv_model = clone(xgb_model)
    cv_model.set_params(early_stopping_rounds=None)
    cv_scores = cross_val_score(cv_model, X_train_selected, y_train, cv=5, scoring='r2')
    print(f"✓ Check {args.cv_folds}‑fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    plot_learning_curve(xgb_model, X_train_selected, y_train.values.ravel(), out_dir, cv=5)
    plot_ale_all_features(xgb_model, X_train_selected, selected_feature_names, out_dir, grid_size=30, max_features=20)
    plot_rulefit(X_train_selected, y_train, selected_feature_names, out_dir, n_top_rules=30, min_importance=0.01)
    print_outlier_analysis(y_train, y_train_pred, "Train", out_dir, df_deduped=df_deduped, orig_indices=train_indices, state_col='State_Name', district_col='State_District_Name')
    print_outlier_analysis(y_test, y_test_pred, "Test", out_dir, df_deduped=df_deduped, orig_indices=test_indices, state_col='State_Name', district_col='State_District_Name')
    print("\n" + "="*70)
    print("✅ XGBoost Model Results")
    print('=' * 70)
    print(f"🎯 Test: R²={test_r2:.4f} | Adj R²={test_adj_r2:.4f} | RMSE={test_rmse:.4f} | MAE={test_mae:.4f}")
    print(f"📊 Features: {n_selected}/{len(feature_names)} ← Pre/post RFECV")
    print(f"📁 Outputs: {out_dir}")
    print('=' * 70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'{TITLE} - Machine learning for infant mortality rate prediction')
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--target', required=True, help='Target column name')
    parser.add_argument('--id-cols', nargs='+', default=[], help='ID columns to exclude')
    parser.add_argument('--cv-folds', type=int, default=5, help='KFold/RFECV cross-validation folds')
    parser.add_argument('--correlation', type=float, default=70.0, help='Drop features by pct correlation with target')
    parser.add_argument('--top-n-plot', type=int, default=25, help='Top N features for correlation/importance plots')
    parser.add_argument('--chunk-size', type=int, default=5000, help='Chunk size for correlation computation (memory)')
    parser.add_argument('--test-size', type=float, default=0.25)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--outdir', default='artifacts/xgb')
    parser.add_argument('--n-estimators', type=int, default=2000)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample-bytree', type=float, default=0.8)
    parser.add_argument('--reg-alpha', type=float, default=0.1)
    parser.add_argument('--reg-lambda', type=float, default=1.0)
    parser.add_argument('--min-child-weight', type=float, default=5.0)
    parser.add_argument('--eta', type=float, default=0.1, help='XGBoost learning rate')
    parser.add_argument('--early-stopping-rounds', type=int, default=50, help='Early stopping rounds')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)