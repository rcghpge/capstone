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
import jax
os.environ["KERAS_BACKEND"] = "jax"  # add "tensorflow" to build if no dep conflicts
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
jax.config.update('jax_platform_name', 'cpu')
import re
import sys
import time
import json
import shap
import lime
import keras
import joblib
import logging
import warnings
import argparse
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import lime.lime_tabular
from pathlib import Path
from tqdm.auto import tqdm
from scipy.stats import norm
from sklearn.base import clone
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from typing import Union, Optional
from collections import namedtuple
import matplotlib.colors as mcolors
from keras import layers, callbacks
from keras.models import Sequential
from matplotlib.patches import Patch
from contextlib import contextmanager
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.compose import ColumnTransformer
from statsmodels.tools.tools import add_constant
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.parallel import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, learning_curve, validation_curve
"""
Example Usage:
python keras_dnn.py --data ../data/Key_indicator_districtwise.csv \
--target Infant_Mortality_Rate_Imr_Total_Person --id-cols State_Name State_District_Name \
--lr 0.0009 --dropout 0.15 --l2-reg 0.01 --epochs 500 --correlation 72 --vif-threshold 10 \
--test-size 0.25 --val-size 0.10 --random-state 42 --outdir keras-dnn-final-test
"""
sns.set_palette("husl")
plt.style.use('default')

def setup_logging(out_dir: str, debug: bool):
    log_path = Path(out_dir)/"logs"
    log_path.mkdir(parents=True, exist_ok=True)

    console_level = logging.DEBUG if debug else logging.WARNING
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path/"global_debug.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler],
        force=True,
    )

    logging.captureWarnings(True)
    logging.getLogger("shap").setLevel(logging.ERROR)

    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.setLevel(logging.WARNING)
    warnings_logger.handlers.clear()

    warn_file_handler = logging.FileHandler(log_path/"warnings.log")
    warn_file_handler.setLevel(logging.WARNING)
    warn_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    warn_console_handler = logging.StreamHandler(sys.stdout)
    warn_console_handler.setLevel(console_level)
    warn_console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    warnings_logger.addHandler(warn_file_handler)
    warnings_logger.addHandler(warn_console_handler)
    warnings_logger.propagate = False

    for logger_name in ["jax", "keras"]:
        lib_logger = logging.getLogger(logger_name)
        lib_logger.setLevel(logging.DEBUG if debug else logging.WARNING)
        lib_logger.handlers.clear()

        debug_handler = logging.FileHandler(log_path/f"{logger_name}_debug.log")
        debug_handler.setLevel(logging.DEBUG if debug else logging.WARNING)
        debug_handler.setFormatter(formatter)
        lib_logger.addHandler(debug_handler)

        if debug:
            lib_console_handler = logging.StreamHandler(sys.stdout)
            lib_console_handler.setLevel(logging.DEBUG)
            lib_console_handler.setFormatter(formatter)
            lib_logger.addHandler(lib_console_handler)

        lib_logger.propagate = False

    if debug:
        os.environ["JAX_LOGGING_LEVEL"] = "DEBUG"

    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete (debug=%s)", debug)
    return logger

def ensure_subdir(base_dir: Path, *subpaths: str) -> Path:
    full_path = base_dir.joinpath(*subpaths)
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path

@contextmanager
def feature_selection(total_steps=64):
    logger = logging.getLogger(__name__)
    spinners = ['|', '/', '-', '\\']
    start = time.time()
    iter_count = 0

    def update(n_features):
        nonlocal iter_count
        iter_count += 1
        elapsed = time.time() - start
        eta = (elapsed/iter_count)*(total_steps-iter_count)/60 if iter_count>0 else 0
        pct = min(100, (iter_count/total_steps)*100)
        spinner = spinners[iter_count%4]
        sys.stdout.write(f'\r[{spinner} {iter_count:3d}/{total_steps}] {n_features:4d} feats | ETA: {eta:.0f}m | {pct:3.0f}% ')
        sys.stdout.flush()

    yield update
    elapsed = time.time() - start
    logger.info(f'✅ Feature Selection Complete! {elapsed/60:.1f}m total')

def load_data(data_path):
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        return pd.read_parquet(data_path)
    raise ValueError('Unsupported file type. Use .csv or .parquet')

def check_data_imputations(X_train, X_test, preprocessor):
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    print("-"*80)
    print("📊 Data Imputation Report:")
    print(f"✅ Shape Gain: {X_train.shape[1]} → {X_train_proc.shape[1]} (+{X_train_proc.shape[1]-X_train.shape[1]} feats)")
    print(f"✅ Missing Train: {np.isnan(X_train_proc).sum()} | Test: {np.isnan(X_test_proc).sum()}")
    print(f"✅ Variance Collapse: {(X_train_proc.std(axis=0)==0).sum()} Constant Feature Columns")

    if np.isnan(X_test_proc).any():
        raise ValueError("Leakage: test NaNs post-transform!")

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    print("📊 Verify Distinct Categorical Feature Columns:")
    for col in cat_cols:
        n_unique = X[col].nunique()
        print(f"  {col}: {n_unique} unique → {n_unique} OHE cols")

    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)), ('scaler', RobustScaler())])
    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True)), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    print(f"✅ Numeric Columns: {len(num_cols)}")
    print(f"✅ Categorical Columns: {len(cat_cols)}")
    return ColumnTransformer([('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])

def preprocessor_checkpoint(X_train, preprocessor):
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("ignore")
        with np.errstate(divide='ignore', invalid='ignore'):
            X_proc = preprocessor.fit_transform(X_train)

        for w in warning_list:
            if "Skipping features without any observed values" in str(w.message):
                skipped_str = str(w.message).split("['")[1].split("'].")[0]
                skipped_features = [f.strip().strip("'") for f in skipped_str.split("', '")]
                print(f"[INFO] Skipping all-NaN features: {skipped_features}")
                break

    return X_proc

def build_keras_dnn(input_dim, learning_rate=0.0009, dropout_rate=0.15, l2_reg=0.01):
    dnn = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dense(512, activation='swish', kernel_regularizer=keras.regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        layers.Dense(256, activation='swish', kernel_regularizer=keras.regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        layers.Dense(128, activation='swish', kernel_regularizer=keras.regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        layers.Dense(64, activation='swish'),
        layers.Dropout(dropout_rate/2),
        layers.Dense(1, activation='linear')
    ])
    dnn.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='huber', metrics=['mae'])
    return dnn

def train_keras_dnn(dnn, X_train, y_train, X_val, y_val, out_dir, epochs=500, batch_size=32, patience=20):
    callbacks_list = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=500, min_lr=1e-7, verbose=0, mode='min'),
        keras.callbacks.ModelCheckpoint(str(out_dir/'keras_best_dnn.keras'), monitor='val_loss', save_best_only=True, verbose=0)
    ]

    history = dnn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks_list, verbose=0)
    return dnn, history

def plot_training_summary(history, out_dir):
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(Path(out_dir)/'training_summary.csv', index=False)
    print(f"📊 Saved {len(history_df)} epochs")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0,0].plot(history.history['loss'], label='Train MSE', color='steelblue', lw=2)
    axes[0,0].plot(history.history['val_loss'], label='Val MSE', color='coral', lw=2)
    axes[0,0].set_title('MSE Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('MSE')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.7)
    axes[0,1].plot(history.history['mae'], label='Train MAE', color='steelblue', lw=2)
    axes[0,1].plot(history.history['val_mae'], label='Val MAE', color='coral', lw=2)
    axes[0,1].set_title('MAE Loss')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('MAE')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.7)

    if 'lr' in history.history:
        axes[1,0].semilogy(history.history['lr'], color='coral', lw=2)
        axes[1,0].set_title('Learning Rate'); axes[1,0].grid(True, alpha=0.7)
    else:
        best_epoch = np.argmin(history.history['val_loss'])
        axes[1,0].text(0.37, 0.6, f'EarlyStopping Active\nBest Epoch: {best_epoch}\nVal Loss: {history.history["val_loss"][best_epoch]:.3f}',
                       transform=axes[1,0].transAxes, fontsize=12, bbox=dict(boxstyle="round, pad=0.3", facecolor="coral", alpha=0.7), ha='left', va='center')
        axes[1,0].set_title('Training Dynamics', pad=0, y=0.85)
        axes[1,0].axis('off')

    axes[1,1].axis('off')
    metrics = ['Final MSE', 'Final MAE']
    train_vals = [f"{history.history['loss'][-1]:.3f}", f"{history.history['mae'][-1]:.3f}"]
    val_vals = [f"{history.history['val_loss'][-1]:.3f}", f"{history.history['val_mae'][-1]:.3f}"]

    table = axes[1,1].table(cellText=np.array([train_vals, val_vals]).T,
                            colLabels=['Train', 'Val'],
                            rowLabels=metrics,
                            cellLoc='center', loc='center',
                            bbox=[0.15, 0.5, 0.7, 0.25],
                            colColours=['#f0f8ff']*2,
                            rowColours=['lightcoral']*2)

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0)

    for i in range(2):
        for j in range(2):
            table[(i+1, j)].set_text_props(weight='normal', ha='center')
            table[(i+1, j)].set_facecolor('#f8f9fa')
        table[(i+1, -1)].set_text_props(weight='normal', ha='left')

    axes[1,1].set_title('Final Metrics', weight='normal', pad=0, y=0.85)
    axes[1,1].axis('off')

    plt.suptitle('Keras DNN Train Summary', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(out_dir/'keras_training_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ keras_training_summary.png")

def get_feature_names(preprocessor):
    feature_names = list(preprocessor.get_feature_names_out())
    return [name.split('__', 1)[1] if '__' in name else name for name in feature_names]

def drop_feature_correlations(X_train, X_test, y_train, feature_names, drop_pct):
    drop_pct = float(max(0.0, min(100.0, drop_pct)))
    if drop_pct <= 0:
        return X_train, X_test, feature_names
    y_arr = np.asarray(y_train).ravel()
    n_features = X_train.shape[1]
    if n_features <= 1:
        return X_train, X_test, feature_names
    with np.errstate(divide='ignore', invalid='ignore'):
        corrs = np.corrcoef(X_train.T, y_arr)[-1, :-1]

    abs_corrs = np.abs(corrs)
    n_drop = int(round(n_features * drop_pct/100.0))
    if n_drop <= 0 or n_drop >= n_features:
        return X_train, X_test, feature_names
    keep_idx = np.argsort(abs_corrs)[n_drop:]
    X_train_new = X_train[:, keep_idx]
    X_test_new = X_test[:, keep_idx]
    feature_names_new = [feature_names[i] for i in keep_idx]
    print(f"✅ Dropped {n_drop}/{n_features} features ({drop_pct:.2f}%) by |corr|; {len(feature_names_new)} remain.")
    return X_train_new, X_test_new, feature_names_new

def feature_redundancy_checks(X_proc, feature_names):
    medians = []
    for i, name in enumerate(feature_names):
        col = X_proc[:,i]
        if np.std(col) < 1e-8:
            median_val = np.median(col)
            pct_missing_raw = 1 - (np.abs(col - median_val) > 1e-8).mean()
            medians.append((name, pct_missing_raw))

    df_med = pd.DataFrame(medians, columns=['feature', 'raw_missing_pct'])
    print("-"*80)
    print("📊 Feature Redundancy Checks:")
    print(df_med[df_med.raw_missing_pct > 0.95].sort_values('raw_missing_pct'))

def find_state_columns(df, id_cols):
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

def find_target_columns(df, target_name):
    target_candidates = [col for col in df.columns if target_name.lower() in col.lower()]
    return target_candidates[0] if target_candidates else target_name.lower()

def print_dataset_stats(df, target_col):
    print("="*80)
    print("📈 Raw Dataset Summary")
    print("="*80)
    total_samples, total_features = df.shape
    print(f"📊 Dataset Shape:({total_samples}, {total_features})")
    print(f"🎯 Target Column:'{target_col}'")
    total_missing = df.isnull().sum().sum()
    missing_pct = (total_missing/(total_samples*total_features))*100
    print(f"🔍 Total Missing Null/NaN Values:{total_missing:,} ({missing_pct:.2f}%)")
    print("-"*80)

def print_preprocessing_stats(X_processed, y_train, feature_names, num_features):
    print("="*80)
    print("📊 Feature Selection Summary")
    print("="*80)

    n_samples, n_features = X_processed.shape
    print(f"📊 Processed Dataset:({n_samples}, {n_features})")
    print(f"🎯 Target Samples:{len(y_train)}")
    print(f"📋 Feature Names Available:{len(feature_names)}")

    total_missing = np.isnan(X_processed).sum()
    print(f"🔍 Data Preprocessing Missing Values:{total_missing:,}")
    if total_missing == 0:
        print("✅ No Missing Null/NaN Values!")

    post_ohe_df = pd.DataFrame(X_processed, columns=feature_names[:n_features])
    unique_counts = post_ohe_df.nunique()
    high_card_num = sum(unique_counts > 20)
    low_card_num = sum((unique_counts > 2) & (unique_counts <= 20))
    binary_flags = sum(unique_counts <= 2)
    constant = sum(unique_counts == 1)
    ohe_generated = sum('__' in name for name in feature_names[:n_features])
    missing_indicators = sum('missingindicator' in name.lower() for name in feature_names[:n_features])
    print("-"*80)
    print(f"🔧 Feature Breakdown:")
    print(f"📊 Continuous Numeric:{high_card_num}")
    print(f"🔢 Discrete Numeric:{low_card_num}")
    print(f"🔢 Binary Flags/OHE:{binary_flags}")
    print(f"📈 Constant (Feature(s) Are Dropped):{constant}")
    print(f"🔄 OHE Generated:{ohe_generated}")
    print(f"📊 Total Active:{high_card_num + low_card_num + binary_flags - constant}")
    print("-"*80)
    print(f"📋 10 Lowest Cardinality Features:")
    print(unique_counts.sort_values().head(10))
    print("-"*80)
    print(f"📋 10 Highest Cardinality Features:")
    print(unique_counts.sort_values(ascending=False).head(10))

    sparsity = (X_processed == 0).sum()/X_processed.size*100
    print("-"*80)
    print(f"📊 Data Quality:")
    print(f"📋 Sparsity (% Zeros):{sparsity:.1f}%")
    print(f"🎯 Feature To Target Ratio:{n_features/len(y_train):.2f}")
    print("="*80)
    print("📊 Feature Selection")
    print("="*80)
    print(f"✅ Feature Selection:{n_features} → Will look to select the best features")
    print(f"🔍 Running a pass on the data with RFECV + VIF (this can take a bit)...")
    print("-"*80)

def plot_feature_importances(importances, feature_names, selector_support, out_dir, top_n=10):
    top_n = min(top_n, len(importances))
    idx = np.argsort(importances)[-top_n:][::-1]
    colors = ['steelblue' if selector_support[i] else 'coral' for i in idx]

    plt.figure(figsize=(12, 10))
    bars = plt.barh(range(top_n), importances[idx], color=colors, alpha=0.7)
    plt.yticks(range(top_n), [feature_names[i][:35] + '...' if len(feature_names[i]) > 35 else feature_names[i] for i in idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Estimator Importances')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.7)

    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='Selected'),
        Patch(facecolor='coral', alpha=0.7, label='Not Selected'),
        Patch(facecolor='coral', alpha=0.7, label='If Not Selected = Keeps All Features')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(Path(out_dir)/'feature_estimator_importances.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ feature_estimator_importances.png")

def plot_feature_target_correlations(X_processed, y_train, feature_names, selector_support, out_dir, top_n=10):
    try:
        X = np.asarray(X_processed)
        y = np.asarray(y_train).ravel()
        n_features = X.shape[1]

        if len(feature_names) > n_features:
            feature_names = feature_names[:n_features]
        if len(selector_support) > n_features:
            selector_support = selector_support[:n_features]
        elif len(selector_support) < n_features:
            selector_support = np.append(selector_support, [False] * (n_features - len(selector_support)))

        valid_mask = (np.std(X, axis=0) > 1e-8) & (~np.isnan(X).all(axis=0))
        X_valid = X[:, valid_mask]
        feature_names_valid = [feature_names[i] for i, mask in enumerate(valid_mask) if mask]
        selector_support_valid = [selector_support[i] for i, mask in enumerate(valid_mask) if mask]

        if X_valid.shape[1] == 0:
            print("⚠️ No valid features for correlation")
            return

        corrs = np.corrcoef(np.vstack([X_valid.T, y]))[:X_valid.shape[1], -1]
        abs_corrs = np.abs(corrs)

        top_idx = np.argsort(abs_corrs)[-top_n:][::-1]
        top_features = [feature_names_valid[i] for i in top_idx]
        top_corrs = corrs[top_idx]
        is_selected = [selector_support_valid[i] for i in top_idx]

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['steelblue' if sel else 'coral' for sel in is_selected]
        bars = ax.barh(range(len(top_corrs)), top_corrs, color=colors, alpha=0.7)

        ax.set_yticks(range(len(top_corrs)))
        ax.set_yticklabels([f[:30] + '...' if len(f)>30 else f for f in top_features], fontsize=10)
        ax.set_xlabel('Correlation with Target')
        ax.set_title('Feature Selector Feature To Target Correlations')
        ax.grid(axis='x', alpha=0.7)
        ax.axvline(0, color='black', alpha=0.7)

        legend_elements = [
            Patch(facecolor='steelblue', alpha=0.7, label='Selected'),
            Patch(facecolor='coral', alpha=0.7, label='Not Selected'),
            Patch(facecolor='coral', alpha=0.7, label='If Not Selected = Keeps All Features')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig(out_dir/"feature_selector_correlations.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Selected: {sum(is_selected)}/{len(is_selected)}")
        print("✓ feature_selector_correlations.png")
    except Exception as e:
        print(f"⚠️ feature_target_correlations skipped: {e}")

def plot_dnn_predictions(y_true, y_pred, out_dir, subset_label, metrics):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, color='steelblue', alpha=0.7, s=35, label='Predicted vs True')
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Keras DNN {subset_label} Predictions')
    plt.legend()
    plt.grid(True, alpha=0.7)

    textstr = f'R²: {metrics["R2"]:.3f}\nAdj R²: {metrics["Adj_R2"]:.3f}\nRMSE: {metrics["RMSE"]:.3f}\nMAE: {metrics["MAE"]:.3f}'
    plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(Path(out_dir)/f'{subset_label.lower()}_dnn_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {subset_label.lower()}_dnn_predictions.png")

def print_outlier_analysis(y_true, y_pred, split_name, out_dir, df_deduped=None, orig_indices=None, state_col='State_Name', district_col='State_District_Name'):
    print("="*80)
    print(f"🔍 {split_name} Outlier Analysis")
    print("-"*80)

    residuals = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    abs_residuals = np.abs(residuals)

    Q1, Q3 = np.percentile(residuals, [25, 75])
    IQR = Q3 - Q1
    outliers = np.abs(residuals) > (1.5*IQR)
    print(f"📊 IQR Outliers:{outliers.sum():,}/{len(residuals):,} ({outliers.mean()*100:.1f}%)")

    worst_idx = np.argsort(abs_residuals)[-5:][::-1]
    print(f"\n🚨 Top 5 Worst Predictions:")
    print(f"{'#':>2} | {'True':>6} | {'Pred':>6} | {'Error':>6}")
    print("-"*30)

    for i, idx in enumerate(worst_idx):
        print(f"{i+1:2d} | {y_true.iloc[idx]:6.1f} | {y_pred[idx]:6.1f} | {abs_residuals[idx]:6.1f}")

    outliers_df = pd.DataFrame({
        'true': y_true, 'pred': y_pred, 'residual': residuals,
        'abs_error': abs_residuals, 'is_outlier': outliers
    })
    top_outliers = outliers_df.iloc[worst_idx].reset_index(drop=True)
    top_outliers.insert(0, 'row_index', worst_idx)
    top_outliers.to_csv(Path(out_dir)/f'{split_name.lower()}_top_outliers.csv', index=False)
    print(f"💾 {split_name.lower()}_top_outliers.csv")

def plot_statewise_histograms(df, valuecol, statecol, out_dir):
    try:
        plt.figure(figsize=(14, 8))
        state_means = df.groupby(statecol)[valuecol].mean().sort_values()
        states_order = state_means.index.tolist()[:10]
        plot_data = df[df[statecol].isin(states_order)]
        palette = sns.color_palette("husl", n_colors=len(states_order))
        sns.histplot(data=plot_data, x=valuecol, hue=statecol, hue_order=states_order,
                    bins=15, alpha=0.7, palette=palette, stat="density")
        plt.title(f"{valuecol} by {statecol}")
        plt.grid(axis='y', alpha=0.7)
        plt.tight_layout()
        plt.savefig(Path(out_dir)/"statewise_histogram.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ statewise_histogram.png")
    except Exception:
        print("⚠️ statewise_histogram skipped")

def plot_statewise_facets(df, value_col, state_col, out_dir):
    try:
        top_states = df[state_col].value_counts().head(6).index
        plot_data = df[df[state_col].isin(top_states)]
        g = sns.FacetGrid(plot_data, col=state_col, col_wrap=3, height=3, aspect=1.3)
        g.map(sns.histplot, value_col, bins=5, color="steelblue", alpha=0.7)
        g.set_titles("{col_name}")
        g.set_axis_labels(value_col, "Count")
        for ax in g.axes.flat:
            ax.set_axisbelow(True)
            ax.grid(axis='y', alpha=0.7, linestyle='-', color='gray')
        plt.savefig(Path(out_dir)/"statewise_facets.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ statewise_facets.png")
    except Exception:
        print("⚠️ statewise_facets skipped")

def plot_dnn_residuals(y_true, y_pred, out_dir, split_name):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_pred, residuals, alpha=0.7, s=40, color='steelblue')
    axes[0].axhline(0, color='coral', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'{split_name} Residuals')
    axes[0].grid(True, alpha=0.7)
    axes[1].hist(residuals, bins=10, color='steelblue', alpha=0.7, edgecolor='coral')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{split_name} Residuals Distribution')
    axes[1].grid(True, alpha=0.7)

    plt.tight_layout()
    plt.savefig(Path(out_dir)/f'{split_name.lower()}_dnn_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {split_name.lower()}_dnn_residuals.png")

def plot_residuals_scatter(y_true, y_pred, out_dir, split_name, jitter_level=1e-6):
    residuals = y_true - y_pred
    residuals = np.asarray(residuals).ravel()
    jitter = np.random.normal(0, jitter_level, size=residuals.shape)
    residuals_jitter = residuals + jitter

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals_jitter, alpha=0.7, s=40, color="steelblue")
    plt.axhline(0, color="coral", linestyle="--", linewidth=2)
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title(f"Keras DNN {split_name} Residuals")
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.savefig(Path(out_dir)/f"{split_name.lower()}_residuals_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ {split_name.lower()}_residuals_scatter.png")

def plot_prediction_distributions(y_true, y_pred, out_dir, split_name):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(y_true, bins=10, histtype="step", linewidth=3, color="steelblue", density=True, label="True", alpha=0.7)
    plt.hist(y_pred, bins=10, histtype="step", linewidth=3, color="coral", density=True, label="Predicted", alpha=0.7)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title(f"{split_name} Distribution")
    plt.legend()
    plt.grid(True, alpha=0.7)
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.7, s=40, color="steelblue")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs True")
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.savefig(Path(out_dir)/f"{split_name.lower()}_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {split_name.lower()}_distribution.png")

def plot_residual_distributions(y_true, y_pred, out_dir, split_name):
    residuals = y_true - y_pred
    mean_bias = np.mean(residuals)
    std_dev = np.std(residuals)

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=10, color="steelblue", alpha=0.7, density=True, label="Residuals")

    x = np.linspace(min(residuals), max(residuals), 100)
    plt.plot(x, norm.pdf(x, mean_bias, std_dev), color="orange", lw=2, label=f'Normal Fit (σ={std_dev:.2f})')
    plt.axvline(mean_bias, color='coral', linestyle='dashed', linewidth=2, label=f"Mean Bias: {mean_bias:.4f}")
    plt.title(f"{split_name} Residual Distribution", fontsize=14)
    plt.xlabel("Error (True - Predicted)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.7)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_name = f"{split_name.lower()}_residual_distribution.png"
    plt.savefig(out_path/file_name, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ {file_name}")

def plot_inference_comparisons(train_metrics, test_metrics, out_dir):
    metrics_df = pd.DataFrame([train_metrics, test_metrics], index=['Train', 'Test'])
    x = np.arange(len(metrics_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_keys = ['R2', 'Adj_R2']
    colors = ['steelblue', 'coral']

    for i, key in enumerate(metrics_keys):
        bars = ax.bar(x + i*width - width/2, metrics_df[key], width, label=key, alpha=0.8, color=colors[i], edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=12, weight='normal')

    ax.set_xlabel('Split')
    ax.set_ylabel('Score')
    ax.set_title('Keras DNN Inference Comparisons')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df.index)
    ax.legend()
    ax.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.savefig(Path(out_dir)/'dnn_inference_comparisons.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ dnn_inference_comparisons.png")

VIFResults = namedtuple('VIFResults', ['X_train', 'X_test', 'vif_df', 'dropped_features', 'feature_names'])
def prune_features(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    feature_names: Optional[list] = None,
    vif_threshold: float = 5.0,
    outdir: Optional[str] = None,
    logger=None,
    debug: bool = False,
    show_progress: bool = True
):
    if logger and debug:
        logger.info("🔧 Starting VIF feature pruning (threshold=%.1f)...", vif_threshold)

    if isinstance(X_train, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names must be provided when X_train is a NumPy array")
        if len(feature_names) != X_train.shape[1]:
            raise ValueError("len(feature_names) must match X_train.shape[1]")
        X_df = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_df = X_train.copy()
        if feature_names is not None:
            if len(feature_names) != X_df.shape[1]:
                raise ValueError("len(feature_names) must match DataFrame column count")
            X_df.columns = feature_names

    feature_names = list(X_df.columns)
    orig_col_indices = list(range(len(feature_names)))

    if X_test is not None:
        if isinstance(X_test, np.ndarray):
            assert X_train.shape[1] == X_test.shape[1], "X_train/test must have same # features"
            X_test_current = X_test.copy()
        else:
            assert len(X_test.columns) == len(X_df.columns), "X_test columns mismatch"
            X_test_current = X_test.copy()
    else:
        X_test_current = None

    dropped_features = []
    iteration = 0
    max_iterations = len(feature_names)

    vif_df = compute_vif(X_df)
    if outdir:
        path = Path(outdir)
        path.mkdir(parents=True, exist_ok=True)
        vif_df.to_csv(path/'vif_initial.csv', index=False)

    if logger:
        logger.info("Initial features: %s (max VIF: %s)", len(feature_names), vif_df['VIF'].max())

    if debug:
        print(f"🔧 VIF active: {len(feature_names)} → target VIF < {vif_threshold}")

    pbar = tqdm(
        total=max_iterations,
        disable=not show_progress,
        leave=True
    )

    try:
        while (
            len(feature_names) > 1 and
            iteration < max_iterations and
            vif_df['VIF'].max() > vif_threshold
        ):
            iteration += 1

            max_vif_idx = vif_df['VIF'].idxmax()
            max_vif_val = vif_df.loc[max_vif_idx, 'VIF']
            drop_feature = vif_df.loc[max_vif_idx, 'feature']
            col_idx = feature_names.index(drop_feature)

            if logger:
                logger.info(
                    "Iter %s/%s: Dropping '%s' (VIF=%s)",
                    iteration, max_iterations, drop_feature, max_vif_val
                )

            if debug:
                print(f"  → {drop_feature} (VIF={max_vif_val})")

            dropped_features.append(drop_feature)
            feature_names.pop(col_idx)
            orig_col_indices.pop(col_idx)

            if len(feature_names) > 0:
                X_df = X_df[feature_names]
                vif_df = compute_vif(X_df)
            else:
                break
            pbar.update(1)

        if show_progress:
            pbar.set_postfix(
                max_vif=f"{vif_df['VIF'].max():.2f}",
                kept=len(feature_names),
                dropped=len(dropped_features)
            )

    finally:
        pbar.close()

    X_train_final = X_df.values if len(feature_names) > 0 else np.empty((X_df.shape[0], 0))

    if X_test_current is not None:
        if isinstance(X_test, pd.DataFrame):
            X_test_final = (
                X_test[feature_names].values
                if len(feature_names) > 0
                else np.empty((X_test.shape[0], 0))
            )
        else:
            X_test_final = (
                X_test_current[:, orig_col_indices]
                if len(feature_names) > 0
                else np.empty((X_test.shape[0], 0))
            )
    else:
        X_test_final = None

    if outdir:
        vif_df.to_csv(Path(outdir)/'vif_final.csv', index=False)
        dropped_path = Path(outdir)/'vif_dropped_features.txt'
        with open(dropped_path, 'w') as f:
            f.write('\n'.join(dropped_features))

    if logger:
        orig_n = len(feature_names) + len(dropped_features)
        pct_kept = 100 * len(feature_names)/orig_n if orig_n > 0 else 0
        logger.info(
            "✅ VIF complete: %s → %s features (%.2f%% kept, max VIF=%s)",
            orig_n, len(feature_names), pct_kept, vif_df['VIF'].max()
        )

    if debug:
        orig_n = len(feature_names) + len(dropped_features)
        print(f"✅ VIF: {orig_n} → {len(feature_names)} (max VIF={vif_df['VIF'].max():.2f})")

    return VIFResults(
        X_train=X_train_final,
        X_test=X_test_final,
        vif_df=vif_df,
        dropped_features=dropped_features,
        feature_names=feature_names
    )

def compute_vif(X_df: pd.DataFrame) -> pd.DataFrame:
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant
        X_const = add_constant(X_df, has_constant='add')
        vif_data = [
            variance_inflation_factor(X_const.values, i)
            for i in range(X_const.shape[1])
        ]
        vif_df = pd.DataFrame({
            'feature': X_const.columns,
            'VIF': vif_data
        })
        return (
            vif_df[vif_df['feature'] != 'const']
            .sort_values('VIF', ascending=False, na_position='last')
            .reset_index(drop=True)
        )
    except (ImportError, np.linalg.LinAlgError, ValueError):
        return pd.DataFrame({
            'feature': X_df.columns,
            'VIF': [np.inf] * len(X_df.columns)
        }).sort_values('VIF', ascending=False).reset_index(drop=True)

def compute_adjusted_r2(r2_score, n_samples, n_features):
    if n_samples <= n_features + 1:
        return np.nan
    return 1 - (1 - r2_score) * (n_samples - 1)/(n_samples - n_features - 1)

# XAI
"""
This is a barebones proposal implementation for an explainability mechanism in artifical intelligence (AI). Inspired by the work of Bahdanau et. al, 2016, Luong et. al, 2015, Lundberg and Lee, 2017, Ribeiro et. al, 2017, and Vaswani et. al, 2017. This JAX/Keras DNN implementation had these considerations in mind during research and development. No formal research publication was published at the time of this project's development.

See references for more infomration:
- https://arxiv.org/abs/1409.0473
- https://arxiv.org/abs/1508.04025
- https://arxiv.org/abs/1705.07874
- https://arxiv.org/abs/1602.04938
- https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html
"""
def generate_explainability_plots(
    out_dir: str,
    keras_model,
    X_train_final: np.ndarray,
    X_test_selected: np.ndarray,
    feature_names: list,
    random_state: int = 42,
    max_background_samples: int = 500,
    max_display: int = 20,
    n_explain_instances: int = 3,
) -> None:
    print("="*80)
    print("SHAP + LIME Explainability Plots")
    print("="*80)

    out_dir = Path(out_dir)/"xai_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    steelblue_hex = "#4682b4"
    coral_hex = "#ff7f50"
    custom_cmap = LinearSegmentedColormap.from_list(
        "coral_steelblue", [coral_hex, steelblue_hex]
    )

    n_features = X_test_selected.shape[1]
    if len(feature_names) != n_features:
        feature_names = feature_names[:n_features]
        print(f"⚠️ Truncated feature_names to {n_features}")

    print(f"✅ SHAP validated: {len(feature_names)} features match X.shape[1]")

    rng = np.random.default_rng(seed=random_state)

    n_train = X_train_final.shape[0]
    if n_train <= 0:
        raise ValueError("X_train_final is empty; cannot sample background data.")

    n_bg = min(max_background_samples, n_train)
    n_bg = max(1, n_bg)
    bg_idx = rng.choice(n_train, size=n_bg, replace=False)
    background = X_train_final[bg_idx]

    print(f"✅ SHAP Background:{n_bg}/{n_train} training rows")

    def pred_fn(X: np.ndarray) -> np.ndarray:
        return keras_model.predict(X, verbose=0).flatten()

    def style_axis(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.set_facecolor("white")

    def recolor_shap_waterfall(fig=None, pos_color=steelblue_hex, neg_color=coral_hex):
        default_pos = "#ff0051"
        default_neg = "#008bfb"

        fig = fig or plt.gcf()

        for ax in fig.axes:
            for artist in ax.get_children():
                if isinstance(artist, matplotlib.patches.FancyArrow):
                    try:
                        face_hex = mcolors.to_hex(artist.get_facecolor(), keep_alpha=False).lower()
                        edge_hex = mcolors.to_hex(artist.get_edgecolor(), keep_alpha=False).lower()

                        if face_hex == default_pos:
                            artist.set_facecolor(pos_color)
                            artist.set_edgecolor(pos_color)
                        elif face_hex == default_neg:
                            artist.set_facecolor(neg_color)
                            artist.set_edgecolor(neg_color)
                        elif edge_hex == default_pos:
                            artist.set_edgecolor(pos_color)
                        elif edge_hex == default_neg:
                            artist.set_edgecolor(neg_color)
                    except Exception:
                        pass

                elif isinstance(artist, matplotlib.text.Text):
                    try:
                        txt_hex = mcolors.to_hex(artist.get_color(), keep_alpha=False).lower()
                        if txt_hex == default_pos:
                            artist.set_color(pos_color)
                        elif txt_hex == default_neg:
                            artist.set_color(neg_color)
                    except Exception:
                        pass

    def save_global_bar(shap_values_array, names, save_path, top_k=10):
        mean_abs = np.abs(shap_values_array).mean(axis=0)
        order = np.argsort(mean_abs)[-top_k:][::-1]
        vals = mean_abs[order]
        feats = [names[i] for i in order]
        short_names = [f"{name[:35]}..." if len(name) > 35 else name for name in feats]

        fig, ax = plt.subplots(figsize=(12, max(8, len(short_names)*0.45)))
        y = np.arange(len(short_names))
        bars = ax.barh(y, vals, color=steelblue_hex, edgecolor="black", linewidth=0.25, alpha=0.7)

        ax.set_yticks(y)
        ax.set_yticklabels(short_names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|", fontsize=12, weight='bold')
        ax.set_title("SHAP Global Explainability Plot", fontsize=14, weight='bold', pad=20)

        style_axis(ax)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
        plt.close()

    def save_force_plot(explanation, pred_value, save_path, top_k=None):
    	shap.plots.force(
        	explanation,
        	matplotlib=True,
        	show=False,
        	figsize=(14, 4),
        	plot_cmap=[coral_hex, steelblue_hex],
        	text_rotation=35
    	)
    	recolor_shap_waterfall(pos_color=steelblue_hex, neg_color=coral_hex)
    	plt.tight_layout()
    	plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    	plt.close()

    def save_lime_plot(lime_exp, pred_value, save_path, top_k=10):
        items = lime_exp.as_list()[:top_k]
        labels = [x[0] for x in items]
        weights = np.array([x[1] for x in items])

        order = np.argsort(np.abs(weights))[::-1]
        labels = [labels[i] for i in order]
        weights = weights[order]

        fig, ax = plt.subplots(figsize=(12, max(5, len(labels) * 0.55)))
        y = np.arange(len(labels))
        colors = [steelblue_hex if w >= 0 else coral_hex for w in weights]

        ax.barh(y, weights, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=1.0)
        ax.set_xlabel("LIME weight")
        ax.set_title(f"LIME Local Explanation (Prediction={pred_value:.3f})")
        style_axis(ax)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    kexplainer = shap.KernelExplainer(pred_fn, background)
    n_test_for_shap = min(100, X_test_selected.shape[0])
    X_shap = X_test_selected[:n_test_for_shap]

    print(f"🔧 Computing Kernel SHAP for {n_test_for_shap} test rows...")
    print("-" * 80)

    shap_values = kexplainer.shap_values(X_shap)

    shap_exp = shap.Explanation(
        values=shap_values,
        base_values=np.repeat(kexplainer.expected_value, shap_values.shape[0]),
        data=X_shap,
        feature_names=feature_names,
    )

    plt.figure(figsize=(12, 7))
    shap.plots.beeswarm(
        shap_exp[:, :max_display],
        max_display=max_display,
        color=custom_cmap,
        show=False
    )
    plt.title("SHAP Global Decision Plot")
    plt.subplots_adjust(left=0.25, right=0.95)
    plt.savefig(out_dir/"shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    save_global_bar(
        shap_values_array=shap_values,
        names=feature_names,
        save_path=out_dir/"shap_global_bar.png",
        top_k=max_display
    )

    shap_importance = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(shap_importance)[-10:][::-1]
    n_top = len(top_idx)

    print("-"*80)
    print("📊 SHAP Top Global Features:")
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. {feature_names[idx]}: {shap_importance[idx]:.4f}")

    top_df = pd.DataFrame({
        "rank": range(1, n_top + 1),
        "feature": [feature_names[i] for i in top_idx],
        "shap_importance": shap_importance[top_idx]
    })
    top_df.to_csv(out_dir/"shap_top_global.csv", index=False)
    print("💾 Saved: shap_top_global.csv")

    n_local = min(n_explain_instances, n_test_for_shap)
    print("📋 SHAP Predictions Top Local Features:")

    for i in range(n_local):
        pred_val = pred_fn(X_shap[i:i+1])[0]

        plt.figure(figsize=(12, 7))
        shap.plots.waterfall(shap_exp[i], max_display=max_display, show=False)
        recolor_shap_waterfall()
        ax = plt.gca()
        style_axis(ax)
        plt.subplots_adjust(left=0.40, right=0.95)
        plt.savefig(out_dir/f"shap_waterfall_{i+1}.png", dpi=300, bbox_inches="tight")
        plt.close()

        save_force_plot(
            explanation=shap_exp[i],
            pred_value=pred_val,
            save_path=out_dir/f"shap_force_{i+1}.png",
            top_k=min(10, len(feature_names))
        )

        pred_top3_idx = np.argsort(np.abs(shap_values[i]))[-3:][::-1]
        print(f"SHAP Prediction {i+1} (IMR={pred_val:.2f}) - Top Local Features:")
        for rank, idx in enumerate(pred_top3_idx, 1):
            feat = feature_names[idx]
            shap_val = shap_values[i, idx]
            direction = "increase" if shap_val >= 0 else "decrease"
            print(f"  {rank}. {feat}: {shap_val:+.4f} ({direction})")

    print("-"*80)
    print("📊 Generating LIME plots...")

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_final,
        mode="regression",
        feature_names=feature_names,
        categorical_features=[],
        kernel_width=0.75 * np.sqrt(X_train_final.shape[1]),
        random_state=random_state,
        verbose=False,
    )

    n_lime = min(n_explain_instances, X_test_selected.shape[0])
    for i in range(n_lime):
        lime_exp = lime_explainer.explain_instance(
            X_test_selected[i],
            pred_fn,
            num_features=min(10, len(feature_names)),
        )

        pred_val = pred_fn(X_test_selected[i:i+1])[0]

        print(f"LIME Prediction {i+1} Top Local Features:")
        for feat, weight in lime_exp.as_list():
            direction = "increase" if weight >= 0 else "decrease"
            print(f"  {feat}: {weight:+.4f} ({direction})")

        save_lime_plot(
            lime_exp=lime_exp,
            pred_value=pred_val,
            save_path=out_dir/f"lime_instance_{i+1}.png",
            top_k=10
        )

    print(f"✅ Explainability plots saved to: {out_dir}")

def save_model_metrics(train_metrics, test_metrics, out_dir):
    results = pd.DataFrame([train_metrics, test_metrics], index=['Train', 'Test'])
    results.index.name = 'Split'
    results.to_csv(Path(out_dir)/'metrics.csv', float_format='%.4f')
    print("✓ metrics.csv")

def main(args):
    print("="*80)
    print("Keras Deep Neural Network")
    print("="*80)
    logging.captureWarnings(True)
    logger = setup_logging(args.outdir, args.debug)

    if not args.debug:
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
        warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
    else:
        warnings.filterwarnings("always")

    df = load_data(args.data)
    original_target = args.target
    df.columns = [re.sub(r'^([A-Z]{2})_', '', col) for col in df.columns]
    args.target = find_target_columns(df, original_target)
    print_dataset_stats(df, args.target)

    mask = ~df.duplicated().values
    df_deduped = df[mask].copy()
    if df_deduped[args.target].isnull().any():
        df_deduped = df_deduped.dropna(subset=[args.target])

    prefix_id_cols = [re.sub(r'^([A-Z]{2})_', '', col) for col in args.id_cols]
    X = df_deduped.drop(columns=[args.target] + [col for col in prefix_id_cols if col in df_deduped.columns])
    y = df_deduped[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, shuffle=True)

    train_indices = X_train.index.values
    test_indices = X_test.index.values

    preprocessor = build_preprocessor(X_train)
    check_data_imputations(X_train, X_test, preprocessor)
    X_train_processed = preprocessor_checkpoint(X_train, preprocessor)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)
    feature_redundancy_checks(X_train_processed, feature_names)

    X_train_processed, X_test_processed, feature_names = drop_feature_correlations(
        X_train_processed, X_test_processed, y_train, feature_names, args.correlation)

    num_features = X_train_processed.shape[1]
    target_min, target_max = y.min(), y.max()
    print(f"✅ Features:{num_features} | Target Range:{target_min:.1f}-{target_max:.1f}")
    print_preprocessing_stats(X_train_processed, y_train, feature_names, X_train_processed.shape[1])

    logger.info("VIF: Multicollinearity Check + Feature Pruning")

    pre_vif_names = get_feature_names(preprocessor)
    logger.info(f"Pre-VIF names: {pre_vif_names[:5]}...")

    if args.vif_threshold > 0:
        vif_results = prune_features(
            X_train_processed,
            X_test_processed,
            feature_names=feature_names,
            vif_threshold=args.vif_threshold,
            debug=args.debug,
            outdir=args.outdir,
            logger=logger
        )

        X_train_vif_pruned = vif_results.X_train
        X_test_vif_pruned = vif_results.X_test
        vif_final = vif_results.vif_df
        dropped_vif = vif_results.dropped_features
        vif_feature_names = pre_vif_names[:vif_results.X_train.shape[1]]
        logger.info(f"VIF clean names preview: {vif_feature_names[:5]}...")
    else:
        vif_feature_names = pre_vif_names
        X_train_vif_pruned = X_train_processed
        X_test_vif_pruned = X_test_processed

    cv = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
    rf = RandomForestRegressor(random_state=args.random_state, n_jobs=1)

    with feature_selection(min(64, X_train_vif_pruned.shape[1]//10)):
        selector = RFECV(rf, step=0.01, min_features_to_select=10, cv=cv, scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1)
        selector.fit(X_train_vif_pruned, y_train)

    X_train_selected = selector.transform(X_train_vif_pruned)
    X_test_selected = selector.transform(X_test_vif_pruned)

    vif_rfecv_final = compute_vif(pd.DataFrame(X_train_selected, columns=vif_feature_names[:X_train_selected.shape[1]]))
    vif_rfecv_final.to_csv(Path(args.outdir)/'vif_after_rfecv.csv', index=False)

    logger.info("Final VIF Check After Feature Selection:")
    logger.info(f"Test samples: {len(y_test)}, Features: {X_test_selected.shape[1]}")
    logger.info("Adj R² NaN = normal when n_test < n_features + 1")
    logger.info(f"Features Dropped: {[pre_vif_names[i] for i in range(len(pre_vif_names)) if i not in range(vif_results.X_train.shape[1])]}")
    logger.info(f"RFECV final max VIF: {vif_rfecv_final['VIF'].max():.2f}")

    selected_feature_names = [fname for fname, selected in zip(feature_names, selector.support_) if selected]
    print(f"✅ Number of Features Selected:{X_train_selected.shape[1]}")
    print(f"✅ Selected features:{selected_feature_names}")

    state_col = find_state_columns(df_deduped, prefix_id_cols)
    if state_col and state_col in df_deduped.columns:
        plots_dir = ensure_subdir(Path(args.outdir), 'dnn_plots')
        print(f"="*80)
        print(f"✅ Statewise plots")
        print("="*80)
        plot_statewise_histograms(df_deduped, args.target, state_col, plots_dir)
        plot_statewise_facets(df_deduped, args.target, state_col, plots_dir)

    print("="*80)
    print("🔧 Building & Training Keras Deep Neural Network...")
    print("="*80)
    dnn = build_keras_dnn(X_train_selected.shape[1], learning_rate=args.lr, dropout_rate=args.dropout, l2_reg=args.l2_reg)

    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_selected, y_train, test_size=args.val_size, random_state=args.random_state)

    keras_dnn, history = train_keras_dnn(
        dnn, X_train_final, y_train_final, X_val, y_val, Path(args.outdir))

    selected_feature_names = [fname for fname, selected in zip(feature_names, selector.support_) if selected]
    generate_explainability_plots(args.outdir, keras_dnn, X_train_final, X_test_selected, selected_feature_names, args.random_state)

    y_train_pred = keras_dnn.predict(X_train_selected, verbose=0).ravel()
    y_test_pred = keras_dnn.predict(X_test_selected, verbose=0).ravel()

    n_train, p = len(y_train), X_train_selected.shape[1]
    n_test = len(y_test)

    train_r2 = r2_score(y_train, y_train_pred)
    train_adj_r2 = compute_adjusted_r2(train_r2, n_train, p)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)

    test_r2 = r2_score(y_test, y_test_pred)
    test_adj_r2 = compute_adjusted_r2(test_r2, n_test, p)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    global_dir = ensure_subdir(Path(args.outdir))
    print_outlier_analysis(y_train, y_train_pred, "Train", global_dir, df_deduped=df_deduped,
                          orig_indices=train_indices, state_col='State_Name', district_col='State_District_Name')
    print_outlier_analysis(y_test, y_test_pred, "Test", global_dir, df_deduped=df_deduped,
                          orig_indices=test_indices, state_col='State_Name', district_col='State_District_Name')
    train_metrics = {'R2': train_r2, 'Adj_R2': train_adj_r2, 'RMSE': train_rmse, 'MAE': train_mae}
    test_metrics = {'R2': test_r2, 'Adj_R2': test_adj_r2, 'RMSE': test_rmse, 'MAE': test_mae}
    save_model_metrics(train_metrics, test_metrics, global_dir)

    dnn_dir = ensure_subdir(Path(args.outdir), 'dnn_artifacts')
    keras_dnn.save(dnn_dir/'keras_full_dnn.keras')
    joblib.dump(preprocessor, dnn_dir/'data_preprocessor.joblib')
    joblib.dump(selector, dnn_dir/'feature_selector.joblib')

    print("="*80)
    print("📊 Keras DNN Plots")
    print("="*80)
    print("📊 Generating DNN inference plots...")
    plots_dir = ensure_subdir(Path(args.outdir), 'dnn_plots')
    selected_idx = np.where(selector.support_)[0]
    selected_names = [feature_names[i] for i in selected_idx]
    plot_training_summary(history, plots_dir)
    plot_dnn_residuals(y_train, y_train_pred, plots_dir, "Train")
    plot_dnn_residuals(y_test, y_test_pred, plots_dir, "Test")
    plot_residuals_scatter(y_train, y_train_pred, plots_dir, "Train")
    plot_residuals_scatter(y_test, y_test_pred, plots_dir, "Test")
    plot_feature_importances(selector.estimator_.feature_importances_, feature_names, selector.support_, plots_dir)
    plot_feature_target_correlations(X_train_vif_pruned, y_train, [f'vif_{i}' for i in range(X_train_vif_pruned.shape[1])],
                                np.ones(X_train_vif_pruned.shape[1], dtype=bool), plots_dir)
    plot_feature_target_correlations(X_train_selected, y_train, selected_feature_names, selector.support_, plots_dir)
    plot_dnn_predictions(y_train, y_train_pred, plots_dir, 'Train', train_metrics)
    plot_dnn_predictions(y_test, y_test_pred, plots_dir, 'Test', test_metrics)
    plot_prediction_distributions(y_train, y_train_pred, plots_dir, 'Train')
    plot_prediction_distributions(y_test, y_test_pred, plots_dir, 'Test')
    plot_residual_distributions(y_train, y_train_pred, plots_dir, 'Train')
    plot_residual_distributions(y_test, y_test_pred, plots_dir, 'Test')
    plot_inference_comparisons(train_metrics, test_metrics, plots_dir)

    print("="*80)
    print("✅ Keras Deep Neural Network Results:")
    print("="*80)
    print(f"🎯 Test: R²={test_r2:.4f} | Adj R²={test_adj_r2:.4f} | RMSE={test_rmse:.4f} | MAE={test_mae:.4f}")
    print(f"📊 Features Selected:{selector.n_features_}/{len(feature_names)}")
    print(f"💾 Architecture:512-256-128-64-1 Deep Network")
    print(f"💾 Models:keras_full_dnn.keras | keras_best_dnn.keras")
    print(f"📁 Outputs:{Path(args.outdir)}")
    print("="*80)

    logger.info("Pipeline complete. Test R²=%.4f", test_r2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Keras nonlinear regression deep learning neural network. Machine learning for health analytics. The University of Texas at Arlington.')
    parser.add_argument('--data', required=True, help="Selected dataset")
    parser.add_argument('--target', required=True, help="Predicted target variable")
    parser.add_argument('--id-cols', nargs='+', default=[], help="Feature columns to exclude from training")
    parser.add_argument("--correlation", type=float, default=0.0, help="Drop features by pct correlation")
    parser.add_argument('--vif-threshold', type=float, default=5.0, help='VIF threshold for multicollinearity and feature pruning')
    parser.add_argument('--test-size', type=float, default=0.25, help="Test set size")
    parser.add_argument('--val-size', type=float, default=0.15, help='Validation set size')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--l2-reg', type=float, default=0.01, help='L2 Ridge regularization')
    parser.add_argument('--epochs', type=int, default=500, help='Max epochs')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--outdir', default='artifacts')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)
