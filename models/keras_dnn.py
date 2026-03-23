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
import joblib
import logging
import warnings
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from pathlib import Path
from sklearn.base import clone
import matplotlib.pyplot as plt
from contextlib import contextmanager
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.parallel import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, learning_curve, validation_curve

# Keras Build
import shap
import lime
import keras
from keras import layers, callbacks
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
"""
Example Usage:
python keras_dnn.py --data ../data/Key_indicator_districtwise.csv \
--id-cols State_Name State_District_Name --target Infant_Mortality_Rate_Imr_Total_Person \
--correlation 72 --test-size 0.24 --outdir artifacts/keras_dnn
"""
sns.set_palette("husl")
plt.style.use('default')

def setup_logging(out_dir: str, debug: bool):
    log_path = Path(out_dir)/"logs"
    log_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path/"global_debug.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    logging.captureWarnings(True)

    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.setLevel(logging.WARNING)
    warn_handler = logging.FileHandler(log_path/"warnings.log")
    warn_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    warnings_logger.addHandler(warn_handler)
    warnings_logger.addHandler(logging.StreamHandler())

    for logger_name in ["jax", "keras"]:
        lib_logger = logging.getLogger(logger_name)
        lib_logger.setLevel(logging.DEBUG if debug else logging.WARNING)
        debug_handler = logging.FileHandler(log_path/f"{logger_name}_debug.log")
        debug_handler.setLevel(logging.DEBUG if debug else logging.WARNING)
        debug_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        lib_logger.addHandler(debug_handler)

    if debug:
        os.environ["JAX_LOGGING_LEVEL"] = "DEBUG"

    logger = logging.getLogger(__name__)
    logger.info("Logging setup (debug=%s): Warnings → warnings.log", debug)
    return logger

def ensure_subdir(base_dir: Path, *subpaths: str) -> Path:
    full_path = base_dir.joinpath(*subpaths)
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path

@contextmanager
def spinner_progress(total_steps=64):
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
    print(f'\r✅ Feature Selection Complete! {elapsed/60:.1f}m total')

def load_data(data_path):
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        return pd.read_parquet(data_path)
    raise ValueError('Unsupported file type. Use .csv or .parquet')

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)), ('scaler', RobustScaler())])
    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True)), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
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

def build_keras_model(input_dim, learning_rate=0.001, dropout_rate=0.2, l2_reg=0.2):
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate/2),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='huber', metrics=['mae'])
    return model

def train_keras_model(model, X_train, y_train, X_val, y_val, out_dir, epochs=500, batch_size=32, patience=50):
    callbacks_list = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, min_lr=1e-7, verbose=0, mode='min'),
        keras.callbacks.ModelCheckpoint(str(out_dir/'best_keras_model.keras'), monitor='val_loss', save_best_only=True, verbose=0)
    ]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks_list, verbose=0)
    return model, history

def plot_training_history(history, out_dir):
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(Path(args.outdir)/'training_history.csv', index=False)
    print(f"📊 Saved {len(history_df)} epochs")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0,0].plot(history.history['loss'], label='Train MSE', color='steelblue', lw=2)
    axes[0,0].plot(history.history['val_loss'], label='Val MSE', color='coral', lw=2)
    axes[0,0].set_title('Model Loss\n(MSE=2.21 → RMSE=√MSE=1.49)')
    axes[0,0].set_xlabel('Epoch'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
    axes[0,1].plot(history.history['mae'], label='Train MAE', color='steelblue', lw=2)
    axes[0,1].plot(history.history['val_mae'], label='Val MAE', color='coral', lw=2)
    axes[0,1].set_title('Model MAE')
    axes[0,1].set_xlabel('Epoch'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.5)

    if 'lr' in history.history:
        axes[1,0].semilogy(history.history['lr'], color='darkgreen', lw=2)
        axes[1,0].set_title('Learning Rate'); axes[1,0].grid(True, alpha=0.5)
    else:
        best_epoch = np.argmin(history.history['val_loss'])
        axes[1,0].text(0.1, 0.6, f'EarlyStopping Active\nBest Epoch: {best_epoch}\nVal Loss: {history.history["val_loss"][best_epoch]:.3f}',
                       transform=axes[1,0].transAxes, fontsize=12, bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))
        axes[1,0].set_title('Training Dynamics')
        axes[1,0].axis('off')

    axes[1,1].axis('off')
    col_labels = ['Metric', 'Train', 'Val']
    row_labels = ['Final MSE', 'Final MAE']
    cell_text = [[f"{history.history['loss'][-1]:.3f}", f"{history.history['val_loss'][-1]:.3f}"],
                 [f"{history.history['mae'][-1]:.3f}", f"{history.history['val_mae'][-1]:.3f}"]]

    table = axes[1,1].table(cellText=cell_text, colLabels=col_labels, rowLabels=row_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.3, 1.8)
    axes[1,1].set_title('Final Metrics', weight='bold')

    plt.suptitle('Keras Deep Neural Network Training Summary', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(out_dir/'keras_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ keras_training_history.png")

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
    print(f"📊 Dataset Shape: ({total_samples}, {total_features})")
    print(f"🎯 Target Column: '{target_col}'")
    total_missing = df.isnull().sum().sum()
    missing_pct = (total_missing/(total_samples * total_features)) * 100
    print(f"🔍 Total Missing Null/NaN Values: {total_missing:,} ({missing_pct:.2f}%)")
    print("="*80)

def print_preprocessing_stats(X_processed, y_train, feature_names, num_features):
    print("="*80)
    print("📊 Feature Selection Summary")
    print("="*80)

    n_samples, n_features = X_processed.shape
    print(f"📊 Processed Dataset: ({n_samples}, {n_features})")
    print(f"🎯 Target Samples: {len(y_train)}")
    print(f"📋 Feature Names Available: {len(feature_names)}")

    total_missing = np.isnan(X_processed).sum()
    print(f"🔍 Post-Preprocessing Missing: {total_missing:,}")
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
    print(f"📊 Continuous Numeric: {high_card_num}")
    print(f"🔢 Discrete Numeric: {low_card_num}")
    print(f"🔢 Binary Flags/OHE: {binary_flags} (incl {missing_indicators} missing indicators)")
    print(f"📈 Constant (drop): {constant}")
    print(f"🔄 OHE Generated: {ohe_generated}")
    print(f"📊 Total Active: {high_card_num + low_card_num + binary_flags - constant}")
    print("-"*80)
    print(f"📋 10 Lowest Cardinality Features:")
    print(unique_counts.sort_values().head(10))
    print("-"*80)
    print(f"📋 10 Highest Cardinality Features:")
    print(unique_counts.sort_values(ascending=False).head(10))
    
    sparsity = (X_processed == 0).sum() / X_processed.size * 100
    print("-"*80)
    print(f"📊 Data Quality:")
    print(f"   Sparsity (% zeros): {sparsity:.1f}%")
    print(f"   Feature/Target ratio: {n_features/len(y_train):.2f}")
    print("="*80)
    print("📊 Feature Selection")
    print("="*80)
    print(f"✅ Feature Selection: {n_features} → Will look to select the best features")
    
def plot_feature_importances(importances, feature_names, selector_support, out_dir, top_n=25):
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
    plt.tight_layout()
    plt.savefig(Path(out_dir)/'feature_estimator_importances.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ feature_estimator_importances.png")

def plot_feature_target_correlations(X_processed, y_train, feature_names, selector_support, out_dir, top_n=20):
    try:
        X = np.asarray(X_processed)
        y = np.asarray(y_train).ravel()

        valid_mask = (np.std(X, axis=0) > 1e-8) & (~np.isnan(X).all(axis=0))
        X_valid = X[:, valid_mask]
        feature_names_valid = [feature_names[i] for i, mask in enumerate(valid_mask) if mask]
        selector_support_valid = [selector_support[i] for i, mask in enumerate(valid_mask) if mask]

        if X_valid.shape[1] == 0:
            print("⚠️ No valid features for correlation")
            return

        corrs = np.corrcoef(X_valid.T, y)[:X_valid.shape[1], -1]
        abs_corrs = np.abs(corrs)

        top_idx = np.argsort(abs_corrs)[-top_n:][::-1]

        top_features = [feature_names[i] for i in top_idx]
        top_corrs = corrs[top_idx]
        is_selected = [bool(selector_support[i]) for i in top_idx]

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['steelblue' if sel else 'coral' for sel in is_selected]
        ax.barh(range(len(top_corrs)), top_corrs, color=colors, alpha=0.7)

        ax.set_yticks(range(len(top_corrs)))
        ax.set_yticklabels([f[:30] for f in top_features], fontsize=10)
        ax.set_xlabel('Correlation with Target')
        ax.set_title('Feature Selector Feature-Target Correlations')
        ax.grid(axis='x', alpha=0.7)
        ax.axvline(0, color='black', alpha=0.7)

        plt.tight_layout()
        plt.savefig(out_dir/"feature_selector_correlations.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("✓ feature_selector_correlations.png")
    except Exception as e:
        print(f"⚠️ feature_target_correlations skipped: {e}")

def plot_model_predictions(y_true, y_pred, out_dir, subset_label, metrics):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, color='steelblue', alpha=0.7, s=35, label='Pred vs True')
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
    plt.savefig(Path(out_dir)/f'{subset_label.lower()}_model_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {subset_label.lower()}_model_predictions.png")

def print_outlier_analysis(y_true, y_pred, split_name, out_dir, df_deduped=None, orig_indices=None, state_col='State_Name', district_col='State_District_Name'):
    print("="*80)
    print(f"🔍 {split_name} Outlier Analysis")
    print("-"*80)

    residuals = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    abs_residuals = np.abs(residuals)

    Q1, Q3 = np.percentile(residuals, [25, 75])
    IQR = Q3 - Q1
    outliers = np.abs(residuals) > (1.5*IQR)
    print(f"📊 IQR Outliers: {outliers.sum():,}/{len(residuals):,} ({outliers.mean()*100:.1f}%)")

    worst_idx = np.argsort(abs_residuals)[-5:][::-1]
    print(f"\n🚨 Top 5 Worst Predictions:")
    print(f"{'#':2s} | True | Pred | Error")
    print("-"*30)
    for i, idx in enumerate(worst_idx):
        print(f"{i+1:2d} | {y_true.iloc[idx]:6.1f} | {y_pred[idx]:6.1f} | {abs_residuals[idx]:6.1f}")

    outliers_df = pd.DataFrame({
        'true': y_true, 'pred': y_pred, 'residual': residuals,
        'abs_error': abs_residuals, 'is_outlier': outliers
    })
    outliers_df.iloc[worst_idx].to_csv(Path(out_dir)/f'{split_name.lower()}_top_outliers.csv')
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
        g.map(sns.histplot, value_col, bins=7, color="steelblue", alpha=0.7)
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

def plot_model_residuals(y_true, y_pred, out_dir, split_name):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_pred, residuals, alpha=0.7, s=40, color='steelblue')
    axes[0].axhline(0, color='coral', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'{split_name} Residuals vs Predicted')
    axes[0].grid(True, alpha=0.7)
    axes[1].hist(residuals, bins=15, color='steelblue', alpha=0.7, edgecolor='coral')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{split_name} Residuals Distribution')
    axes[1].grid(True, alpha=0.7)

    plt.tight_layout()
    plt.savefig(Path(out_dir)/f'{split_name.lower()}_model_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {split_name.lower()}_model_residuals.png")

def plot_granular_residuals(y_true, y_pred, out_dir, split_name, jitter_level=1e-6):
    residuals = y_true - y_pred
    residuals = np.asarray(residuals).ravel()
    jitter = np.random.normal(0, jitter_level, size=residuals.shape)
    residuals_jitter = residuals + jitter

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals_jitter, alpha=0.7, s=40, color="steelblue")
    plt.axhline(0, color="coral", linestyle="--", linewidth=2)
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title(f"Model {split_name} Granular Residuals")
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.savefig(Path(out_dir)/f"{split_name.lower()}_granular_residuals.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ {split_name.lower()}_granular_residuals.png")

def plot_prediction_distributions(y_true, y_pred, out_dir, split_name):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(y_true, bins=15, histtype="step", linewidth=3, color="steelblue", density=True, label="True", alpha=0.9)
    plt.hist(y_pred, bins=15, histtype="step", linewidth=3, color="coral", density=True, label="Predicted", alpha=0.9)
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
    plt.title("Pred vs True")
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.savefig(Path(out_dir)/f"{split_name.lower()}_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {split_name.lower()}_distribution.png")

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
    ax.set_title('Model Inference Comparisons')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df.index)
    ax.legend()
    ax.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.savefig(Path(out_dir)/'model_inference_comparisons.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ model_inference_comparisons.png")

def plot_learning_curves(history, out_dir):
    history_df = pd.DataFrame(history.history)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history_df['mae'], label='Train MAE', color='steelblue', lw=2)
    ax1.plot(history_df['val_mae'], label='Val MAE', color='coral', lw=2)
    ax1.set_title('MAE Learning Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MAE')
    ax1.legend()
    ax1.grid(True, alpha=0.7)

    ax2.plot(np.sqrt(history_df['loss']), label='Train RMSE', color='steelblue', lw=2)
    ax2.plot(np.sqrt(history_df['val_loss']), label='Val RMSE', color='coral', lw=2)
    ax2.set_title('RMSE Learning Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE')
    ax2.legend()
    ax2.grid(True, alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_dir/"learning_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ learning_curves.png")

def plot_validation_curves(history, out_dir, param_name="Learning Rate"):
    history_df = pd.DataFrame(history.history)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(history_df['val_mae'], label='Val MAE', color='coral', lw=2)
    ax2 = ax.twinx()
    ax2.plot(np.sqrt(history_df['val_loss']), label='Val RMSE', color='orange', lw=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val MAE', color='coral')
    ax2.set_ylabel('Val RMSE', color='orange')
    ax.set_title(f'Validation Curves ({param_name})')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_dir/"validation_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ validation_curves.png")

"""
def generate_explainability_plots(out_dir: str,
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
    print("Generating SHAP + LIME Explainability Plots")
    print("="*80)

    out_dir = Path(out_dir)/"xai_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    assert X_train_final.shape[1] == len(feature_names), f"MISMATCH: X has {X_train_final.shape[1]} cols, {len(feature_names)} names"
    assert X_test_selected.shape[1] == len(feature_names), f"MISMATCH: X_test has {X_test_selected.shape[1]} cols, {len(feature_names)} names"
    
    print(f"✅ SHAP validated: {len(feature_names)} features match X.shape[1]")
    warnings.filterwarnings("ignore", category=UserWarning, module="shap")
    rng = np.random.default_rng(seed=random_state)

    n_train = X_train_final.shape[0]
    if n_train <= 0:
        raise ValueError("X_train_final is empty; Cannot sample background data.")

    n_bg = min(max_background_samples, max(50, n_train))
    bg_idx = rng.choice(n_train, size=n_bg, replace=False)
    background = X_train_final[bg_idx]
    
    print(f"✅ SHAP background: {n_bg}/{n_train} training rows")

    def pred_fn(X: np.ndarray) -> np.ndarray:
        return keras_model.predict(X, verbose=0).flatten()

    kexplainer = shap.KernelExplainer(pred_fn, background)
    n_test_for_shap = min(100, X_test_selected.shape[0])
    X_shap = X_test_selected[:n_test_for_shap]

    print(f"📊 Computing Kernel SHAP for {n_test_for_shap} test rows (this can take a bit)...")
    print("-"*80)
    shap_array = kexplainer.shap_values(X_shap)  
    
    shap_exp = shap.Explanation(
        values=shap_array,
        base_values=np.repeat(kexplainer.expected_value, shap_array.shape[0]),
        data=X_shap,
        feature_names=feature_names,
    )

    plt.figure(figsize=(12, 7))
    shap.plots.beeswarm(shap_exp[:, :max_display], max_display=max_display, show=False)
    plt.subplots_adjust(left=0.2, right=0.95)
    plt.savefig(out_dir/"shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_exp[:, :max_display].abs.mean(0), show=False)
    plt.subplots_adjust(left=0.25, right=0.95)
    plt.savefig(out_dir/"shap_global_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    n_local = min(n_explain_instances, n_test_for_shap)
    print("-"*80)
    print(f"📊 Generating SHAP explanations for {n_local} test instances...")
    for i in range(n_local):
        plt.figure(figsize=(10, 7))
        shap.plots.waterfall(shap_exp[i], max_display=max_display, show=False)
        plt.subplots_adjust(left=0.35, right=0.95)
        plt.savefig(out_dir/f"shap_waterfall_{i+1}.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("📊 Generating LIME explanations...")
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
            num_features=min(3, len(feature_names)),
        )
    
    print(f"✅ LIME Instance {i+1} Top Features:")
    for feat, weight in lime_exp.as_list(): 
        print(f"  {feat}: {weight:.4f}")

    fig = lime_exp.as_pyplot_figure()  
    plt.tight_layout()
    plt.savefig(out_dir/f"lime_instance_{i+1}.png", dpi=300, bbox_inches="tight")
    plt.close()
        
    print(f"✅ Explainability plots saved to: {out_dir}")
"""
def generate_explainability_plots(out_dir: str,
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
    print("Generating SHAP + LIME Explainability Plots")
    print("="*80)

    out_dir = Path(out_dir)/"xai_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    assert X_train_final.shape[1] == len(feature_names), f"MISMATCH: X has {X_train_final.shape[1]} cols, {len(feature_names)} names"
    assert X_test_selected.shape[1] == len(feature_names), f"MISMATCH: X_test has {X_test_selected.shape[1]} cols, {len(feature_names)} names"
    
    print(f"✅ SHAP validated: {len(feature_names)} features match X.shape[1]")
    warnings.filterwarnings("ignore", category=UserWarning, module="shap")
    rng = np.random.default_rng(seed=random_state)

    n_train = X_train_final.shape[0]
    if n_train <= 0:
        raise ValueError("X_train_final is empty; Cannot sample background data.")

    n_bg = min(max_background_samples, max(50, n_train))
    bg_idx = rng.choice(n_train, size=n_bg, replace=False)
    background = X_train_final[bg_idx]
    
    print(f"✅ SHAP background: {n_bg}/{n_train} training rows")

    def pred_fn(X: np.ndarray) -> np.ndarray:
        return keras_model.predict(X, verbose=0).flatten()

    kexplainer = shap.KernelExplainer(pred_fn, background)
    n_test_for_shap = min(100, X_test_selected.shape[0])
    X_shap = X_test_selected[:n_test_for_shap]

    print(f"📊 Computing Kernel SHAP for {n_test_for_shap} test rows (this can take a bit)...")
    print("-"*80)
    shap_array = kexplainer.shap_values(X_shap)  
    
    shap_exp = shap.Explanation(
        values=shap_array,
        base_values=np.repeat(kexplainer.expected_value, shap_array.shape[0]),
        data=X_shap,
        feature_names=feature_names,
    )

    plt.figure(figsize=(12, 7))
    shap.plots.beeswarm(shap_exp[:, :max_display], max_display=max_display, show=False)
    plt.subplots_adjust(left=0.2, right=0.95)
    plt.savefig(out_dir/"shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_exp[:, :max_display].abs.mean(0), show=False)
    plt.subplots_adjust(left=0.25, right=0.95)
    plt.savefig(out_dir/"shap_global_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    n_local = min(n_explain_instances, n_test_for_shap)
    print("-"*80)
    print(f"📊 Generating SHAP explanations for {n_local} test instances...")
    for i in range(n_local):
        plt.figure(figsize=(10, 7))
        shap.plots.waterfall(shap_exp[i], max_display=max_display, show=False)
        plt.subplots_adjust(left=0.35, right=0.95)
        plt.savefig(out_dir/f"shap_waterfall_{i+1}.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # New: SHAP Force Plot for instance i
        plt.figure(figsize=(14, 4))
        shap.plots.force(shap_exp[i], matplotlib=True, show=False, figsize=(14, 4))
        plt.tight_layout()
        plt.savefig(out_dir/f"shap_force_{i+1}.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved SHAP force plot for instance {i+1}")

    print("📊 Generating LIME explanations...")
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
            num_features=min(3, len(feature_names)),
        )
    
    print(f"✅ LIME Instance {i+1} Top Features:")
    for feat, weight in lime_exp.as_list(): 
        print(f"  {feat}: {weight:.4f}")

    fig = lime_exp.as_pyplot_figure()  
    plt.tight_layout()
    plt.savefig(out_dir/f"lime_instance_{i+1}.png", dpi=300, bbox_inches="tight")
    plt.close()
        
    print(f"✅ Explainability plots saved to: {out_dir}")

def calculate_adjusted_r2(r2_score, n_samples, n_features):
    if n_samples <= n_features + 1:
        return np.nan
    return 1 - (1 - r2_score) * (n_samples - 1)/(n_samples - n_features - 1)

def save_model_metrics(train_metrics, test_metrics, out_dir):
    results = pd.DataFrame([train_metrics, test_metrics], index=['Train', 'Test'])
    results.index.name = 'Split'
    results.to_csv(Path(out_dir)/'keras_metrics.csv', float_format='%.4f')
    print("✓ keras_metrics.csv")

def main(args):
    print("="*80)
    print("Keras Deep Neural Network")
    print("="*80)
    logging.captureWarnings(True)
    logger = setup_logging(args.outdir, args.debug)
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
    X_train_processed = preprocessor_checkpoint(X_train, preprocessor)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)
    X_train_processed, X_test_processed, feature_names = drop_feature_correlations(
        X_train_processed, X_test_processed, y_train, feature_names, args.correlation)

    num_features = X_train_processed.shape[1]
    target_min, target_max = y.min(), y.max()
    print(f"✅ Features: {num_features} | Target Range: {target_min:.1f}-{target_max:.1f}")
    print_preprocessing_stats(X_train_processed, y_train, feature_names, X_train_processed.shape[1])
    cv = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
    rf = RandomForestRegressor(random_state=42, n_jobs=1)

    with spinner_progress(min(64, X_train_processed.shape[1]//10)):
        selector = RFECV(rf, step=10, cv=cv, scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1)
        selector.fit(X_train_processed, y_train)

    X_train_selected = selector.transform(X_train_processed)
    X_test_selected = selector.transform(X_test_processed)

    print(f"✅ Number of Features Selected: {X_train_selected.shape[1]}")
    selected_feature_names = [fname for fname, selected in zip(feature_names, selector.support_) if selected]
    print(f"✅ Selected features: {selected_feature_names}")

    state_col = find_state_columns(df_deduped, prefix_id_cols)
    if state_col and state_col in df_deduped.columns:
        plots_dir = ensure_subdir(Path(args.outdir), 'model_plots')
        plot_statewise_histograms(df_deduped, args.target, state_col, plots_dir)
        plot_statewise_facets(df_deduped, args.target, state_col, plots_dir)

    print("-"*80)
    print("🔧 Building & Training Keras Deep Neural Network...")
    model = build_keras_model(X_train_selected.shape[1], learning_rate=args.lr, dropout_rate=args.dropout, l2_reg=args.l2_reg)

    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_selected, y_train, test_size=args.val_size, random_state=args.random_state)

    keras_model, history = train_keras_model(
        model, X_train_final, y_train_final, X_val, y_val, Path(args.outdir))

    selected_feature_names = [fname for fname, selected in zip(feature_names, selector.support_) if selected]
    generate_explainability_plots(args.outdir, keras_model, X_train_final, X_test_selected, selected_feature_names, args.random_state)
    
    y_train_pred = keras_model.predict(X_train_selected, verbose=0).ravel()
    y_test_pred = keras_model.predict(X_test_selected, verbose=0).ravel()

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

    global_dir = ensure_subdir(Path(args.outdir))
    print_outlier_analysis(y_train, y_train_pred, "Train", global_dir, df_deduped=df_deduped, 
                          orig_indices=train_indices, state_col='State_Name', district_col='State_District_Name')
    print_outlier_analysis(y_test, y_test_pred, "Test", global_dir, df_deduped=df_deduped, 
                          orig_indices=test_indices, state_col='State_Name', district_col='State_District_Name')
    train_metrics = {'R2': train_r2, 'Adj_R2': train_adj_r2, 'RMSE': train_rmse, 'MAE': train_mae}
    test_metrics = {'R2': test_r2, 'Adj_R2': test_adj_r2, 'RMSE': test_rmse, 'MAE': test_mae}
    save_model_metrics(train_metrics, test_metrics, global_dir)

    model_dir = ensure_subdir(Path(args.outdir), 'model_artifacts')
    keras_model.save(model_dir/'keras_model_full.keras')
    joblib.dump(preprocessor, model_dir/'data_preprocessor.joblib')
    joblib.dump(selector, model_dir/'feature_selector.joblib')

    print("="*80)
    print("📊 Model Plots")
    print("="*80)
    print("📊 Generating model inference plots...")
    plots_dir = ensure_subdir(Path(args.outdir), 'model_plots')
    plot_training_history(history, plots_dir)
    plot_learning_curves(history, plots_dir)
    plot_validation_curves(history, plots_dir)
    plot_model_residuals(y_train, y_train_pred, plots_dir, "Train")
    plot_model_residuals(y_test, y_test_pred, plots_dir, "Test")
    plot_granular_residuals(y_train, y_train_pred, plots_dir, "Training")
    plot_granular_residuals(y_test, y_test_pred, plots_dir, "Testing")
    plot_feature_target_correlations(X_train_processed, y_train, feature_names, selector.support_, plots_dir)
    plot_model_predictions(y_train, y_train_pred, plots_dir, 'Train', train_metrics)
    plot_model_predictions(y_test, y_test_pred, plots_dir, 'Test', test_metrics)
    plot_feature_importances(selector.estimator_.feature_importances_, feature_names, selector.support_, plots_dir)
    plot_prediction_distributions(y_train, y_train_pred, plots_dir, 'Train')
    plot_prediction_distributions(y_test, y_test_pred, plots_dir, 'Test')
    plot_inference_comparisons(train_metrics, test_metrics, plots_dir)

    print("="*80)
    print("✅ Keras Deep Neural Network Results:")
    print("="*80)
    print(f"🎯 Test: R²={test_r2:.4f} | Adj R²={test_adj_r2:.4f} | RMSE={test_rmse:.4f} | MAE={test_mae:.4f}")
    print(f"📊 Features Selected: {selector.n_features_}/{len(feature_names)}")
    print(f"💾 Architecture: 512-256-128-64-1 Deep Network")
    print(f"💾 Models: keras_model_full.keras | best_keras_model.keras")
    print(f"📁 Outputs: {Path(args.outdir)}")
    print("="*80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Keras nonlinear regression deep learning neural network. Health Analytics: Machine learning for infant mortality rate prediction. The University of Texas at Arlington.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--id-cols', nargs='+', default=[])
    parser.add_argument("--correlation", type=float, default=0.0, help="Drop features by pct correlation")
    parser.add_argument('--test-size', type=float, default=0.25)
    parser.add_argument('--val-size', type=float, default=0.15, help='Train/val split size')
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
