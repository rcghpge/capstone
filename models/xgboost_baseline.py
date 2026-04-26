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
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import json
"""
Example Usage:
python xgboost_baseline.py --data ../data/Key_indicator_districtwise.csv \
--target YY_Infant_Mortality_Rate_Imr_Total_Person --id-cols State_Name State_District_Name \
--outdir xgboost_baseline
"""

def calculate_adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1)/(n - p - 1)

def save_model_metrics(train_metrics, test_metrics, outdir):
    results = pd.DataFrame([train_metrics, test_metrics], index=['Train', 'Test'])
    results.index.name = 'Split'
    results.to_csv(Path(args.outdir)/'metrics.csv', float_format='%.4f')
    print("✓ metrics.csv")
    
def build_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), 
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

def parse_args():
    p = argparse.ArgumentParser(description="XGBoost Baseline Regression Model")
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--id-cols", nargs="*", default=[])
    p.add_argument("--outdir", required=True)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()

def main(args):
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    print("XGBoost Baseline Regression Model")
    print("=" * 40)

    df = pd.read_csv(args.data)
    X = df.drop(columns=[args.target] + args.id_cols)
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = XGBRegressor(random_state=args.random_state, verbosity=0, n_jobs=-1)
    model.fit(X_train_processed, y_train)
    
    y_train_pred = model.predict(X_train_processed).ravel()
    y_test_pred = model.predict(X_test_processed).ravel()

    n_train, p = len(y_train), X_train_processed.shape[1]
    n_test = len(y_test)

    train_r2 = r2_score(y_train, y_train_pred)
    train_adj_r2 = calculate_adjusted_r2(train_r2, n_train, p)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)

    test_r2 = r2_score(y_test, y_test_pred)
    test_adj_r2 = calculate_adjusted_r2(test_r2, n_test, p)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_metrics = {'R2': train_r2, 'Adj_R2': train_adj_r2, 'RMSE': train_rmse, 'MAE': train_mae}
    test_metrics = {'R2': test_r2, 'Adj_R2': test_adj_r2, 'RMSE': test_rmse, 'MAE': test_mae}
    save_model_metrics(train_metrics, test_metrics, (Path(args.outdir)))

    print(f"✅ XGBoost Baseline Model Results")
    print(f"🎯 Test: R²={test_r2:.4f} | Adj R²={test_adj_r2:.4f} | RMSE={test_rmse:.4f} | MAE={test_mae:.4f}")
    print(f"📁 Outputs: {Path(args.outdir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XGBoost nonlinear regression baseline model. Machine learning for health analytics. The University of Texas at Arlington.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--id-cols', nargs='+', default=[])
    parser.add_argument('--outdir', default='artifacts')
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    main(args)
