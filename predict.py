# -*- coding: utf-8 -*-
"""
陰陽予測コアロジック
spp_iy.py の学習・予測処理を run_prediction() として関数化したモジュール
"""
import os
import sys
import time
import numpy as np
import joblib
import lightgbm as lgbm
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

VERSION = "v002"
OUTPUT_DIR = "./models/spp_model_" + VERSION
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./csv", exist_ok=True)


def run_prediction(yf_start: str = "2020-01-01", progress_callback=None) -> dict:
    """
    陰陽予測を実行して結果を返す。

    Parameters
    ----------
    yf_start : str
        yfinanceのデータ取得開始日 (例: "2020-01-01")
    progress_callback : callable, optional
        進捗通知コールバック。signature: callback(step: int, total: int, message: str)

    Returns
    -------
    dict
        prediction / metrics / feature_importance / meta のキーを持つ結果辞書
    """
    start_time = time.time()

    def _notify(step, total, message):
        if progress_callback:
            progress_callback(step, total, message)

    try:
        # -----------------------------------------------------------------
        # 1. データ取得・特徴量エンジニアリング
        # -----------------------------------------------------------------
        _notify(0, 16, "データを取得中…（yfinance）")
        from spp_data_iy import spp_data_create
        train_X, train_y, test_X = spp_data_create(
            use_yfinance=True, yf_start=yf_start
        )

        # -----------------------------------------------------------------
        # 2. モデル定義
        # -----------------------------------------------------------------
        lgbm_params = {
            "objective": "regression_l2",
            "boosting": "gbdt",
            "max_depth": -1,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "learning_rate": 0.001,
            "feature_fraction": 0.4,
            "bagging_fraction": 0.6,
            "bagging_freq": 2,
            "lambda_l1": 0.6,
            "lambda_l2": 0.6,
            "metrics": "rmse",
            "force_col_wise": True,
            "random_seed": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        xgb_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "booster": "gbtree",
            "eta": 0.001,
            "max_depth": 6,
            "seed": 2525,
            "verbosity": 0,
        }

        # -----------------------------------------------------------------
        # 3. KFold 学習・予測（5 fold × 3 model = 15 ステップ）
        # -----------------------------------------------------------------
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        models_lgbm, models_xgb, models_rft = [], [], []
        metrics_lgbm, metrics_xgb, metrics_rft = [], [], []
        step = 0

        for fold_i, (train_idx, val_idx) in enumerate(kf.split(train_X, train_y), start=1):
            X_train = train_X.iloc[train_idx]
            X_valid = train_X.iloc[val_idx]
            y_train = train_y.iloc[train_idx]
            y_valid = train_y.iloc[val_idx]

            # --- LightGBM ---
            step += 1
            _notify(step, 16, f"Fold {fold_i}/5 : LightGBM 学習中…")
            lgb_train = lgbm.Dataset(X_train, y_train)
            lgb_vali = lgbm.Dataset(X_valid, y_valid, reference=lgb_train)
            model_lgb = lgbm.train(
                lgbm_params,
                lgb_train,
                valid_sets=lgb_vali,
                num_boost_round=10000,
                callbacks=[
                    lgbm.early_stopping(stopping_rounds=2000, verbose=False),
                    lgbm.log_evaluation(period=-1),
                ],
            )
            y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
            metrics_lgbm.append(np.sqrt(mean_squared_error(y_valid, y_pred)))
            models_lgbm.append(model_lgb)

            # --- RandomForest ---
            step += 1
            _notify(step, 16, f"Fold {fold_i}/5 : RandomForest 学習中…")
            model_rft = RandomForestRegressor(
                n_estimators=100,
                criterion="squared_error",
                max_depth=None,
                max_features=1.0,
                random_state=42,
                verbose=0,
                n_jobs=-1,
            )
            model_rft.fit(X_train, y_train.values.ravel())
            y_pred = model_rft.predict(X_valid)
            metrics_rft.append(np.sqrt(mean_squared_error(y_valid, y_pred)))
            models_rft.append(model_rft)

            # --- XGBoost ---
            step += 1
            _notify(step, 16, f"Fold {fold_i}/5 : XGBoost 学習中…")
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_valid, label=y_valid)
            model_xgb = xgb.train(
                params=xgb_params,
                dtrain=dtrain,
                num_boost_round=50000,
                early_stopping_rounds=100,
                evals=[(dtrain, "train"), (dtest, "eval")],
                verbose_eval=False,
            )
            y_pred = model_xgb.predict(dtest)
            metrics_xgb.append(np.sqrt(mean_squared_error(y_valid, y_pred)))
            models_xgb.append(model_xgb)

        # -----------------------------------------------------------------
        # 4. テストデータへの予測
        # -----------------------------------------------------------------
        _notify(16, 16, "予測結果を集計中…")

        # LightGBM
        preds_lgbm = []
        for i, model in enumerate(models_lgbm):
            if model.best_iteration > 5:
                joblib.dump(model, f"{OUTPUT_DIR}/spp_LightGBM_{VERSION}_{i:02d}.pkl")
                preds_lgbm.append(
                    model.predict(test_X, num_iteration=model.best_iteration)
                )
        lgbm_mean = float(np.mean(preds_lgbm)) if preds_lgbm else 0.0

        # RandomForest
        preds_rft = []
        for i, model in enumerate(models_rft):
            joblib.dump(model, f"{OUTPUT_DIR}/spp_RForest_{VERSION}_{i:02d}.pkl")
            preds_rft.append(model.predict(test_X))
        rft_mean = float(np.mean(preds_rft)) if preds_rft else 0.0

        # XGBoost
        preds_xgb = []
        dtest_x = xgb.DMatrix(test_X)
        for i, model in enumerate(models_xgb):
            if model.best_iteration > 20:
                joblib.dump(model, f"{OUTPUT_DIR}/spp_XGBoost_{VERSION}_{i:02d}.pkl")
                preds_xgb.append(model.predict(dtest_x))
        xgb_mean = float(np.mean(preds_xgb)) if preds_xgb else 0.0

        # アンサンブル（3モデルの平均）
        all_preds = [v for v in [lgbm_mean, rft_mean, xgb_mean]]
        ensemble_mean = float(np.mean(all_preds))
        vote_陽 = sum(1 for v in all_preds if v > 0)

        # -----------------------------------------------------------------
        # 5. 特徴量重要度（LightGBM の gain 平均）
        # -----------------------------------------------------------------
        feat_imp_map: dict = {}
        for model in models_lgbm:
            imp = model.feature_importance(importance_type="gain")
            names = model.feature_name()
            for n, v in zip(names, imp):
                feat_imp_map[n] = feat_imp_map.get(n, 0.0) + v
        # 平均化して降順ソート
        n_models = len(models_lgbm) or 1
        sorted_feats = sorted(feat_imp_map.items(), key=lambda x: x[1], reverse=True)
        feature_names = [f for f, _ in sorted_feats]
        feature_gains = [v / n_models for _, v in sorted_feats]

        # -----------------------------------------------------------------
        # 6. 予測対象日（test_X の最後の日付 or 今日）
        # -----------------------------------------------------------------
        import datetime
        pred_date = datetime.date.today().strftime("%Y/%m/%d")

        elapsed = time.time() - start_time

        return {
            "prediction": {
                "lgbm": {
                    "value": lgbm_mean,
                    "label": "陽線" if lgbm_mean > 0 else "陰線",
                    "confidence": abs(lgbm_mean),
                    "valid_model_count": len(preds_lgbm),
                },
                "xgb": {
                    "value": xgb_mean,
                    "label": "陽線" if xgb_mean > 0 else "陰線",
                    "confidence": abs(xgb_mean),
                    "valid_model_count": len(preds_xgb),
                },
                "rft": {
                    "value": rft_mean,
                    "label": "陽線" if rft_mean > 0 else "陰線",
                    "confidence": abs(rft_mean),
                    "valid_model_count": len(preds_rft),
                },
                "ensemble": {
                    "value": ensemble_mean,
                    "label": "陽線" if ensemble_mean > 0 else "陰線",
                    "vote_count": vote_陽,
                },
            },
            "metrics": {
                "lgbm_rmse": float(np.mean(metrics_lgbm)),
                "xgb_rmse": float(np.mean(metrics_xgb)),
                "rft_rmse": float(np.mean(metrics_rft)),
            },
            "feature_importance": {
                "features": feature_names[:20],
                "gains": feature_gains[:20],
            },
            "meta": {
                "prediction_date": pred_date,
                "train_data_count": len(train_X),
                "yf_start": yf_start,
                "elapsed_seconds": elapsed,
                "error": None,
            },
        }

    except Exception as e:
        import traceback
        return {
            "prediction": None,
            "metrics": None,
            "feature_importance": None,
            "meta": {
                "prediction_date": None,
                "train_data_count": 0,
                "yf_start": yf_start,
                "elapsed_seconds": time.time() - start_time,
                "error": traceback.format_exc(),
            },
        }
