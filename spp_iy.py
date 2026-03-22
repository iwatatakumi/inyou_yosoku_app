# -*- coding: utf-8 -*-
"""
Created on 2025/10/20
前日の米国、中国株式、為替情報より豊田通商（8015）の当日の陰陽予測
機械学習ベース
@author: RyotoTanaka
"""
import os
import sys

import japanize_matplotlib

# from sklearn.metrics import mean_absolute_error
import joblib
import lightgbm as lgbm
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from spp_data_iy import spp_data_create  # データ前処理

sys.path.append(".")  # 自作モジュールへのPathを追加する（colabで必要）
VERSION = "v002"
# モデル名
ARG = "LightGBM"
ARG2 = "XGBoost"
ARG3 = "RForest"

# train/testのディレクトリ
INPUT_DIR = "./csv"
# model/submissionの出力先
OUTPUT_DIR = "./models/spp_model_" + VERSION

# ファイル名
MODEL_NAME_LGBM = "spp_" + ARG + "_" + VERSION
MODEL_NAME_XGB = "spp_" + ARG2 + "_" + VERSION
MODEL_NAME_RFT = "spp_" + ARG3 + "_" + VERSION

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # カレントディレクトリをプログラムのあるディレクトリに移動する

# main
train_X, train_y, test_X = spp_data_create(use_yfinance=True, yf_start="2020-01-01")  # データ加工（特徴量抽出）

# LightGBM パラメータ 回帰
lgbm_params = {
    # 'objective': 'regression',  # 回帰
    # 'objective': 'regression_l1',  # l1回帰
    "objective": "regression_l2",  # l2回帰
    "boosting": "gbdt",  # 勾配ブースティング
    # 'boosting': 'dart',
    "max_depth": -1,  # ツリーの最深度 -1
    "num_leaves": 31,  # 葉の数 31
    "min_data_in_leaf": 20,  # 葉に含まれる最小データ数 20
    "learning_rate": 0.001,  # 0.1
    "feature_fraction": 0.4,  # 特徴量の使用率 1.0
    "bagging_fraction": 0.6,  # Baggingの使用率 1.0
    "bagging_freq": 2,  # Baggingを何回ごとに行うか 0
    "lambda_l1": 0.6,  # L1正則化の重み 0
    "lambda_l2": 0.6,  # L2正則化の重み 0
    # 'metrics': 'auc',
    "metrics": "rmse",  # root_mean_squared_error 平均二乗誤差の平方根
    # 'metrics': 'mae',  # 平均絶対誤差
    "force_col_wise": True,  # overheadの警告が表示されたので指定した
    "random_seed": 42,
    "n_jobs": -1,
}

# LightGBM モデル
model_lgb = lgbm.LGBMRegressor(**lgbm_params)

# XGBoostハイパーパラメータ
xgb_params = {
    "objective": "reg:squarederror",  # 目的関数
    "eval_metric": "rmse",  # 学習に用いる評価指標
    "booster": "gbtree",  # boosterに何を用いるか
    "eta": 0.001,  # learning_rateと同義
    "max_depth": 6,  # 木の最大深さ
    "seed": 2525,  # random_stateと同義
}

# RandomForestハイパーパラメータ
model_rft = RandomForestRegressor(
    n_estimators=100,
    criterion="squared_error",  # tensorflow 2.12.0
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    # max_features='auto',
    max_features=1.0,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=42,
    verbose=1,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None,
)


BATCH_SIZE = 10  # バッチサイズ
EPOCHS = 1000  # エポック数

# KFoldで予測
kf = KFold(n_splits=5, shuffle=True, random_state=42)
models_lgbm = []
models_xgb = []
models_rft = []
metrics_avg_lgbm = []
metrics_avg_xgb = []
metrics_avg_rft = []
# metrics_avg_dnn = []
i_cnt = 0

for train_index, val_index in kf.split(train_X, train_y):
    X_train = train_X.iloc[train_index]
    X_valid = train_X.iloc[val_index]
    y_train = train_y.iloc[train_index]
    y_valid = train_y.iloc[val_index]

    lgb_train = lgbm.Dataset(X_train, y_train)
    lgb_vali = lgbm.Dataset(X_valid, y_valid, reference=lgb_train)

    model_lgb = lgbm.train(
        lgbm_params,
        lgb_train,
        valid_sets=lgb_vali,
        num_boost_round=10000,  # 最大学習サイクル数。early_stopping使用時は大きな値を入力
        # early_stopping_rounds= 500,  # このパラメーターは廃止予定
        callbacks=[
            lgbm.early_stopping(stopping_rounds=2000, verbose=True),  # early_stopping用コールバック関数
            lgbm.log_evaluation(100),  # コマンドライン出力用コールバック関数 結果を100行ごとに表示
        ],
    )
    y_pred_lgb = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)  # 予測結果
    print(pd.DataFrame(y_pred_lgb).describe())
    tmp_metrics = np.sqrt(mean_squared_error(y_valid, y_pred_lgb))  # rmse
    # tmp_metrics = mean_absolute_error(y_valid, y_pred_lgb)  # MAE
    models_lgbm.append(model_lgb)  # モデルの保存
    metrics_avg_lgbm.append(tmp_metrics)

    # ランダムフォレスト・モデル学習
    model_rft.fit(X_train, y_train.values.ravel())
    y_pred_rft = model_rft.predict(X_valid)  # 予測結果
    print(pd.DataFrame(y_pred_rft).describe())
    tmp_metrics = np.sqrt(mean_squared_error(y_valid, y_pred_rft))  # rmse
    models_rft.append(model_rft)  # モデルの保存
    metrics_avg_rft.append(tmp_metrics)
    # 評価
    print(model_rft.score(X_train, y_train))

    # XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)  # feature_namesを指定しない 2024/02/22
    dtest = xgb.DMatrix(X_valid, label=y_valid)
    evals_result = {}
    model_xgb = xgb.train(
        params=xgb_params,
        dtrain=dtrain,  # 学習用データ
        num_boost_round=50000,  # 学習のラウンド数
        early_stopping_rounds=100,  # early stoppinguのラウンド数
        evals=[(dtrain, "train"), (dtest, "eval")],  # 検証用データ
        evals_result=evals_result,  # 学習途中結果
    )
    # y_pred_xgb = model_xgb.predict(dtest, num_iteration=model_xgb.best_iteration)
    y_pred_xgb = model_xgb.predict(dtest)
    print(pd.DataFrame(y_pred_xgb).describe())
    tmp_metrics = np.sqrt(mean_squared_error(y_valid, y_pred_xgb))  # rmse
    models_xgb.append(model_xgb)  # モデルの保存
    metrics_avg_xgb.append(tmp_metrics)

# 平均metric
print("平均rmse(lgbm)= ", sum(metrics_avg_lgbm) / len(metrics_avg_lgbm))
print("平均rmse_rft = ", sum(metrics_avg_rft) / len(metrics_avg_rft))
print("平均rmse_xgb = ", sum(metrics_avg_xgb) / len(metrics_avg_xgb))

# 特徴量の影響度を表示
for model in models_lgbm:
    lgbm.plot_importance(model, importance_type="gain", max_num_features=20)
# lgbm それぞれのモデルで予測して結果を配列に格納する
preds_lgbm = []
for i, model in enumerate(models_lgbm):
    print("best_iteration: ", model.best_iteration)
    if model.best_iteration > 5:  # イテレーションが10以上（ある程度回った）だけを対象とする
        joblib.dump(model, f"{OUTPUT_DIR}/{MODEL_NAME_LGBM}_{i:02d}.pkl")  # modelの保存
        pred = model.predict(test_X, num_iteration=model.best_iteration)  # Predict
        print("予測: ", pred)
        preds_lgbm.append(pred)
# 予測結果の平均をとる
preds_array = np.array(preds_lgbm)
preds_mean = np.mean(preds_array, axis=0)
print("有効モデル数（lgbm）= ", len(preds_lgbm), " 本日の陰陽予測= ", preds_mean)

# ランダムフォレストそれぞれのモデルで予測して結果を配列に格納する
preds_rft = []
for i, model in enumerate(models_rft):
    # modelの保存
    joblib.dump(model, f"{OUTPUT_DIR}/{MODEL_NAME_RFT}_{i:02d}.pkl")
    # Predict
    pred = model.predict(test_X)
    preds_rft.append(pred)
# 予測結果の平均をとる
preds_array = np.array(preds_rft)
preds_mean = np.mean(preds_array, axis=0)
print("有効モデル数rft = ", len(preds_rft), " 翌日の陰陽予測 = ", preds_mean)

# xgb それぞれのモデルで予測して結果を配列に格納する
preds_xgb = []
dtest_x = xgb.DMatrix(test_X)
for i, model in enumerate(models_xgb):
    if model.best_iteration > 20:  # イテレーションが99以上（ある程度回った）だけを対象とする
        # modelの保存
        joblib.dump(model, f"{OUTPUT_DIR}/{MODEL_NAME_XGB}_{i:02d}.pkl")
        # Predict
        pred = model.predict(dtest_x)
        preds_xgb.append(pred)
# 予測結果の平均をとる
preds_array = np.array(preds_xgb)
preds_mean = np.mean(preds_array, axis=0)
print("有効モデル数xgb = ", len(preds_xgb), " 翌日の陰陽予測 = ", preds_mean)
