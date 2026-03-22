# -*- coding: utf-8 -*-
"""
Created on 2025/10/20
陰陽予測のデータ作成処理
@author: RyotoTanaka
"""


def spp_data_create(use_yfinance=False, yf_start="2020-01-01", yf_end=None):
    import os
    import datetime

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler  # 正規化
    from sklearn.preprocessing import StandardScaler  # 標準化

    # カレントディレクトリをプログラムのあるディレクトリに移動する
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def up_hige_calc(df, p_start, p_end, p_high):  # 上ヒゲを計算
        start_p = df[p_start]
        end_p = df[p_end]
        taka_p = df[p_high]
        # 始値と終値で高い方を判定
        if start_p > end_p:
            high_p = start_p
        else:
            high_p = end_p

        # 高い方と高値との比較
        result = (taka_p - high_p) / end_p * 100
        return result

    def dw_hige_calc(df, p_start, p_end, p_low):  # 下ヒゲを計算
        start_p = df[p_start]
        end_p = df[p_end]
        yasu_p = df[p_low]
        # 始値と終値で低い方を判定
        if start_p < end_p:
            low_p = start_p
        else:
            low_p = end_p

        # 低い方と安値との比較
        result = (low_p - yasu_p) / end_p * 100
        return result

    pd.set_option("future.no_silent_downcasting", True)  # Pandas将来の警告に対応

    if use_yfinance:
        # ----------------------------------------------------------------
        # yfinanceからデータを取得する
        # ----------------------------------------------------------------
        import yfinance as yf

        if yf_end is None:
            yf_end = datetime.date.today().strftime("%Y-%m-%d")

        def _download_ohlcv(ticker):
            """yfinanceからOHLCVデータを取得し、昇順DataFrameで返す"""
            df = yf.download(ticker, start=yf_start, end=yf_end,
                             auto_adjust=True, progress=False)
            # multi-level columnsへの対応
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            # 列名を日本語に変換
            df = df.rename(columns={
                "Open": "始値", "High": "高値", "Low": "安値",
                "Close": "終値", "Volume": "出来高",
            })
            # インデックス（日付）を文字列に変換
            df = df.reset_index()
            date_col = "Date" if "Date" in df.columns else "Datetime"
            df["日付"] = pd.to_datetime(df[date_col]).dt.strftime("%Y/%m/%d")
            df = df.drop(columns=[date_col])
            # 昇順ソート（移動平均ローリング計算のため）
            df = df.sort_values("日付", ascending=True).reset_index(drop=True)
            return df

        # ---- 豊田通商（8015） ----
        _df = _download_ohlcv("8015.T")
        train_target = _df[["日付", "始値", "高値", "安値", "終値", "出来高"]].copy()
        train_target["前日比"]      = train_target["終値"].diff(1)
        train_target["売買代金"]    = (train_target["終値"] * train_target["出来高"]).astype("int64")
        # 信用取引データはyfinanceで取得不可のため0をセット
        # 精度向上にはJ-Quants APIの利用を推奨（https://jpx-jquants.com）
        train_target["貸株残高"]    = 0.0
        train_target["融資残高"]    = 0.0
        train_target["貸借倍率"]    = 0.0
        train_target["逆日歩"]      = 0.0
        train_target["特別空売り料"] = 0.0
        # 移動平均（昇順のまま計算）
        train_target["5DMA"]   = train_target["終値"].rolling(5).mean()
        train_target["25DMA"]  = train_target["終値"].rolling(25).mean()
        train_target["75DMA"]  = train_target["終値"].rolling(75).mean()
        train_target["100DMA"] = train_target["終値"].rolling(100).mean()
        train_target["200DMA"] = train_target["終値"].rolling(200).mean()
        train_target["5DVMA"]  = train_target["出来高"].rolling(5).mean()
        train_target["25DVMA"] = train_target["出来高"].rolling(25).mean()
        train_target["75DVMA"] = train_target["出来高"].rolling(75).mean()
        # 降順ソート（元のCSV形式に合わせる）
        train_target = train_target.sort_values("日付", ascending=False).reset_index(drop=True)

        # ---- 各市場指数（始値・高値・安値・終値のみ） ----
        def _get_index(ticker):
            df = _download_ohlcv(ticker)[["日付", "始値", "高値", "安値", "終値"]]
            return df.sort_values("日付", ascending=False).reset_index(drop=True)

        train_sp500    = _get_index("^GSPC")
        train_nyd      = _get_index("^DJI")
        train_nasdaq   = _get_index("^IXIC")
        train_vix      = _get_index("^VIX")
        train_hongkong = _get_index("^HSI")
        train_shanhi   = _get_index("000001.SS")

        # ---- ドル/円 ----
        _df = _download_ohlcv("USDJPY=X")
        train_usdyen = _df[["日付", "始値", "高値", "安値", "終値"]].copy()
        train_usdyen["前日比"] = train_usdyen["終値"].diff(1)
        train_usdyen["5DMA"]  = train_usdyen["終値"].rolling(5).mean()
        train_usdyen["25DMA"] = train_usdyen["終値"].rolling(25).mean()
        train_usdyen = train_usdyen.sort_values("日付", ascending=False).reset_index(drop=True)

    else:
        # ----------------------------------------------------------------
        # CSVファイルからデータを取得する（既存処理）
        # ----------------------------------------------------------------
        IN_DIR = "./in_data/"
        IN_TARGET   = IN_DIR + "8015.csv"
        IN_SP500    = IN_DIR + "SP500.csv"
        IN_NYD      = IN_DIR + "NYD.csv"
        IN_NASDAQ   = IN_DIR + "NASDAQ.csv"
        IN_VIX      = IN_DIR + "VIX.csv"
        IN_USDYEN   = IN_DIR + "USDYEN_bit.csv"
        IN_HONGKONG = IN_DIR + "HongKong.csv"
        IN_SHANHAI  = IN_DIR + "ShanHai.csv"

        train_target   = pd.read_csv(IN_TARGET,    index_col=False)  # 予測したい銘柄
        train_sp500    = pd.read_csv(IN_SP500,     index_col=False)  # S&P500指数
        train_nyd      = pd.read_csv(IN_NYD,       index_col=False)  # NYダウ
        train_nasdaq   = pd.read_csv(IN_NASDAQ,    index_col=False)  # NASDAQ総合指数
        train_vix      = pd.read_csv(IN_VIX,       index_col=False)  # VIX
        train_usdyen   = pd.read_csv(IN_USDYEN,    index_col=False)  # ドル/円（Bid）
        train_hongkong = pd.read_csv(IN_HONGKONG,  index_col=False)  # 香港ハンセン指数
        train_shanhi   = pd.read_csv(IN_SHANHAI,   index_col=False)  # 上海総合指数

    # ----------------------------------------------------------------
    # 以下は共通処理
    # ----------------------------------------------------------------

    # 左マージして項目にサフィックスを付加する
    train_target = pd.merge(train_target, train_sp500,    on=["日付"], how="left", suffixes=["", "_sp500"])
    train_target = pd.merge(train_target, train_nyd,      on=["日付"], how="left", suffixes=["", "_nyd"])
    train_target = pd.merge(train_target, train_nasdaq,   on=["日付"], how="left", suffixes=["", "_nasdaq"])
    train_target = pd.merge(train_target, train_vix,      on=["日付"], how="left", suffixes=["", "_vix"])
    train_target = pd.merge(train_target, train_usdyen,   on=["日付"], how="left", suffixes=["", "_usdyen"])
    train_target = pd.merge(train_target, train_hongkong, on=["日付"], how="left", suffixes=["", "_hongkong"])
    train_target = pd.merge(train_target, train_shanhi,   on=["日付"], how="left", suffixes=["", "_shanhi"])

    train_target = train_target[~(train_target["始値"] == "-")]  # 始値が-は削除 2020/10/01のデータ
    train_target.replace("-", 0, inplace=True, regex=True)  # -を0に変換 逆日歩、特別空売り料など

    rp_list = [  # カンマ付き数字を不動小数点に変換
        "始値",
        "始値_sp500",
        "始値_nyd",
        "始値_nasdaq",
        "始値_hongkong",
        "始値_shanhi",
        "高値",
        "高値_sp500",
        "高値_nyd",
        "高値_nasdaq",
        "高値_hongkong",
        "高値_shanhi",
        "安値",
        "安値_sp500",
        "安値_nyd",
        "安値_nasdaq",
        "安値_hongkong",
        "安値_shanhi",
        "終値",
        "終値_sp500",
        "終値_nyd",
        "終値_nasdaq",
        "終値_hongkong",
        "終値_shanhi",
        "出来高",
        "売買代金",
        "貸株残高",
        "融資残高",
        "貸借倍率",
        "5DMA",
        "25DMA",
        "5DVMA",
        "25DVMA",
    ]
    for rp in rp_list:
        # CSVデータはカンマ付き文字列のため変換、yfinanceデータは既にfloat64のためスキップ
        if train_target[rp].dtype == object:
            train_target[rp] = train_target[rp].str.replace(",", "").astype("float64")

    # 日足の高さ
    train_target["日足高安"] = (train_target["高値"] - train_target["安値"]) / train_target["終値"] * 100
    train_target["日足高安_sp500"] = (
        (train_target["高値_sp500"] - train_target["安値_sp500"]) / train_target["終値_sp500"] * 100
    )
    train_target["日足高安_nyd"] = (
        (train_target["高値_nyd"] - train_target["安値_nyd"]) / train_target["終値_nyd"] * 100
    )
    train_target["日足高安_nasdaq"] = (
        (train_target["高値_nasdaq"] - train_target["安値_nasdaq"]) / train_target["終値_nasdaq"] * 100
    )
    train_target["日足高安_vix"] = (
        (train_target["高値_vix"] - train_target["安値_vix"]) / train_target["終値_vix"] * 100
    )
    train_target["日足高安_usdyen"] = (
        (train_target["高値_usdyen"] - train_target["安値_usdyen"]) / train_target["終値_usdyen"] * 100
    )
    train_target["日足高安_hongkong"] = (
        (train_target["高値_hongkong"] - train_target["安値_hongkong"]) / train_target["終値_hongkong"] * 100
    )
    train_target["日足高安_shanhi"] = (
        (train_target["高値_shanhi"] - train_target["安値_shanhi"]) / train_target["終値_shanhi"] * 100
    )

    # ヒゲ（上下）の計算
    train_target["上ヒゲ"] = train_target.apply(up_hige_calc, p_start="始値", p_end="終値", p_high="高値", axis=1)
    train_target["上ヒゲ_sp500"] = train_target.apply(
        up_hige_calc, p_start="始値_sp500", p_end="終値_sp500", p_high="高値_sp500", axis=1
    )
    train_target["上ヒゲ_nyd"] = train_target.apply(
        up_hige_calc, p_start="始値_nyd", p_end="終値_nyd", p_high="高値_nyd", axis=1
    )
    train_target["上ヒゲ_nasdaq"] = train_target.apply(
        up_hige_calc, p_start="始値_nasdaq", p_end="終値_nasdaq", p_high="高値_nasdaq", axis=1
    )
    train_target["上ヒゲ_vix"] = train_target.apply(
        up_hige_calc, p_start="始値_vix", p_end="終値_vix", p_high="高値_vix", axis=1
    )
    train_target["上ヒゲ_usdyen"] = train_target.apply(
        up_hige_calc, p_start="始値_usdyen", p_end="終値_usdyen", p_high="高値_usdyen", axis=1
    )
    train_target["上ヒゲ_hongkong"] = train_target.apply(
        up_hige_calc, p_start="始値_hongkong", p_end="終値_hongkong", p_high="高値_hongkong", axis=1
    )
    train_target["上ヒゲ_shanhi"] = train_target.apply(
        up_hige_calc, p_start="始値_shanhi", p_end="終値_shanhi", p_high="高値_shanhi", axis=1
    )
    train_target["下ヒゲ"] = train_target.apply(dw_hige_calc, p_start="始値", p_end="終値", p_low="安値", axis=1)
    train_target["下ヒゲ_sp500"] = train_target.apply(
        dw_hige_calc, p_start="始値_sp500", p_end="終値_sp500", p_low="安値_sp500", axis=1
    )
    train_target["下ヒゲ_nyd"] = train_target.apply(
        dw_hige_calc, p_start="始値_nyd", p_end="終値_nyd", p_low="安値_nyd", axis=1
    )
    train_target["下ヒゲ_nasdaq"] = train_target.apply(
        dw_hige_calc, p_start="始値_nasdaq", p_end="終値_nasdaq", p_low="安値_nasdaq", axis=1
    )
    train_target["下ヒゲ_vix"] = train_target.apply(
        dw_hige_calc, p_start="始値_vix", p_end="終値_vix", p_low="安値_vix", axis=1
    )
    train_target["下ヒゲ_usdyen"] = train_target.apply(
        dw_hige_calc, p_start="始値_usdyen", p_end="終値_usdyen", p_low="安値_usdyen", axis=1
    )
    train_target["下ヒゲ_hongkong"] = train_target.apply(
        dw_hige_calc, p_start="始値_hongkong", p_end="終値_hongkong", p_low="安値_hongkong", axis=1
    )
    train_target["下ヒゲ_shanhi"] = train_target.apply(
        dw_hige_calc, p_start="始値_shanhi", p_end="終値_shanhi", p_low="安値_shanhi", axis=1
    )

    # 終値の1-5日前との差異（比率）を計算する
    train_target["終値_5"] = train_target["終値"].pct_change(-5, fill_method=None)
    train_target["終値_4"] = train_target["終値"].pct_change(-4, fill_method=None)
    train_target["終値_3"] = train_target["終値"].pct_change(-3, fill_method=None)
    train_target["終値_2"] = train_target["終値"].pct_change(-2, fill_method=None)
    train_target["終値_1"] = train_target["終値"].pct_change(-1, fill_method=None)
    train_target["終値_1_sp500"] = train_target["終値_sp500"].pct_change(-1, fill_method=None)
    train_target["終値_1_nyd"] = train_target["終値_nyd"].pct_change(-1, fill_method=None)
    train_target["終値_1_nasdaq"] = train_target["終値_nasdaq"].pct_change(-1, fill_method=None)
    train_target["終値_1_vix"] = train_target["終値_vix"].pct_change(-1, fill_method=None)
    train_target["終値_1_usdyen"] = train_target["終値_usdyen"].pct_change(-1, fill_method=None)
    train_target["終値_1_hongkong"] = train_target["終値_hongkong"].pct_change(-1, fill_method=None)
    train_target["終値_1_shanhi"] = train_target["終値_shanhi"].pct_change(-1, fill_method=None)
    train_target["始値_5"] = train_target["始値"].pct_change(-5, fill_method=None)
    train_target["始値_4"] = train_target["始値"].pct_change(-4, fill_method=None)
    train_target["始値_3"] = train_target["始値"].pct_change(-3, fill_method=None)
    train_target["始値_2"] = train_target["始値"].pct_change(-2, fill_method=None)
    train_target["始値_1"] = train_target["始値"].pct_change(-1, fill_method=None)
    train_target["高値_5"] = train_target["高値"].pct_change(-5, fill_method=None)
    train_target["高値_4"] = train_target["高値"].pct_change(-4, fill_method=None)
    train_target["高値_3"] = train_target["高値"].pct_change(-3, fill_method=None)
    train_target["高値_2"] = train_target["高値"].pct_change(-2, fill_method=None)
    train_target["高値_1"] = train_target["高値"].pct_change(-1, fill_method=None)
    train_target["安値_5"] = train_target["安値"].pct_change(-5, fill_method=None)
    train_target["安値_4"] = train_target["安値"].pct_change(-4, fill_method=None)
    train_target["安値_3"] = train_target["安値"].pct_change(-3, fill_method=None)
    train_target["安値_2"] = train_target["安値"].pct_change(-2, fill_method=None)
    train_target["安値_1"] = train_target["安値"].pct_change(-1, fill_method=None)
    train_target["出来高_3"] = train_target["出来高"].pct_change(-3, fill_method=None)
    train_target["出来高_2"] = train_target["出来高"].pct_change(-2, fill_method=None)
    train_target["出来高_1"] = train_target["出来高"].pct_change(-1, fill_method=None)
    train_target["売買代金_3"] = train_target["売買代金"].pct_change(-3, fill_method=None)
    train_target["売買代金_2"] = train_target["売買代金"].pct_change(-2, fill_method=None)
    train_target["売買代金_1"] = train_target["売買代金"].pct_change(-1, fill_method=None)
    train_target["貸株残高_3"] = train_target["貸株残高"].diff(-3)  # 貸株残高はゼロになることがあるので差異にする
    train_target["貸株残高_2"] = train_target["貸株残高"].diff(-2)
    train_target["貸株残高_1"] = train_target["貸株残高"].diff(-1)
    train_target["融資残高_3"] = train_target["融資残高"].pct_change(-3, fill_method=None)
    train_target["融資残高_2"] = train_target["融資残高"].pct_change(-2, fill_method=None)
    train_target["融資残高_1"] = train_target["融資残高"].pct_change(-1, fill_method=None)

    # データタイプを浮動小数点に
    train_target["逆日歩"] = train_target["逆日歩"].astype(np.float64)
    train_target["前日比"] = train_target["前日比"].astype(np.float64)
    train_target["5DMA_usdyen"] = train_target["5DMA_usdyen"].astype(np.float64)
    train_target["25DMA_usdyen"] = train_target["25DMA_usdyen"].astype(np.float64)

    # 終値との比率で表す
    train_target["5DMA_rt"] = train_target["5DMA"] / train_target["終値"]
    train_target["25DMA_rt"] = train_target["25DMA"] / train_target["終値"]
    train_target["5VDMA_rt"] = train_target["5DVMA"] / train_target["出来高"]
    train_target["25VDMA_rt"] = train_target["25DVMA"] / train_target["出来高"]

    # 曜日
    train_target["week"] = pd.DataFrame({"weekday": pd.to_datetime(train_target["日付"]).dt.dayofweek})

    # 当日の陰陽の終値ベースの比率を計算する（目的変数）
    train_target["陰陽"] = (train_target["終値"] - train_target["始値"]) / train_target["終値"] * 100
    train_target["陰陽"] = train_target["陰陽"].shift(1)  # 目的変数を１日ずらす（翌日の答えを当日に移動する）

    # yfinance使用時は貸株残高・融資残高がすべて0のためpct_changeがNaNになる→dropna対象から除外
    dropna_subset = ["始値_5", "始値_sp500", "始値_nyd", "始値_hongkong", "始値_shanhi"]
    if not use_yfinance:
        dropna_subset += ["貸株残高_3", "融資残高_3"]
    train_target.dropna(
        subset=dropna_subset,
        how="any",
        inplace=True,
    )  # Nanを含む行を削除（陰陽は除く）
    train_target.to_csv("./csv/train_target0501.csv", header=True, index=False)

    # 不要項目の削除
    drop_list = [
        "日付",
        "75DMA",
        "100DMA",
        "200DMA",
        "75DVMA",
    ]
    for dl in drop_list:
        train_target.drop(columns=[dl], inplace=True)

    # 特徴量を正規化 目的変数である陰陽を除く全項目
    ms_ss_list = [
        "始値",
        "高値",
        "安値",
        "終値",
        "前日比",
        "出来高",
        "売買代金",
        "貸株残高",
        "融資残高",
        "貸借倍率",
        "逆日歩",
        "特別空売り料",
        "5DMA",
        "25DMA",
        "5DVMA",
        "25DVMA",
        "始値_sp500",
        "高値_sp500",
        "安値_sp500",
        "終値_sp500",
        "始値_nyd",
        "高値_nyd",
        "安値_nyd",
        "終値_nyd",
        "始値_nasdaq",
        "高値_nasdaq",
        "安値_nasdaq",
        "終値_nasdaq",
        "始値_vix",
        "高値_vix",
        "安値_vix",
        "終値_vix",
        "始値_usdyen",
        "高値_usdyen",
        "安値_usdyen",
        "終値_usdyen",
        "前日比_usdyen",
        "5DMA_usdyen",
        "25DMA_usdyen",
        "始値_hongkong",
        "高値_hongkong",
        "安値_hongkong",
        "終値_hongkong",
        "始値_shanhi",
        "高値_shanhi",
        "安値_shanhi",
        "終値_shanhi",
        "日足高安",
        "日足高安_sp500",
        "日足高安_nyd",
        "日足高安_nasdaq",
        "日足高安_vix",
        "日足高安_usdyen",
        "日足高安_hongkong",
        "日足高安_shanhi",
        "上ヒゲ",
        "上ヒゲ_sp500",
        "上ヒゲ_nyd",
        "上ヒゲ_nasdaq",
        "上ヒゲ_vix",
        "上ヒゲ_usdyen",
        "上ヒゲ_hongkong",
        "上ヒゲ_shanhi",
        "下ヒゲ",
        "下ヒゲ_sp500",
        "下ヒゲ_nyd",
        "下ヒゲ_nasdaq",
        "下ヒゲ_vix",
        "下ヒゲ_usdyen",
        "下ヒゲ_hongkong",
        "下ヒゲ_shanhi",
        "終値_5",
        "終値_4",
        "終値_3",
        "終値_2",
        "終値_1",
        "終値_1_sp500",
        "終値_1_nyd",
        "終値_1_nasdaq",
        "終値_1_vix",
        "終値_1_usdyen",
        "終値_1_hongkong",
        "終値_1_shanhi",
        "始値_5",
        "始値_4",
        "始値_3",
        "始値_2",
        "始値_1",
        "高値_5",
        "高値_4",
        "高値_3",
        "高値_2",
        "高値_1",
        "安値_5",
        "安値_4",
        "安値_3",
        "安値_2",
        "安値_1",
        "出来高_3",
        "出来高_2",
        "出来高_1",
        "売買代金_3",
        "売買代金_2",
        "売買代金_1",
        "貸株残高_3",
        "貸株残高_2",
        "貸株残高_1",
        "融資残高_3",
        "融資残高_2",
        "融資残高_1",
        "5DMA_rt",
        "25DMA_rt",
        "5VDMA_rt",
        "25VDMA_rt",
        "week",
        "陰陽",
    ]
    ms = MinMaxScaler(feature_range=(0, 1))  # データの正規化(0-1の範囲)
    train_target.loc[:, ms_ss_list] = ms.fit_transform(train_target[ms_ss_list])
    ss = StandardScaler()  # データの標準化（正規分布に変換）
    train_target.loc[:, ms_ss_list] = ss.fit_transform(train_target[ms_ss_list])
    print(train_target.describe())  # 各種集計

    # 陰陽予測 トレーニングデータと正解ラベル、予測用DataFrameの作成
    train_X, train_y = (
        train_target.loc[~train_target["陰陽"].isna(), :].drop("陰陽", axis=1),
        train_target.loc[~train_target["陰陽"].isna(), ["陰陽"]],
    )
    test_X = train_target.loc[train_target["陰陽"].isna(), :].drop("陰陽", axis=1)
    test_X.to_csv("./csv/test_predict.csv", header=True, index=False)  # 予測用CSVを保存する

    return train_X, train_y, test_X
