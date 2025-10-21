# --- 投資 × AI：実データ版 最初の一歩 ---
# 取得: yfinance（トヨタ 7203.T を3年分）
# 特徴量: 5日/25日移動平均
# モデル: 線形回帰（翌日の終値を予測）
# 検証: 学習/テストに時系列で分割、RMSEなど表示
# 簡易バックテスト: 予測 > 今日のClose なら翌日ロング、それ以外はキャッシュ

import os
import math
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ---------- 1) データ取得 ----------
ticker = "7203.T"          # トヨタ。別銘柄に変えてOK（例: "6758.T"＝ソニー）
period = "3y"               # 3年分
print(f"[INFO] Downloading {ticker} ({period}) ...")
df = yf.download(ticker, period=period, auto_adjust=True)  # 株式分割等を調整

if df.empty:
    raise RuntimeError("yfinanceからデータが取れていません。銘柄コード or ネット接続を確認してください。")

# ---------- 2) 特徴量 ----------
df["SMA5"]  = df["Close"].rolling(5).mean()
df["SMA25"] = df["Close"].rolling(25).mean()
df["Ret1"]  = df["Close"].pct_change(1)
df["Vol10"] = df["Ret1"].rolling(10).std()
df = df.dropna().copy()

# 目的変数：翌日の終値
df["Close_t+1"] = df["Close"].shift(-1)
df = df.dropna().copy()

features = ["SMA5", "SMA25", "Ret1", "Vol10"]
X = df[features].values
y = df["Close_t+1"].values

# ---------- 3) 時系列スプリット（学習70% / テスト30%） ----------
split_idx = int(len(df) * 0.7)
X_train, y_train = X[:split_idx], y[:split_idx]
X_test,  y_test  = X[split_idx:], y[split_idx:]
test_index = df.index[split_idx:]
close_test = df["Close"].iloc[split_idx:]

# ---------- 4) 学習・予測 ----------
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------- 5) 評価 ----------
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
mape = (np.abs((y_test - y_pred) / y_test)).mean() * 100

print("\n=== 回帰評価（テスト） ===")
print(f"RMSE: {rmse:.3f}")
print(f"MAE : {mae:.3f}")
print(f"MAPE: {mape:.2f}%")

# ---------- 6) 簡易バックテスト（ロング/キャッシュ） ----------
# ルール：予測値（明日の終値） > 今日の終値 → 翌日ロング（1倍）
# それ以外 → キャッシュ（0）
signal = (y_pred > close_test.values).astype(int)
# 翌日のリターン
ret_next = df["Close"].pct_change().iloc[split_idx+1:]   # テスト開始翌日から
sig_for_ret = signal[:-1]                                 # 翌日のリターンに対して今日の判断を適用
strat_ret = sig_for_ret * ret_next.values
bh_ret = ret_next.values                                  # 買い持ち

def ann_return(ret, periods=252):
    total = (1 + pd.Series(ret)).prod()
    years = len(ret) / periods
    return total**(1/years) - 1 if years > 0 else np.nan

def ann_vol(ret, periods=252):
    return pd.Series(ret).std() * np.sqrt(periods)

ar_strat = ann_return(strat_ret)
ar_bh = ann_return(bh_ret)
vol_strat = ann_vol(strat_ret)
vol_bh = ann_vol(bh_ret)

print("\n=== 簡易バックテスト（テスト期間） ===")
print(f"年率  Strategy: {ar_strat:.2%} / Buy&Hold: {ar_bh:.2%}")
print(f"年率ボラ Strategy: {vol_strat:.2%} / Buy&Hold: {vol_bh:.2%}")

# 累積カーブ
eq_strat = (1 + pd.Series(strat_ret, index=ret_next.index)).cumprod()
eq_bh    = (1 + pd.Series(bh_ret,    index=ret_next.index)).cumprod()

# 出力フォルダ
os.makedirs("reports", exist_ok=True)

# 価格と予測のプロット
plt.figure(figsize=(10,5))
plt.plot(test_index, close_test, label="Actual Close")
plt.plot(test_index, y_pred, label="Predicted (t+1)")
plt.title(f"{ticker} 予測（テスト期間）")
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig("reports/pred_vs_actual.png")
plt.show()

# エクイティカーブ
plt.figure(figsize=(10,5))
plt.plot(eq_bh.index, eq_bh.values, label="Buy & Hold")
plt.plot(eq_strat.index, eq_strat.values, label="AI Strategy (long/flat)")
plt.title("Equity Curve (Test slice)")
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig("reports/equity_curve.png")
plt.show()

# 指標を書き出し
with open("reports/metrics.txt", "w", encoding="utf-8") as f:
    f.write("=== 回帰評価（テスト） ===\n")
    f.write(f"RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nMAPE: {mape:.2f}%\n\n")
    f.write("=== 簡易バックテスト（テスト期間） ===\n")
    f.write(f"年率  Strategy: {ar_strat:.2%} / Buy&Hold: {ar_bh:.2%}\n")
    f.write(f"年率ボラ Strategy: {vol_strat:.2%} / Buy&Hold: {vol_bh:.2%}\n")

print("\n[INFO] 画像と指標を reports/ に保存しました。")
print(" - reports/pred_vs_actual.png")
print(" - reports/equity_curve.png")
print(" - reports/metrics.txt")
