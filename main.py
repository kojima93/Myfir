# --- 投資 × AI：翌日リターン予測・手数料ゼロ・売買ログ付き ---
# 概要：
# - 予測対象：翌日リターン（%ではなく小数）。売買判断に直結。
# - 手数料は考慮しない（fee=0.0）。
# - 予測リターンが eps を超えたら翌日ロング（それ以外はキャッシュ）。
# - trades.csv に売買の「いつ/どれだけ/いくらで」を保存。
# - 価格＋BUY/SELLマーカー、エクイティカーブ画像を出力。
#
# 必要ライブラリ（requirements.txt）：
# numpy, pandas, matplotlib, scikit-learn, yfinance
　
import os, math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge      # 正則化回帰で安定化
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ========= パラメータ =========
ticker = "9432.T"      # NTT（必要に応じて変更）
period = "3y"          # データ期間
eps = 0.0015           # 予測リターンがこれ以上の時だけ参入（0.15%）
max_pos = 1.0          # 最大ポジション（1.0 = 全額）
alpha = 1.0            # Ridgeの正則化強さ
train_ratio = 0.7      # 学習/テスト分割（時系列）
# ============================

print(f"[INFO] Downloading {ticker} ({period}) ...")
df = yf.download(ticker, period=period, auto_adjust=True)
if df.empty:
    raise RuntimeError("yfinanceからデータ取得に失敗しました。銘柄コード/ネットを確認してください。")

# ---- 特徴量 ----
def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / (down + 1e-12)
    return 100 - (100/(1+rs))

df["Ret1"]   = df["Close"].pct_change(1)
df["SMA5"]   = df["Close"].rolling(5).mean()
df["SMA25"]  = df["Close"].rolling(25).mean()
df["Mom5"]   = df["Close"].pct_change(5)
df["Vol10"]  = df["Ret1"].rolling(10).std()
df["RSI14"]  = rsi(df["Close"], 14)
df = df.dropna().copy()

# ---- 目的変数：翌日リターン（小数）----
df["Ret_t+1"] = df["Close"].pct_change().shift(-1)
df = df.dropna().copy()

features = ["SMA5","SMA25","Mom5","Vol10","RSI14","Ret1"]
X = df[features].copy()
y = df["Ret_t+1"].copy()

# ---- 時系列スプリット ----
split_idx = int(len(df) * train_ratio)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# 形状安定化（どの環境でも1次元に）
y_train = y_train.values.ravel()
y_test  = y_test.values.ravel()

# ---- 標準化（学習データでfit→テストにtransform）----
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ---- 学習・予測 ----
model = Ridge(alpha=alpha, random_state=0)
model.fit(X_train_sc, y_train)
y_pred = model.predict(X_test_sc)  # 予測：翌日リターン（小数）

# ---- 回帰評価（テスト）----
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
# y_test が 0 に近い日があるため安全に計算（参考値）
mape = (np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-9))).mean() * 100

print("\n=== 回帰評価（テスト） ===")
print(f"RMSE: {rmse:.6f}")
print(f"MAE : {mae:.6f}")
print(f"MAPE: {mape:.2f}%")

# ---- 売買ロジック（手数料ゼロ版）----
threshold = eps
raw_signal = (y_pred > threshold).astype(int)   # 0 or 1

# “今日の判断”で“翌日のリターン”を取る → 1日ずらす
signal_for_ret = raw_signal[:-1]
ret_next = df["Ret1"].iloc[split_idx+1:].values  # 実際の翌日リターン

# インデックス整備
test_idx = df.index[split_idx:]                 # 判定日のindex
exec_idx = df.index[split_idx+1:]               # 約定日のindex（翌日）
price_exec = df["Close"].loc[exec_idx].values   # 約定価格（翌日終値）

# ---- 売買とエクイティ ----
equity = 1.0
equity_curve = []
trades = []

for i in range(len(signal_for_ret)):
    desired = float(signal_for_ret[i]) * max_pos   # 0 or 1
    prev_desired = (float(signal_for_ret[i-1]) * max_pos) if i>0 else 0.0

    # 変化あればトレード記録（手数料はゼロ）
    if desired != prev_desired:
        size_change = desired - prev_desired
        action = "BUY" if size_change > 0 else "SELL"
        trades.append({
            "date": exec_idx[i].strftime("%Y-%m-%d"),
            "action": action,
            "size_change": round(float(size_change),4),
            "position_after": round(desired,4),
            "price": float(price_exec[i]),
            "pred_ret": float(y_pred[i]),
            "threshold": float(threshold),
            "note": ""
        })
    # 翌日の損益（desiredは“今日の判断”＝翌日に適用）
    equity *= (1 + desired * ret_next[i])
    equity_curve.append(equity)

eq_series = pd.Series(equity_curve, index=exec_idx[:len(equity_curve)])

# ---- 買い持ち（同期間）----
bh_ret = ret_next
bh_eq = (1 + pd.Series(bh_ret, index=exec_idx)).cumprod()

# ---- 年率・ボラ ----
def ann_return(ret, periods=252):
    if len(ret) == 0: return np.nan
    total = (1 + pd.Series(ret)).prod()
    years = len(ret) / periods
    return total**(1/years) - 1 if years > 0 else np.nan

def ann_vol(ret, periods=252):
    return pd.Series(ret).std() * np.sqrt(periods)

# 戦略リターン系列（0/1 * 翌日リターン）
strat_ret_series = pd.Series(signal_for_ret, index=exec_idx[:len(signal_for_ret)]) * pd.Series(ret_next, index=exec_idx)
ar_strat = ann_return(strat_ret_series.values)
ar_bh    = ann_return(bh_ret)
vol_strat = ann_vol(strat_ret_series.values)
vol_bh    = ann_vol(bh_ret)

print("\n=== 簡易バックテスト（テスト期間） ===")
print(f"年率  Strategy: {ar_strat:.2%} / Buy&Hold: {ar_bh:.2%}")
print(f"年率ボラ Strategy: {vol_strat:.2%} / Buy&Hold: {vol_bh:.2%}")
print(f"取引回数: {len(trades)}")

# ---- 出力保存 ----
os.makedirs("reports", exist_ok=True)

# 1) 売買履歴
trades_df = pd.DataFrame(trades, columns=[
    "date","action","size_change","position_after","price","pred_ret","threshold","note"
])
trades_df.to_csv("reports/trades.csv", index=False)

# 2) メトリクス
with open("reports/metrics.txt", "w", encoding="utf-8") as f:
    f.write("=== 回帰評価（テスト） ===\n")
    f.write(f"RMSE: {rmse:.6f}\nMAE: {mae:.6f}\nMAPE: {mape:.2f}%\n\n")
    f.write("=== 簡易バックテスト（テスト期間） ===\n")
    f.write(f"年率  Strategy: {ar_strat:.2%} / Buy&Hold: {ar_bh:.2%}\n")
    f.write(f"年率ボラ Strategy: {vol_strat:.2%} / Buy&Hold: {vol_bh:.2%}\n")
    f.write(f"取引回数: {len(trades_df)}\n")
    f.write(f"閾値: eps = {eps:.4f}\n")

# 3) 価格＋売買マーカー
plt.figure(figsize=(10,5))
plt.plot(df.index, df["Close"], label="Close")
buy_idx  = [t["date"] for t in trades if t["action"]=="BUY"]
sell_idx = [t["date"] for t in trades if t["action"]=="SELL"]
if len(buy_idx)>0:
    buy_px  = df.loc[pd.to_datetime(buy_idx),  "Close"]
    plt.scatter(buy_px.index,  buy_px.values,  marker="^", s=80, label="BUY",  zorder=5)
if len(sell_idx)>0:
    sell_px = df.loc[pd.to_datetime(sell_idx), "Close"]
    plt.scatter(sell_px.index, sell_px.values, marker="v", s=80, label="SELL", zorder=5)
plt.title(f"{ticker} Price with Trades")
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig("reports/price_with_trades.png")
plt.close()

# 4) エクイティカーブ
plt.figure(figsize=(10,5))
plt.plot(eq_series.index, eq_series.values, label="AI Strategy")
plt.plot(bh_eq.index, bh_eq.values, label="Buy & Hold")
plt.title("Equity Curve (Test slice)")
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig("reports/equity_curve.png")
plt.close()

print("\n[INFO] 保存:")
print(" - reports/trades.csv")
print(" - reports/price_with_trades.png")
print(" - reports/equity_curve.png")
print(" - reports/metrics.txt")