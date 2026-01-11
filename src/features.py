from __future__ import annotations

import numpy as np


def _to_num(s):
    try:
        if s is None:
            return np.nan
        if isinstance(s, (int, float)):
            return float(s)
        ss = str(s).replace(",", "").replace("%", "").strip()
        if ss in ("", "-", "None", "nan"):
            return np.nan
        return float(ss)
    except Exception:
        return np.nan


def build_features(hist_df, lookback: int = 60) -> dict:
    df = hist_df.copy()

    # 兼容列名
    date_col = None
    for c in ["日期", "date", "时间"]:
        if c in df.columns:
            date_col = c
            break

    close_col = None
    for c in ["收盘", "close", "最新价"]:
        if c in df.columns:
            close_col = c
            break

    high_col = None
    for c in ["最高", "high"]:
        if c in df.columns:
            high_col = c
            break

    low_col = None
    for c in ["最低", "low"]:
        if c in df.columns:
            low_col = c
            break

    vol_col = None
    for c in ["成交量", "volume"]:
        if c in df.columns:
            vol_col = c
            break

    amt_col = None
    for c in ["成交额", "amount"]:
        if c in df.columns:
            amt_col = c
            break

    if close_col is None or high_col is None or low_col is None:
        return {}

    close = df[close_col].map(_to_num).astype(float).to_numpy()
    high = df[high_col].map(_to_num).astype(float).to_numpy()
    low = df[low_col].map(_to_num).astype(float).to_numpy()

    n = len(df)
    if n < max(80, lookback + 5):
        return {}

    # MA
    def ma(arr, w):
        out = np.full_like(arr, np.nan, dtype=float)
        for i in range(w - 1, len(arr)):
            out[i] = np.nanmean(arr[i - w + 1:i + 1])
        return out

    ma20 = ma(close, 20)
    ma60 = ma(close, 60)

    # 20/60日前高（不含当日）
    lb = lookback
    prev_high = np.nanmax(high[-(lb + 1):-1])

    # 量能确认：优先成交额，否则成交量
    vol = None
    if amt_col is not None:
        vol = df[amt_col].map(_to_num).astype(float).to_numpy()
    elif vol_col is not None:
        vol = df[vol_col].map(_to_num).astype(float).to_numpy()

    vol20 = np.nan
    vol_today = np.nan
    if vol is not None and len(vol) >= 21:
        vol20 = np.nanmean(vol[-21:-1])
        vol_today = vol[-1]

    # ATR14
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr14 = np.nan
    if len(tr) >= 14:
        atr14 = np.nanmean(tr[-14:])

    # 最新值
    last_close = float(close[-1])
    last_ma20 = float(ma20[-1])
    last_ma60 = float(ma60[-1])

    return {
        "last_close": last_close,
        "prev_high": float(prev_high),
        "ma20": last_ma20,
        "ma60": last_ma60,
        "vol_today": float(vol_today) if vol_today == vol_today else None,
        "vol20": float(vol20) if vol20 == vol20 else None,
        "atr14": float(atr14) if atr14 == atr14 else None,
    }
