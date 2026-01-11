import json
import math
import os
import time
import traceback
import threading
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import pandas as pd
import numpy as np
import yaml

# --- Timezone (Python 3.12) ---
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


# ============== Utilities ==============

def now_bjt_str(tz_name="Asia/Shanghai"):
    if ZoneInfo is None:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " (UTC)"
    tz = ZoneInfo(tz_name)
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def to_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        s = str(x).strip()
        if s in ("", "nan", "None", "-"):
            return None
        return float(s)
    except Exception:
        return None


def r2(x, nd=2):
    v = to_float(x)
    if v is None:
        return None
    return round(v, nd)


def pct2(x, nd=2):
    v = to_float(x)
    if v is None:
        return None
    return round(v, nd)


# ============== Timeout wrapper ==============

def call_with_timeout(fn, timeout_sec=20, retries=1, sleep_sec=2, *args, **kwargs):
    """
    Runs fn(*args, **kwargs) with a hard timeout.
    Returns (result, err_str). err_str is None on success.
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(fn, *args, **kwargs)
                return fut.result(timeout=timeout_sec), None
        except TimeoutError:
            last_err = f"timeout>{timeout_sec}s"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(sleep_sec * (attempt + 1))
    return None, last_err


# ============== Heartbeat ==============

def heartbeat_loop(state: dict, interval=30):
    while True:
        try:
            time.sleep(1)
            now = time.time()
            last = state.get("_hb_last", 0)
            if now - last >= interval:
                state["_hb_last"] = now
                stage = state.get("stage", "?")
                done = state.get("done", 0)
                total = state.get("total", 0)
                sym = state.get("symbol", "")
                elapsed = int(now - state.get("start", now))
                print(f"[heartbeat] stage={stage} done={done}/{total} symbol={sym} elapsed={elapsed}s", flush=True)
        except Exception:
            # Never break the main job because of heartbeat
            pass


# ============== Technical indicators ==============

def compute_signal(df: pd.DataFrame, breakout_window=20):
    """
    df must contain: date index ascending, columns: close, high, volume(optional)
    """
    if df is None or df.empty or "close" not in df.columns:
        return None

    df = df.copy().dropna(subset=["close"])
    if len(df) < max(60, breakout_window + 5):
        return None

    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()

    # breakout: close >= max(high of previous N days)
    if "high" in df.columns and df["high"].notna().sum() > breakout_window:
        prev_high_n = df["high"].shift(1).rolling(breakout_window).max()
        df["breakout"] = df["close"] >= prev_high_n
    else:
        # fallback: breakout by close
        prev_close_n = df["close"].shift(1).rolling(breakout_window).max()
        df["breakout"] = df["close"] >= prev_close_n

    # volume ratio
    if "volume" in df.columns and df["volume"].notna().sum() > 25:
        df["vol_avg20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_avg20"]
    else:
        df["vol_ratio"] = np.nan

    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(last["close"])
    prev_close = float(prev["close"])
    pct = (close / prev_close - 1.0) * 100.0 if prev_close else None

    ma20 = to_float(last.get("ma20"))
    ma60 = to_float(last.get("ma60"))
    breakout = bool(last.get("breakout", False))
    vol_ratio = to_float(last.get("vol_ratio"))

    # score
    score = 0.0
    reasons = []

    if breakout:
        score += 2.0
        reasons.append(f"突破近{breakout_window}日高点")
    if ma20 is not None and close > ma20:
        score += 1.0
        reasons.append("站上MA20")
    if ma20 is not None and ma60 is not None and ma20 > ma60:
        score += 1.0
        reasons.append("MA20>MA60(多头结构)")
    if vol_ratio is not None and vol_ratio >= 1.5:
        score += 1.0
        reasons.append(f"量能放大{vol_ratio:.1f}x")

    score = min(5.0, score)

    return {
        "close": close,
        "pct": pct,
        "ma20": ma20,
        "ma60": ma60,
        "breakout": breakout,
        "vol_ratio": vol_ratio,
        "score": score,
        "tech_reasons": reasons
    }


def label_action(score: float, breakout: bool):
    if score >= 4.5:
        return "试仓/关注突破确认"
    if score >= 3.0:
        return "观察，等待形态确认"
    return "短期无机会"


def reason_text(tech_reasons, fallback_err=None):
    if fallback_err:
        return f"数据获取失败({fallback_err})，已跳过该标的。"
    if not tech_reasons:
        return "信号不足（未满足突破/均线/量能条件）。"
    return "；".join(tech_reasons) + "。"


# ============== Data Fetchers ==============

def fetch_us_hk_history_yf(symbol: str, lookback_days=180):
    """
    yfinance daily data
    """
    import yfinance as yf
    # period could be '6mo','1y' etc. We'll use 1y for robustness.
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1y", interval="1d", auto_adjust=False)
    if hist is None or hist.empty:
        return None
    hist = hist.reset_index()
    # Normalize columns
    out = pd.DataFrame({
        "date": pd.to_datetime(hist["Date"]).dt.date,
        "open": hist.get("Open"),
        "high": hist.get("High"),
        "low": hist.get("Low"),
        "close": hist.get("Close"),
        "volume": hist.get("Volume"),
    }).dropna(subset=["close"])
    out = out.sort_values("date")
    out = out.tail(lookback_days)
    return out


def fetch_a_history_ak(symbol: str, lookback_days=180):
    """
    akshare daily data for A-share (code only, e.g. 600519)
    """
    import akshare as ak
    # Use last ~1y. ak requires start/end date strings.
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - pd.Timedelta(days=400)).strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="")
    if df is None or df.empty:
        return None
    # columns: 日期 开盘 收盘 最高 最低 成交量 ...
    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.date
    out = df[["date", "open", "high", "low", "close", "volume"]].copy()
    out = out.dropna(subset=["close"]).sort_values("date").tail(lookback_days)
    return out


# ============== Pipeline ==============

def load_config():
    with open("config.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_rows(cfg):
    tz_name = cfg.get("run", {}).get("timezone", "Asia/Shanghai")
    hb_sec = int(cfg.get("run", {}).get("heartbeat_sec", 30))
    timeout_sec = int(cfg.get("run", {}).get("per_symbol_timeout_sec", 20))
    retries = int(cfg.get("run", {}).get("retries", 1))
    max_workers = int(cfg.get("run", {}).get("max_workers", 6))

    lookback_days = int(cfg.get("strategy", {}).get("lookback_days", 140))
    breakout_window = int(cfg.get("strategy", {}).get("breakout_window", 20))
    sl_pct = float(cfg.get("strategy", {}).get("stop_loss_pct", 0.05))
    tp_pct = float(cfg.get("strategy", {}).get("take_profit_pct", 0.10))

    universe = cfg.get("universe", {})

    updated_at = now_bjt_str(tz_name)

    rows = []
    skipped = 0

    # state for heartbeat
    state = {"stage": "INIT", "done": 0, "total": 0, "symbol": "", "start": time.time()}
    t = threading.Thread(target=heartbeat_loop, args=(state, hb_sec), daemon=True)
    t.start()

    def process_one(market: str, symbol: str):
        nonlocal skipped
        stage_name = market

        # fetch
        if market in ("US", "HK"):
            df, err = call_with_timeout(fetch_us_hk_history_yf, timeout_sec=timeout_sec, retries=retries,
                                        symbol=symbol, lookback_days=lookback_days)
        else:  # A
            df, err = call_with_timeout(fetch_a_history_ak, timeout_sec=timeout_sec, retries=retries,
                                        symbol=symbol, lookback_days=lookback_days)

        if err or df is None or df.empty:
            skipped += 1
            return {
                "公司名及代码": symbol,
                "所在市场": market,
                "评分": 0,
                "建议入场价": None,
                "止盈价": None,
                "止损价": None,
                "当前价": None,
                "涨跌幅": None,
                "更新时间": updated_at,
                "动作建议": "跳过",
                "动作原因": reason_text([], fallback_err=err or "empty"),
                "我的补充": ""
            }

        # compute signal
        dfn = df.rename(columns=str.lower)
        # ensure required
        if "close" not in dfn.columns:
            skipped += 1
            return {
                "公司名及代码": symbol,
                "所在市场": market,
                "评分": 0,
                "建议入场价": None,
                "止盈价": None,
                "止损价": None,
                "当前价": None,
                "涨跌幅": None,
                "更新时间": updated_at,
                "动作建议": "跳过",
                "动作原因": "数据字段不完整(close缺失)，已跳过。",
                "我的补充": ""
            }

        sig = compute_signal(dfn, breakout_window=breakout_window)
        if sig is None:
            skipped += 1
            return {
                "公司名及代码": symbol,
                "所在市场": market,
                "评分": 0,
                "建议入场价": None,
                "止盈价": None,
                "止损价": None,
                "当前价": r2(dfn["close"].iloc[-1]),
                "涨跌幅": None,
                "更新时间": updated_at,
                "动作建议": "观察",
                "动作原因": "历史数据不足，无法计算突破/均线信号。",
                "我的补充": ""
            }

        score = float(sig["score"])
        close = r2(sig["close"], 2)
        pct = pct2(sig["pct"], 2)

        action = label_action(score, sig["breakout"])

        # entry/SL/TP (simple rule)
        entry = close
        if sig["ma20"] is not None:
            entry = r2(max(close, sig["ma20"]), 2)

        stop = r2(entry * (1 - sl_pct), 2) if entry else None
        take = r2(entry * (1 + tp_pct), 2) if entry else None

        reason = reason_text(sig["tech_reasons"])

        return {
            "公司名及代码": symbol,
            "所在市场": market,
            "评分": r2(score, 1),
            "建议入场价": entry,
            "止盈价": take,
            "止损价": stop,
            "当前价": close,
            "涨跌幅": pct,
            "更新时间": updated_at,
            "动作建议": action,
            "动作原因": reason,
            "我的补充": ""
        }

    # process markets in order
    for market in ["US", "HK", "A"]:
        symbols = universe.get(market, {}).get("symbols", [])
        if not symbols:
            continue

        state["stage"] = market
        state["done"] = 0
        state["total"] = len(symbols)

        print(f"[stage] start {market} symbols={len(symbols)}", flush=True)

        # Use limited concurrency
        out_rows = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for sym in symbols:
                futures.append((sym, ex.submit(process_one, market, sym)))

            for i, (sym, fut) in enumerate(futures, 1):
                state["done"] = i - 1
                state["symbol"] = sym
                try:
                    r = fut.result()
                    out_rows.append(r)
                    print(f"[progress] {market} {i}/{len(symbols)} {sym} OK score={r.get('评分')}", flush=True)
                except Exception as e:
                    skipped += 1
                    print(f"[progress] {market} {i}/{len(symbols)} {sym} FAIL {type(e).__name__}: {e}", flush=True)
                    out_rows.append({
                        "公司名及代码": sym,
                        "所在市场": market,
                        "评分": 0,
                        "建议入场价": None,
                        "止盈价": None,
                        "止损价": None,
                        "当前价": None,
                        "涨跌幅": None,
                        "更新时间": updated_at,
                        "动作建议": "跳过",
                        "动作原因": f"处理异常({type(e).__name__}: {e})，已跳过。",
                        "我的补充": ""
                    })

        rows.extend(out_rows)

    counts = {
        "A": sum(1 for r in rows if r.get("所在市场") == "A"),
        "HK": sum(1 for r in rows if r.get("所在市场") == "HK"),
        "US": sum(1 for r in rows if r.get("所在市场") == "US"),
    }

    return {
        "updated_at_bjt": updated_at,
        "counts": counts,
        "skipped": skipped,
        "rows": rows
    }


def write_docs(payload):
    safe_mkdir("docs")
    # write data
    with open("docs/data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    cfg = load_config()
    payload = build_rows(cfg)
    write_docs(payload)
    # A hard success condition: at least some rows exist
    if not payload.get("rows"):
        raise RuntimeError("No rows generated. All data sources may have failed.")
    print("[done] docs/data.json updated.", flush=True)


if __name__ == "__main__":
    main()
