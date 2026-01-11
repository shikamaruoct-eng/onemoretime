import json
import os
import time
import math
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import numpy as np
import pandas as pd
import yaml

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


# ---------------- Time & IO ----------------

def now_str(tz_name: str) -> str:
    if ZoneInfo is None:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " (UTC)"
    return datetime.now(ZoneInfo(tz_name)).strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_config() -> dict:
    with open("config.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str, obj: dict):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def f2(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return round(float(x), nd)
    except Exception:
        return None


# ---------------- Observability ----------------

def heartbeat_loop(state: dict, interval: int):
    while True:
        time.sleep(1)
        try:
            now = time.time()
            last = state.get("_hb_last", 0)
            if now - last >= interval:
                state["_hb_last"] = now
                print(
                    f"[heartbeat] stage={state.get('stage')} "
                    f"done={state.get('done')}/{state.get('total')} "
                    f"symbol={state.get('symbol')} "
                    f"elapsed={int(now - state.get('start'))}s",
                    flush=True
                )
        except Exception:
            pass


def call_with_timeout(fn, timeout_sec=20, retries=1, sleep_sec=2, *args, **kwargs):
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


# ---------------- Data: A/HK spot (AkShare) ----------------

def fetch_a_spot():
    import akshare as ak
    df = ak.stock_zh_a_spot_em()
    # 常见列：代码 名称 最新价 涨跌幅 成交额 ...
    # 兼容不同列名
    col_map = {}
    for c in df.columns:
        if c in ("代码", "股票代码"):
            col_map[c] = "code"
        elif c in ("名称", "股票名称"):
            col_map[c] = "name"
        elif c in ("最新价", "最新", "现价"):
            col_map[c] = "price"
        elif c in ("涨跌幅",):
            col_map[c] = "pct"
        elif c in ("成交额", "成交额(元)", "成交额（元）"):
            col_map[c] = "turnover"
        elif c in ("总市值", "总市值(元)", "总市值（元）"):
            col_map[c] = "mcap"
    df = df.rename(columns=col_map)
    for need in ["code", "name", "price"]:
        if need not in df.columns:
            raise RuntimeError(f"A spot missing column: {need}")
    if "turnover" not in df.columns:
        df["turnover"] = np.nan
    if "pct" not in df.columns:
        df["pct"] = np.nan
    out = df[["code", "name", "price", "pct", "turnover"]].copy()
    out["market"] = "A"
    return out


def fetch_hk_spot():
    import akshare as ak
    df = ak.stock_hk_spot_em()
    col_map = {}
    for c in df.columns:
        if c in ("代码", "股票代码"):
            col_map[c] = "code"
        elif c in ("名称", "股票名称"):
            col_map[c] = "name"
        elif c in ("最新价", "最新", "现价"):
            col_map[c] = "price"
        elif c in ("涨跌幅",):
            col_map[c] = "pct"
        elif c in ("成交额", "成交额(元)", "成交额（元）", "成交额(港元)"):
            col_map[c] = "turnover"
    df = df.rename(columns=col_map)
    for need in ["code", "name", "price"]:
        if need not in df.columns:
            raise RuntimeError(f"HK spot missing column: {need}")
    if "turnover" not in df.columns:
        df["turnover"] = np.nan
    if "pct" not in df.columns:
        df["pct"] = np.nan
    out = df[["code", "name", "price", "pct", "turnover"]].copy()
    # HK code 形如 00700，这里统一转为 yfinance 用的 0700.HK
    out["code"] = out["code"].astype(str).str.zfill(5).str[-4:] + ".HK"
    out["market"] = "HK"
    return out


# ---------------- Data: US universe sources (auto) ----------------

def us_universe_nasdaq100():
    """
    Nasdaq-100 constituents via Wikipedia table (auto updated).
    """
    import requests
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    html = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}).text
    tables = pd.read_html(html)
    # 找到包含 Ticker 的表
    best = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("ticker" in c for c in cols) and any("company" in c for c in cols):
            best = t
            break
    if best is None:
        raise RuntimeError("Failed to parse Nasdaq-100 table.")
    # 兼容列名
    ticker_col = None
    name_col = None
    for c in best.columns:
        lc = str(c).lower()
        if "ticker" in lc:
            ticker_col = c
        if "company" in lc:
            name_col = c
    df = pd.DataFrame({
        "code": best[ticker_col].astype(str).str.strip(),
        "name": best[name_col].astype(str).str.strip() if name_col else ""
    })
    df["market"] = "US"
    return df


def us_universe_smh_holdings():
    """
    SMH holdings download (VanEck provides a downloadable holdings file).
    """
    import requests
    # VanEck SMH holdings download endpoint (CSV-like)
    url = "https://www.vaneck.com/us/en/investments/semiconductor-etf-smh/downloads/holdings/"
    text = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}).text
    # 文件通常是CSV，但可能前几行是说明，尝试自动识别
    from io import StringIO
    df = pd.read_csv(StringIO(text))
    # 常见列：Ticker / Holding Name / ... / Daily Holdings (%)
    ticker_col = None
    name_col = None
    for c in df.columns:
        lc = str(c).lower()
        if lc == "ticker" or "ticker" in lc:
            ticker_col = c
        if "holding name" in lc or "name" == lc:
            name_col = c
    if ticker_col is None:
        raise RuntimeError("SMH holdings missing ticker column.")
    out = pd.DataFrame({
        "code": df[ticker_col].astype(str).str.strip(),
        "name": df[name_col].astype(str).str.strip() if name_col else ""
    })
    out = out[out["code"].str.match(r"^[A-Z\.]{1,10}$", na=False)]
    out["market"] = "US"
    return out


def build_us_auto_pool():
    # Nasdaq-100 + SMH holdings
    n100 = us_universe_nasdaq100()
    smh = us_universe_smh_holdings()
    df = pd.concat([n100, smh], ignore_index=True).drop_duplicates(subset=["code"])
    # 去掉特殊符号过多的代码（yfinance 对部分代码支持一般）
    df = df[df["code"].str.len().between(1, 10)]
    df["market"] = "US"
    return df


# ---------------- History Fetch ----------------

def yf_history_batch(symbols, period="1y"):
    import yfinance as yf
    # yfinance 支持批量下载
    data = yf.download(
        tickers=" ".join(symbols),
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    return data


def extract_ohlcv_from_yf_download(data, symbol: str):
    if data is None or len(data) == 0:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        if symbol not in data.columns.levels[0]:
            return None
        d = data[symbol].copy()
    else:
        # 单标的
        d = data.copy()
    d = d.dropna(subset=["Close"])
    if d.empty:
        return None
    d = d.reset_index()
    out = pd.DataFrame({
        "date": pd.to_datetime(d["Date"]).dt.date,
        "open": d["Open"],
        "high": d["High"],
        "low": d["Low"],
        "close": d["Close"],
        "volume": d.get("Volume", np.nan),
    }).dropna(subset=["close"]).sort_values("date")
    return out


def a_history_ak(symbol: str, lookback_days: int):
    import akshare as ak
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - pd.Timedelta(days=max(400, lookback_days + 40))).strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="")
    if df is None or df.empty:
        return None
    df = df.rename(columns={"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    out = df[["date","open","high","low","close","volume"]].dropna(subset=["close"]).sort_values("date")
    return out.tail(lookback_days)


# ---------------- Scoring (per your config) ----------------

def compute_score(df: pd.DataFrame, cfg: dict):
    w = cfg["weights"]
    bcfg = cfg["breakout"]
    filt = cfg["filters"]
    trade = cfg["trade"]

    lookback = int(bcfg["lookback_days"])
    confirm_buf = float(bcfg["confirm_buffer"])
    overheat_dev = float(bcfg["overheat_deviation"])
    vol_mult = float(bcfg["vol_confirm_mult"])

    if df is None or df.empty or len(df) < max(70, lookback + 5):
        return None

    df = df.copy().dropna(subset=["close"]).sort_values("date")
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["vol_avg20"] = df["volume"].rolling(20).mean() if "volume" in df.columns else np.nan

    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(last["close"])
    prev_close = float(prev["close"])
    pct = (close / prev_close - 1.0) * 100.0 if prev_close else None

    ma20 = float(last["ma20"]) if not math.isnan(last["ma20"]) else None
    ma60 = float(last["ma60"]) if not math.isnan(last["ma60"]) else None

    # breakout level: max high over previous lookback days (excluding today)
    window = df.iloc[-(lookback+1):-1]
    breakout_level = float(window["high"].max()) if "high" in window.columns and not window["high"].isna().all() else float(window["close"].max())
    breakout_confirm = close >= breakout_level * (1.0 + confirm_buf)

    # volume ratio
    vol_ratio = None
    if "volume" in df.columns and not math.isnan(last.get("vol_avg20", np.nan)):
        if float(last["vol_avg20"]) > 0:
            vol_ratio = float(last["volume"]) / float(last["vol_avg20"])

    # sub-scores 0..1
    breakout_s = 1.0 if breakout_confirm else 0.0

    trend_s = 0.0
    if ma20 is not None and close > ma20:
        trend_s += 0.5
    if ma20 is not None and ma60 is not None and ma20 > ma60:
        trend_s += 0.5
    trend_s = min(1.0, trend_s)

    volume_s = 0.0
    if vol_ratio is not None:
        volume_s = min(1.0, vol_ratio / vol_mult)

    # risk: overheat penalty
    risk_s = 1.0
    reasons_risk = []
    if ma20 is not None and ma20 > 0:
        dev = close / ma20 - 1.0
        if dev > overheat_dev:
            # linear penalty beyond threshold
            risk_s = max(0.0, 1.0 - (dev - overheat_dev) / max(overheat_dev, 1e-6))
            reasons_risk.append(f"乖离偏高(dev={dev:.1%})")

    # weighted sum
    total = (
        breakout_s * w["breakout"] +
        trend_s * w["trend"] +
        volume_s * w["volume"] +
        risk_s * w["risk"]
    )
    score_5 = (total / 100.0) * 5.0

    # trade levels
    entry = close
    stop_ref = None
    if trade["stop_mode"] == "max(ma20, breakout_level)":
        candidates = [x for x in [ma20, breakout_level] if x is not None]
        stop_ref = max(candidates) if candidates else None
    stop = stop_ref * (1.0 - float(trade["stop_buffer"])) if stop_ref else None
    take = None
    if entry is not None and stop is not None and entry > stop:
        take = entry + (entry - stop) * float(trade["take_profit_rr"])

    # reason text (technical-only, but structured)
    rs = []
    rs.append(f"突破位={breakout_level:.2f}，确认={('是' if breakout_confirm else '否')}")
    if ma20 is not None and ma60 is not None:
        rs.append(f"MA20={ma20:.2f}, MA60={ma60:.2f}")
    if vol_ratio is not None:
        rs.append(f"量能={vol_ratio:.2f}x(阈值{vol_mult})")
    rs.extend(reasons_risk)

    action = "观察"
    if score_5 >= 4.5 and breakout_confirm:
        action = "试仓/关注突破确认"
    elif score_5 >= 3.0:
        action = "观察，等待形态确认"
    else:
        action = "短期无机会"

    return {
        "score": score_5,
        "entry": entry,
        "stop": stop,
        "take": take,
        "close": close,
        "pct": pct,
        "breakout_level": breakout_level,
        "breakout_confirm": breakout_confirm,
        "reason": "；".join(rs) + "。"
    }


# ---------------- Universe builder (AUTO) ----------------

def build_universe_auto(cfg: dict, updated_at: str) -> dict:
    """
    Returns dict: {"A":[{code,name,turnover,pct,price}], "HK":[...], "US":[...]}
    """
    ucfg = cfg["universe"]
    filt = cfg["filters"]

    out = {"A": [], "HK": [], "US": []}

    # A
    a_spot = fetch_a_spot()
    a_spot["turnover"] = pd.to_numeric(a_spot["turnover"], errors="coerce")
    a_spot["pct"] = pd.to_numeric(a_spot["pct"], errors="coerce")
    a_spot = a_spot.dropna(subset=["turnover"])
    a_spot = a_spot[a_spot["turnover"] >= float(ucfg["A"]["min_turnover"])]
    a_spot = a_spot[a_spot["pct"].abs() <= float(filt["max_abs_pct_change"]["A"])]
    a_spot = a_spot.sort_values("turnover", ascending=False).head(int(ucfg["A"]["pool_size"]))
    out["A"] = a_spot[["code","name","price","pct","turnover"]].to_dict("records")

    # HK
    hk_spot = fetch_hk_spot()
    hk_spot["turnover"] = pd.to_numeric(hk_spot["turnover"], errors="coerce")
    hk_spot["pct"] = pd.to_numeric(hk_spot["pct"], errors="coerce")
    hk_spot = hk_spot.dropna(subset=["turnover"])
    hk_spot = hk_spot[hk_spot["turnover"] >= float(ucfg["HK"]["min_turnover"])]
    hk_spot = hk_spot[hk_spot["pct"].abs() <= float(filt["max_abs_pct_change"]["HK"])]
    hk_spot = hk_spot.sort_values("turnover", ascending=False).head(int(ucfg["HK"]["pool_size"]))
    out["HK"] = hk_spot[["code","name","price","pct","turnover"]].to_dict("records")

    # US: Nasdaq-100 + SMH holdings -> then estimate turnover using last 7d batch
    us_df = build_us_auto_pool()
    tickers = us_df["code"].dropna().astype(str).unique().tolist()

    # batch fetch short period to estimate turnover and pct filter
    # chunk to avoid overly long requests
    est = []
    chunk = 60
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i+chunk]
        data, err = call_with_timeout(yf_history_batch, timeout_sec=25, retries=1, tickers=part, period="7d")
        if err or data is None or len(data) == 0:
            continue
        for tkr in part:
            ohlcv = extract_ohlcv_from_yf_download(data, tkr)
            if ohlcv is None or len(ohlcv) < 2:
                continue
            c = float(ohlcv["close"].iloc[-1])
            p = float(ohlcv["close"].iloc[-2])
            pct = (c / p - 1.0) * 100.0 if p else None
            vol = float(ohlcv["volume"].iloc[-1]) if "volume" in ohlcv.columns and not pd.isna(ohlcv["volume"].iloc[-1]) else None
            turnover = c * vol if (c is not None and vol is not None) else None
            est.append({"code": tkr, "name": "", "price": c, "pct": pct, "turnover": turnover})

    us_spot = pd.DataFrame(est)
    if us_spot.empty:
        out["US"] = []
        return out

    us_spot["turnover"] = pd.to_numeric(us_spot["turnover"], errors="coerce")
    us_spot["pct"] = pd.to_numeric(us_spot["pct"], errors="coerce")
    us_spot = us_spot.dropna(subset=["turnover"])
    us_spot = us_spot[us_spot["turnover"] >= float(ucfg["US"]["min_turnover"])]
    us_spot = us_spot[us_spot["pct"].abs() <= float(filt["max_abs_pct_change"]["US"])]
    us_spot = us_spot.sort_values("turnover", ascending=False).head(int(ucfg["US"]["pool_size"]))
    out["US"] = us_spot[["code","name","price","pct","turnover"]].to_dict("records")

    return out


# ---------------- Main pipeline ----------------

def build_dashboard(cfg: dict) -> dict:
    tz = cfg["timezone"]
    updated_at = now_str(tz)

    run_cfg = cfg["run"]
    timeout_sec = int(run_cfg["per_symbol_timeout_sec"])
    retries = int(run_cfg["retries"])
    max_workers = int(run_cfg["max_workers"])
    hb_sec = int(run_cfg["heartbeat_sec"])

    cache_path = cfg["universe"]["cache_path"]

    # heartbeat state
    state = {"stage":"INIT","done":0,"total":0,"symbol":"","start":time.time()}
    threading.Thread(target=heartbeat_loop, args=(state, hb_sec), daemon=True).start()

    # Universe auto update with cache fallback
    print("[stage] build universe(auto)", flush=True)
    uni, err = call_with_timeout(build_universe_auto, timeout_sec=60, retries=0, cfg=cfg, updated_at=updated_at)
    if err or uni is None:
        print(f"[warn] universe auto failed: {err}. fallback to cache.", flush=True)
        cached = load_json(cache_path)
        if not cached:
            raise RuntimeError("Universe build failed and no cache found.")
        uni = cached["universe"]
    else:
        save_json(cache_path, {"updated_at": updated_at, "universe": uni})

    topn = cfg["topn"]
    lookback_days = int(cfg["breakout"]["lookback_days"]) + 80  # ensure enough for MA60 etc.

    rows = []
    skipped = 0

    def process_one(market: str, rec: dict):
        nonlocal skipped
        code = rec["code"]
        name = rec.get("name") or ""
        label = f"{name} {code}".strip()

        # fetch history
        if market == "A":
            df, e = call_with_timeout(a_history_ak, timeout_sec=timeout_sec, retries=retries,
                                      symbol=code, lookback_days=lookback_days)
        else:
            # batch not used here; per symbol for simplicity + timeout protection
            import yfinance as yf
            def _one(sym):
                h = yf.Ticker(sym).history(period="1y", interval="1d", auto_adjust=False)
                if h is None or h.empty:
                    return None
                h = h.reset_index()
                out = pd.DataFrame({
                    "date": pd.to_datetime(h["Date"]).dt.date,
                    "open": h.get("Open"),
                    "high": h.get("High"),
                    "low": h.get("Low"),
                    "close": h.get("Close"),
                    "volume": h.get("Volume"),
                }).dropna(subset=["close"]).sort_values("date")
                return out.tail(lookback_days)

            df, e = call_with_timeout(_one, timeout_sec=timeout_sec, retries=retries, sym=code)

        if e or df is None or df.empty:
            skipped += 1
            return {
                "公司名及代码": label if label else code,
                "所在市场": market,
                "评分": 0,
                "建议入场价": None,
                "止盈价": None,
                "止损价": None,
                "当前价": rec.get("price"),
                "涨跌幅": f2(rec.get("pct")),
                "更新时间": updated_at,
                "动作建议": "跳过",
                "动作原因": f"数据获取失败({e or 'empty'})，已跳过。",
                "我的补充": ""
            }

        s = compute_score(df, cfg)
        if s is None:
            skipped += 1
            return {
                "公司名及代码": label if label else code,
                "所在市场": market,
                "评分": 0,
                "建议入场价": None,
                "止盈价": None,
                "止损价": None,
                "当前价": f2(rec.get("price")),
                "涨跌幅": f2(rec.get("pct")),
                "更新时间": updated_at,
                "动作建议": "观察",
                "动作原因": "历史数据不足，无法计算突破/均线信号。",
                "我的补充": ""
            }

        return {
            "公司名及代码": label if label else code,
            "所在市场": market,
            "评分": f2(s["score"], 1),
            "建议入场价": f2(s["entry"]),
            "止盈价": f2(s["take"]),
            "止损价": f2(s["stop"]),
            "当前价": f2(s["close"]),
            "涨跌幅": f2(s["pct"]),
            "更新时间": updated_at,
            "动作建议": ("试仓/关注突破确认" if (s["score"] >= 4.5 and s["breakout_confirm"]) else
                         ("观察，等待形态确认" if s["score"] >= 3.0 else "短期无机会")),
            "动作原因": s["reason"],
            "我的补充": ""
        }

    # market order
    for market in ["US", "HK", "A"]:
        pool = uni.get(market, [])
        if not pool:
            continue

        state["stage"] = market
        state["done"] = 0
        state["total"] = len(pool)

        print(f"[stage] start {market} pool={len(pool)}", flush=True)

        out_rows = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for rec in pool:
                futures.append((rec["code"], ex.submit(process_one, market, rec)))

            for i, (sym, fut) in enumerate(futures, 1):
                state["done"] = i - 1
                state["symbol"] = sym
                try:
                    r = fut.result()
                    out_rows.append(r)
                    print(f"[progress] {market} {i}/{len(pool)} {sym} OK score={r.get('评分')}", flush=True)
                except Exception as e:
                    skipped += 1
                    print(f"[progress] {market} {i}/{len(pool)} {sym} FAIL {type(e).__name__}: {e}", flush=True)

        # sort by score desc and take TopN
        out_rows_sorted = sorted(out_rows, key=lambda x: float(x.get("评分") or 0), reverse=True)
        rows.extend(out_rows_sorted[:int(topn[market])])

    counts = {"A":0,"HK":0,"US":0}
    for r in rows:
        m = r.get("所在市场")
        if m in counts:
            counts[m] += 1

    return {
        "updated_at_bjt": updated_at,
        "counts": counts,
        "skipped": skipped,
        "rows": rows
    }


def main():
    cfg = load_config()
    payload = build_dashboard(cfg)
    ensure_dir("docs")
    save_json("docs/data.json", payload)
    print("[done] docs/data.json updated.", flush=True)


if __name__ == "__main__":
    main()
