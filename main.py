import os
import time
import json
import math
from datetime import datetime, timedelta, timezone

import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

import requests

# 可选：用于股票池抓取（失败也不影响主流程）
try:
    import akshare as ak
except Exception:
    ak = None

import yfinance as yf


# =========================
# 配置
# =========================
DEFAULT_CFG = {
    "timezone": "Asia/Shanghai",
    "topn": {"A": 40, "HK": 20, "US": 40},
    "breakout": {
        "lookback_days": 60,
        "confirm_buffer": 0.005,
        "overheat_deviation": 0.08,
        "vol_confirm_mult": 1.2,
    },
    "filters": {
        "max_abs_pct_change": {"A": 12, "HK": 15, "US": 12},
    },
    "trade": {
        "stop_mode": "max(ma20, breakout_level)",
        "stop_buffer": 0.02,
        "take_profit_rr": 2.0,
    },
    "runtime": {
        "history_days": 120,
        "max_workers": 8,
        "request_timeout_sec": 20,
        "universe_timeout_sec": 60,
        "heartbeat_sec": 30,
        "soft_fail": True,
    }
}


# =========================
# 工具函数
# =========================
def now_bjt() -> datetime:
    # BJT = UTC+8
    return datetime.now(timezone(timedelta(hours=8)))


def load_cfg(path="config.yml") -> dict:
    cfg = DEFAULT_CFG.copy()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        # 递归合并（浅层足够）
        for k, v in user.items():
            cfg[k] = v
    return cfg


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# =========================
# 股票池（自动 + 缓存 + 保底）
# =========================
UNIVERSE_CACHE_PATH = "data/universe_cache.json"
UNIVERSE_SEED_PATH = "universe_seed.yml"


def universe_from_seed() -> dict:
    if os.path.exists(UNIVERSE_SEED_PATH):
        return read_yaml(UNIVERSE_SEED_PATH)
    return {"A": [], "HK": [], "US": []}


def universe_from_cache() -> dict | None:
    if os.path.exists(UNIVERSE_CACHE_PATH):
        try:
            with open(UNIVERSE_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_universe_cache(u: dict):
    safe_mkdir("data")
    with open(UNIVERSE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(u, f, ensure_ascii=False, indent=2)


def build_universe_auto(cfg: dict) -> dict:
    """
    尽量自动获取：
    - A：用 akshare 拉沪深300成分（若可用）
    - HK：用 akshare 拉港股行情后按成交额取前 N（若可用）
    - US：用 Wikipedia / 公开页面抓取 Nasdaq100（失败就跳）
    任何一步失败都不抛异常，交给上层 fallback。
    """
    t0 = time.time()
    timeout = cfg["runtime"]["universe_timeout_sec"]

    out = {"A": [], "HK": [], "US": []}

    # ---- A：沪深300成分（akshare）----
    try:
        if ak is not None:
            df = ak.index_stock_cons(symbol="000300")  # 沪深300
            # 常见列：品种代码 / 成分券代码 等。做兼容兜底
            col = None
            for c in ["成分券代码", "品种代码", "股票代码", "code"]:
                if c in df.columns:
                    col = c
                    break
            if col:
                codes = df[col].astype(str).str.zfill(6).tolist()
                # 统一成 yfinance ticker
                # 60/68 -> .SS，00/30 -> .SZ（简单规则）
                tickers = []
                for x in codes:
                    if x.startswith(("60", "68")):
                        tickers.append(f"{x}.SS")
                    else:
                        tickers.append(f"{x}.SZ")
                out["A"] = tickers
    except Exception:
        pass

    # ---- HK：按成交额取前 200（akshare）----
    try:
        if ak is not None:
            # 这个接口偶尔慢：失败无所谓
            df = ak.stock_hk_spot_em()
            # 常见列：代码、成交额
            code_col = None
            amt_col = None
            for c in ["代码", "symbol", "股票代码"]:
                if c in df.columns:
                    code_col = c
                    break
            for c in ["成交额", "amount", "turnover"]:
                if c in df.columns:
                    amt_col = c
                    break
            if code_col and amt_col:
                df2 = df[[code_col, amt_col]].copy()
                df2[amt_col] = pd.to_numeric(df2[amt_col], errors="coerce")
                df2 = df2.dropna().sort_values(amt_col, ascending=False).head(200)
                hk = df2[code_col].astype(str).str.zfill(5).tolist()
                out["HK"] = [f"{x}.HK" for x in hk]
    except Exception:
        pass

    # ---- US：抓 Nasdaq 100（Wikipedia read_html）----
    try:
        # 给一个“软超时”：超过 timeout 就算失败
        if time.time() - t0 < timeout:
            url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            html = requests.get(url, timeout=cfg["runtime"]["request_timeout_sec"]).text
            tables = pd.read_html(html)
            # 常见：包含 “Ticker” 或 “Company” 的表
            tickers = []
            for tb in tables:
                cols = [str(c).lower() for c in tb.columns]
                if "ticker" in cols or "ticker symbol" in cols:
                    cidx = cols.index("ticker") if "ticker" in cols else cols.index("ticker symbol")
                    tickers = tb.iloc[:, cidx].astype(str).str.strip().tolist()
                    break
            # 清洗
            tickers = [t.replace(".", "-") for t in tickers if t and t != "nan"]
            if tickers:
                out["US"] = tickers
    except Exception:
        pass

    return out


def build_universe(cfg: dict) -> dict:
    """
    最终策略：
    1) 尝试自动获取
    2) 自动失败 → 用 cache
    3) cache 不存在 → 用 seed（保底）
    且永不抛异常（保证主流程能跑出结果）
    """
    auto = build_universe_auto(cfg)
    ok = {k: len(v) for k, v in auto.items()}

    if all(ok[m] > 0 for m in ["A", "HK", "US"]):
        save_universe_cache(auto)
        return auto

    cache = universe_from_cache()
    if cache and all(len(cache.get(m, [])) > 0 for m in ["A", "HK", "US"]):
        return cache

    seed = universe_from_seed()
    return seed


# =========================
# 行情获取（统一用 yfinance，减少依赖 & 更快）
# =========================
def yf_download_batch(tickers: list[str], start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
    """
    返回每个 ticker 的 DataFrame(OHLCV)
    """
    if not tickers:
        return {}
    # yfinance 批量下载：自动并发，但偶尔返回空；做兼容
    data = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        group_by="ticker",
        threads=True,
        auto_adjust=False,
        progress=False,
    )
    out = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.levels[0]:
                df = data[t].copy()
                df = df.dropna(how="all")
                if len(df) >= 30:
                    out[t] = df
    else:
        # 单 ticker 情况
        df = data.dropna(how="all")
        if len(df) >= 30 and len(tickers) == 1:
            out[tickers[0]] = df
    return out


# =========================
# 指标与评分（趋势突破优先）
# =========================
def compute_signals(df: pd.DataFrame, cfg: dict) -> dict:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    lb = int(cfg["breakout"]["lookback_days"])
    prev_high = high.shift(1).rolling(lb).max()

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else float("nan")
    pct_change = (last_close / prev_close - 1) * 100 if prev_close and not math.isnan(prev_close) else float("nan")

    breakout_level = float(prev_high.iloc[-1]) if not math.isnan(prev_high.iloc[-1]) else float("nan")
    buffer = cfg["breakout"]["confirm_buffer"]
    is_breakout = (not math.isnan(breakout_level)) and (last_close >= breakout_level * (1 + buffer))

    # 放量：量 > 20日均量 * mult
    vol_ma20 = vol.rolling(20).mean()
    vol_mult = cfg["breakout"]["vol_confirm_mult"]
    is_vol_confirm = False
    if not vol_ma20.empty and not math.isnan(float(vol_ma20.iloc[-1] or np.nan)) and float(vol_ma20.iloc[-1]) > 0:
        is_vol_confirm = float(vol.iloc[-1]) >= float(vol_ma20.iloc[-1]) * vol_mult

    # 乖离过热
    dev = (last_close / float(ma20.iloc[-1]) - 1) if not math.isnan(float(ma20.iloc[-1] or np.nan)) else 0.0
    overheat = dev >= cfg["breakout"]["overheat_deviation"]

    trend_ok = (not math.isnan(float(ma20.iloc[-1] or np.nan))) and (not math.isnan(float(ma60.iloc[-1] or np.nan))) and (float(ma20.iloc[-1]) >= float(ma60.iloc[-1]))

    return {
        "last_close": last_close,
        "pct_change": pct_change,
        "ma20": float(ma20.iloc[-1]) if not math.isnan(float(ma20.iloc[-1] or np.nan)) else float("nan"),
        "ma60": float(ma60.iloc[-1]) if not math.isnan(float(ma60.iloc[-1] or np.nan)) else float("nan"),
        "breakout_level": breakout_level,
        "is_breakout": bool(is_breakout),
        "is_vol_confirm": bool(is_vol_confirm),
        "trend_ok": bool(trend_ok),
        "overheat": bool(overheat),
        "last_date": str(df.index[-1].date()),
    }


def score_row(sig: dict, cfg: dict) -> tuple[float, str]:
    """
    输出：评分 0~5 + 自动原因文本
    """
    score = 0.0
    reasons = []

    if sig["is_breakout"]:
        score += 2.5
        reasons.append("突破近60日前高（趋势突破）")
    else:
        reasons.append("未突破关键前高（偏等待）")

    if sig["trend_ok"]:
        score += 1.2
        reasons.append("MA20≥MA60（中期趋势向上）")

    if sig["is_vol_confirm"]:
        score += 0.8
        reasons.append("量能确认（放量）")

    if sig["overheat"]:
        score -= 0.5
        reasons.append("乖离偏高（短线过热，需谨慎）")

    # 涨跌幅异常过滤（只是扣分，不直接剔除）
    # 你早上 08:30 看到的是“昨收变化”，可用来识别极端波动
    score = clamp(score, 0.0, 5.0)

    # 动作建议
    if score >= 4.5:
        action = "强烈买入"
    elif score >= 3.0:
        action = "观察"
    else:
        action = "短期没机会"

    reason_text = f"{action}： " + "；".join(reasons)
    return score, reason_text


def calc_trade_levels(sig: dict, cfg: dict) -> tuple[float, float, float]:
    """
    给出：建议入场、风险控制价、目标价（不是“止盈止损”的措辞）
    """
    entry = sig["last_close"]
    ma20 = sig["ma20"]
    brk = sig["breakout_level"]

    ref = ma20
    if not math.isnan(brk):
        ref = max(ma20 if not math.isnan(ma20) else brk, brk)

    stop_buffer = cfg["trade"]["stop_buffer"]
    risk = ref * (1 - stop_buffer)

    rr = cfg["trade"]["take_profit_rr"]
    target = entry + (entry - risk) * rr
    return float(entry), float(risk), float(target)


# =========================
# HTML 报告（GitHub Pages）
# =========================
HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market Top Dashboard</title>
<style>
body{{font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif; margin:24px;}}
h1{{margin:0 0 6px 0;}}
.meta{{color:#666; margin-bottom:16px;}}
table{{border-collapse:collapse; width:100%; font-size:14px;}}
th,td{{border:1px solid #eee; padding:8px; vertical-align:top;}}
th{{background:#fafafa; cursor:pointer; position:sticky; top:0;}}
.badge{{display:inline-block; padding:2px 8px; border-radius:10px; background:#f2f2f2;}}
.controls{{display:flex; gap:12px; margin:12px 0 16px 0; flex-wrap:wrap;}}
input,select{{padding:8px; border:1px solid #ddd; border-radius:8px;}}
.small{{color:#666; font-size:12px;}}
</style>
</head>
<body>
  <h1>Market Top Dashboard</h1>
  <div class="meta">最后更新时间（北京时间）： <b>{updated_at}</b> ｜ Universe：A {ua} / HK {uh} / US {uu} ｜ 成功行：{rows}</div>

  <div class="controls">
    <select id="marketFilter">
      <option value="">全部市场</option>
      <option value="A">A股</option>
      <option value="HK">港股</option>
      <option value="US">美股</option>
    </select>
    <select id="actionFilter">
      <option value="">全部动作</option>
      <option value="强烈买入">强烈买入</option>
      <option value="观察">观察</option>
      <option value="短期没机会">短期没机会</option>
    </select>
    <input id="searchBox" placeholder="搜索：代码/公司名（若有）" />
    <span class="small">提示：点击表头可排序</span>
  </div>

  <table id="tbl">
    <thead>
      <tr>
        <th>市场</th>
        <th>公司名及代码</th>
        <th>评分</th>
        <th>建议入场价</th>
        <th>风险控制价</th>
        <th>目标价</th>
        <th>当前价</th>
        <th>涨跌幅</th>
        <th>更新时间</th>
        <th>动作建议</th>
        <th>动作原因（自动）</th>
        <th>我的补充</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

<script>
const marketFilter = document.getElementById("marketFilter");
const actionFilter = document.getElementById("actionFilter");
const searchBox = document.getElementById("searchBox");
const tbl = document.getElementById("tbl");
let sortCol = -1;
let sortAsc = true;

function applyFilters() {{
  const m = marketFilter.value;
  const a = actionFilter.value;
  const q = searchBox.value.toLowerCase().trim();

  for (const tr of tbl.tBodies[0].rows) {{
    const market = tr.cells[0].innerText.trim();
    const code = tr.cells[1].innerText.toLowerCase();
    const action = tr.cells[9].innerText.trim();

    let ok = true;
    if (m && market !== m) ok = false;
    if (a && action !== a) ok = false;
    if (q && !code.includes(q)) ok = false;

    tr.style.display = ok ? "" : "none";
  }}
}}

marketFilter.addEventListener("change", applyFilters);
actionFilter.addEventListener("change", applyFilters);
searchBox.addEventListener("input", applyFilters);

function sortTable(col) {{
  const tbody = tbl.tBodies[0];
  const rows = Array.from(tbody.rows);

  if (sortCol === col) sortAsc = !sortAsc;
  else {{ sortCol = col; sortAsc = true; }}

  rows.sort((r1, r2) => {{
    const t1 = r1.cells[col].innerText.trim();
    const t2 = r2.cells[col].innerText.trim();

    const n1 = parseFloat(t1.replace('%',''));
    const n2 = parseFloat(t2.replace('%',''));

    const isNum = !isNaN(n1) && !isNaN(n2);
    let cmp = 0;
    if (isNum) cmp = n1 - n2;
    else cmp = t1.localeCompare(t2);

    return sortAsc ? cmp : -cmp;
  }});

  for (const r of rows) tbody.appendChild(r);
}}

for (let i=0;i<tbl.tHead.rows[0].cells.length;i++) {{
  tbl.tHead.rows[0].cells[i].addEventListener("click", () => sortTable(i));
}}
</script>
</body>
</html>
"""


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.{nd}f}"


def build_html(rows: list[dict], meta: dict) -> str:
    trs = []
    for r in rows:
        trs.append(
            "<tr>"
            f"<td>{r['market']}</td>"
            f"<td>{r['name_code']}</td>"
            f"<td>{fmt(r['score'],1)}</td>"
            f"<td>{fmt(r['entry'],2)}</td>"
            f"<td>{fmt(r['risk'],2)}</td>"
            f"<td>{fmt(r['target'],2)}</td>"
            f"<td>{fmt(r['price'],2)}</td>"
            f"<td>{fmt(r['pct'],2)}%</td>"
            f"<td>{r['update_date']}</td>"
            f"<td>{r['action']}</td>"
            f"<td>{r['reason_auto']}</td>"
            f"<td></td>"
            "</tr>"
        )
    return HTML_TEMPLATE.format(
        updated_at=meta["updated_at"],
        ua=meta["universe_counts"]["A"],
        uh=meta["universe_counts"]["HK"],
        uu=meta["universe_counts"]["US"],
        rows=len(rows),
        rows_html="\n".join(trs),
    )


# =========================
# 主流程
# =========================
def main():
    cfg = load_cfg("config.yml")

    t_all0 = time.time()
    hb = cfg["runtime"]["heartbeat_sec"]

    status = {
        "updated_at_bjt": str(now_bjt().replace(microsecond=0)),
        "stages": {},
        "notes": [],
    }

    def heartbeat(stage, done, total, symbol, t0):
        if hb <= 0:
            return
        elapsed = int(time.time() - t0)
        print(f"[heartbeat] stage={stage} done={done}/{total} symbol={symbol} elapsed={elapsed}s")

    # 1) 股票池
    t0 = time.time()
    universe = build_universe(cfg)
    status["stages"]["universe_sec"] = round(time.time() - t0, 2)
    counts = {k: len(universe.get(k, [])) for k in ["A", "HK", "US"]}
    print(f"[universe] counts={counts}")

    # 2) 拉行情（只要近 120 天日线）
    history_days = int(cfg["runtime"]["history_days"])
    end = datetime.utcnow()
    start = end - timedelta(days=history_days)

    market_outputs = []
    per_market_rows = []

    for market in ["A", "HK", "US"]:
        tickers = universe.get(market, [])
        if not tickers:
            status["notes"].append(f"{market}: empty universe")
            continue

        t_m0 = time.time()
        # 为了速度，限制一次批量下载数量（避免 yfinance 大批量卡死）
        batch_size = 60 if market != "HK" else 50
        all_data = {}

        total = len(tickers)
        done = 0
        last_hb = time.time()

        for i in range(0, total, batch_size):
            batch = tickers[i:i+batch_size]
            try:
                data = yf_download_batch(batch, start, end)
                all_data.update(data)
            except Exception as e:
                status["notes"].append(f"{market}: batch {i}-{i+batch_size} failed: {e}")

            done = min(i + batch_size, total)
            if time.time() - last_hb >= hb:
                heartbeat("download", done, total, batch[-1], t_m0)
                last_hb = time.time()

        status["stages"][f"download_{market}_sec"] = round(time.time() - t_m0, 2)
        print(f"[download] {market}: got={len(all_data)}/{len(tickers)}")

        # 3) 计算评分
        rows = []
        t_s0 = time.time()
        last_hb = time.time()
        keys = list(all_data.keys())

        for idx, tk in enumerate(tqdm(keys, desc=f"score {market}", ncols=90)):
            df = all_data.get(tk)
            if df is None or len(df) < 60:
                continue
            try:
                sig = compute_signals(df, cfg)
                score, reason_auto = score_row(sig, cfg)
                entry, risk, target = calc_trade_levels(sig, cfg)

                action = "强烈买入" if score >= 4.5 else ("观察" if score >= 3.0 else "短期没机会")

                rows.append({
                    "market": market,
                    "name_code": tk,  # 你要求“公司名及代码”，后续可扩展公司名；先保证可用
                    "score": float(score),
                    "entry": entry,
                    "risk": risk,
                    "target": target,
                    "price": sig["last_close"],
                    "pct": float(sig["pct_change"]) if not math.isnan(sig["pct_change"]) else 0.0,
                    "update_date": sig["last_date"],
                    "action": action,
                    "reason_auto": reason_auto,
                })
            except Exception as e:
                if cfg["runtime"]["soft_fail"]:
                    status["notes"].append(f"{market}:{tk} score fail: {e}")
                    continue
                raise

            if time.time() - last_hb >= hb:
                heartbeat("score", idx+1, len(keys), tk, t_s0)
                last_hb = time.time()

        status["stages"][f"score_{market}_sec"] = round(time.time() - t_s0, 2)

        # 4) 排序取 TopN
        topn = int(cfg["topn"][market])
        rows = sorted(rows, key=lambda x: (-x["score"], -x["pct"]))[:topn]
        per_market_rows.extend(rows)

    # 5) 输出 docs（GitHub Pages）
    safe_mkdir("docs")
    meta = {
        "updated_at": str(now_bjt().replace(microsecond=0)),
        "universe_counts": counts
    }

    html = build_html(per_market_rows, meta)
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)

    write_json("docs/data.json", {"meta": meta, "rows": per_market_rows})

    status["total_sec"] = round(time.time() - t_all0, 2)
    write_json("docs/status.json", status)

    print("[done] docs/index.html generated")
    print(f"[done] total_sec={status['total_sec']}")


if __name__ == "__main__":
    main()
