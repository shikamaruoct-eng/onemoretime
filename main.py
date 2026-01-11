# -*- coding: utf-8 -*-
import os
import json
import math
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import yaml
from tqdm import tqdm

import akshare as ak


# =========================
# 配置：你的“12列”固定输出
# =========================
COLUMNS = [
    "公司名及代码",
    "所在市场",
    "评分",
    "建议入场价",
    "止盈价",
    "止损价",
    "当前价",
    "涨跌幅",
    "更新时间",
    "动作建议",
    "动作原因",
    "提醒标记",
]


# =========================
# 工具函数
# =========================
def now_beijing_str():
    # 北京时间 = UTC+8
    bj = datetime.now(timezone.utc) + timedelta(hours=8)
    return bj.strftime("%Y-%m-%d %H:%M:%S")


def safe_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, str):
            x = x.replace("%", "").replace(",", "").strip()
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def mkdirp(path: str):
    os.makedirs(path, exist_ok=True)


def heartbeat(stage: str, done: int, total: int, t0: float):
    elapsed = int(time.time() - t0)
    print(f"[heartbeat] stage={stage} done={done}/{total} elapsed={elapsed}s")


# =========================
# 自动股票池：A Top40 / HK Top20 / US Top40
# 逻辑：用 AkShare 的市场快照，按成交额(或估算成交额)排序取 TopN
# =========================
def build_universe_topn(topn_a=40, topn_hk=20, topn_us=40, timeout_soft=60):
    """
    返回 dict:
      {
        "A": [{"code":"600519", "name":"贵州茅台", "price":..., "pct":..., "turnover":...}, ...],
        "HK":[{"code":"0700", "name":"腾讯控股", ...}],
        "US":[{"code":"AAPL", "name":"Apple Inc", ...}],
      }
    """
    t0 = time.time()
    print("==> [1/4] Build universe (TopN) via AkShare snapshots")
    out = {"A": [], "HK": [], "US": []}

    # ---- A股 ----
    try:
        df = ak.stock_zh_a_spot_em()
        # 常见列名：代码 名称 最新价 涨跌幅 成交额
        df = df.copy()
        df["成交额_num"] = df.get("成交额", None)
        if "成交额_num" in df.columns:
            df["成交额_num"] = df["成交额_num"].apply(safe_float)
        else:
            df["成交额_num"] = None

        # 如果成交额缺失，退化用 最新价*成交量 估算（不一定有）
        if df["成交额_num"].isna().all():
            px = df.get("最新价", pd.Series([None] * len(df))).apply(safe_float)
            vol = df.get("成交量", pd.Series([None] * len(df))).apply(safe_float)
            df["成交额_num"] = (px * vol).fillna(0.0)

        df = df.sort_values("成交额_num", ascending=False).head(topn_a)
        for _, r in df.iterrows():
            out["A"].append(
                {
                    "code": str(r.get("代码", "")).strip(),
                    "name": str(r.get("名称", "")).strip(),
                    "price": safe_float(r.get("最新价")),
                    "pct": safe_float(r.get("涨跌幅")),
                    "turnover": safe_float(r.get("成交额_num")),
                }
            )
    except Exception as e:
        print(f"[warn] A universe build failed: {e}")

    heartbeat("UNIVERSE_A", len(out["A"]), topn_a, t0)

    # ---- 港股 ----
    try:
        df = ak.stock_hk_spot_em()
        df = df.copy()

        # 港股代码通常是 0005 / 0700 这种，确保4位或5位字符串
        df["代码_str"] = df.get("代码", "").astype(str).str.replace(".HK", "", regex=False).str.strip()
        df["代码_str"] = df["代码_str"].str.zfill(4)

        df["成交额_num"] = df.get("成交额", None)
        if "成交额_num" in df.columns:
            df["成交额_num"] = df["成交额_num"].apply(safe_float)
        else:
            df["成交额_num"] = None

        if df["成交额_num"].isna().all():
            px = df.get("最新价", pd.Series([None] * len(df))).apply(safe_float)
            vol = df.get("成交量", pd.Series([None] * len(df))).apply(safe_float)
            df["成交额_num"] = (px * vol).fillna(0.0)

        df = df.sort_values("成交额_num", ascending=False).head(topn_hk)
        for _, r in df.iterrows():
            out["HK"].append(
                {
                    "code": str(r.get("代码_str", "")).strip(),
                    "name": str(r.get("名称", "")).strip(),
                    "price": safe_float(r.get("最新价")),
                    "pct": safe_float(r.get("涨跌幅")),
                    "turnover": safe_float(r.get("成交额_num")),
                }
            )
    except Exception as e:
        print(f"[warn] HK universe build failed: {e}")

    heartbeat("UNIVERSE_HK", len(out["HK"]), topn_hk, t0)

    # ---- 美股 ----
    try:
        df = ak.stock_us_spot_em()
        df = df.copy()
        # 常见列名：代码 名称 最新价 涨跌幅 成交额/成交量
        df["成交额_num"] = df.get("成交额", None)
        if "成交额_num" in df.columns:
            df["成交额_num"] = df["成交额_num"].apply(safe_float)
        else:
            df["成交额_num"] = None

        if df["成交额_num"].isna().all():
            px = df.get("最新价", pd.Series([None] * len(df))).apply(safe_float)
            vol = df.get("成交量", pd.Series([None] * len(df))).apply(safe_float)
            df["成交额_num"] = (px * vol).fillna(0.0)

        df = df.sort_values("成交额_num", ascending=False).head(topn_us)
        for _, r in df.iterrows():
            out["US"].append(
                {
                    "code": str(r.get("代码", "")).strip(),
                    "name": str(r.get("名称", "")).strip(),
                    "price": safe_float(r.get("最新价")),
                    "pct": safe_float(r.get("涨跌幅")),
                    "turnover": safe_float(r.get("成交额_num")),
                }
            )
    except Exception as e:
        print(f"[warn] US universe build failed: {e}")

    heartbeat("UNIVERSE_US", len(out["US"]), topn_us, t0)

    # 兜底：如果某个市场完全失败，给一个最小fallback，保证流程不死
    if len(out["A"]) == 0:
        out["A"] = [{"code": "600519", "name": "贵州茅台", "price": None, "pct": None, "turnover": None}]
        print("[warn] A empty -> fallback to minimal list")
    if len(out["HK"]) == 0:
        out["HK"] = [{"code": "0700", "name": "腾讯控股", "price": None, "pct": None, "turnover": None}]
        print("[warn] HK empty -> fallback to minimal list")
    if len(out["US"]) == 0:
        out["US"] = [{"code": "AAPL", "name": "Apple", "price": None, "pct": None, "turnover": None}]
        print("[warn] US empty -> fallback to minimal list")

    return out


# =========================
# 评分与建议（先做“技术面趋势突破”版本）
# 你后续要加基本面/消息面，也是在这里扩展动作原因即可
# =========================
def score_and_action(row):
    """
    输入 row: dict with fields: market, code, name, price, pct
    输出：评分(0-5), 动作建议, 动作原因(技术面为主), 提醒标记
    """
    pct = row.get("pct")
    price = row.get("price")

    # 非交易时段价格可能为空；先做稳健兜底
    if price is None:
        return 3.0, "观察", "当前价缺失（非交易时段/数据源暂不可用），先观察。", "数据缺失"

    # 简化版技术信号（你后续会升级为：突破前高/均线/量能确认）
    # 这里先用“当日涨跌幅 + 市场热度”做一个可用版本
    p = pct if pct is not None else 0.0

    # 评分逻辑（可解释、可扩展）
    if p >= 4:
        score = 4.5
        action = "关注/准备突破"
        reason = f"涨跌幅 {p:.2f}% 较强，疑似趋势走强，适合关注是否出现突破形态。"
        tag = "强势"
    elif p >= 2:
        score = 4.0
        action = "观察（等待确认）"
        reason = f"涨跌幅 {p:.2f}% 偏强，但仍需等待突破确认/回踩企稳信号。"
        tag = "偏强"
    elif p <= -4:
        score = 2.0
        action = "谨慎/回避"
        reason = f"涨跌幅 {p:.2f}% 偏弱，短期风险偏高，优先防守。"
        tag = "偏弱"
    else:
        score = 3.0
        action = "观察"
        reason = f"涨跌幅 {p:.2f}% 中性，暂以观察为主，等趋势突破信号更明确。"
        tag = "中性"

    # 风控价（先给一个“占位可用版本”）：止损=现价-3%，止盈=现价+6%
    entry = price
    stop = price * (1 - 0.03)
    tp = price * (1 + 0.06)

    row["建议入场价"] = round(entry, 4)
    row["止损价"] = round(stop, 4)
    row["止盈价"] = round(tp, 4)

    return score, action, reason, tag


def normalize_code(market: str, code: str):
    market = market.upper()
    c = str(code).strip()

    # A股：AkShare给的是 600000/000001 这种，不含交易所。
    # 这里不强制拼 .SS/.SZ，因为你当前只是要“看板+理由”；后续做K线时再细分。
    if market == "A":
        return c

    # 港股：通常 0700 -> 0700.HK（如果后续要拉K线可用）
    if market == "HK":
        c = c.zfill(4)
        return c

    # 美股：AAPL 原样
    return c


def build_table(universe: dict):
    t0 = time.time()
    print("==> [2/4] Build table rows (score + action)")

    rows = []
    total = sum(len(v) for v in universe.values())
    done = 0

    for market, items in universe.items():
        for it in items:
            done += 1
            if done % 10 == 0:
                heartbeat("SCORE", done, total, t0)

            code = normalize_code(market, it.get("code", ""))
            name = it.get("name", "")

            price = it.get("price")
            pct = it.get("pct")

            row = {
                "公司名及代码": f"{name}({code})" if name else code,
                "所在市场": market,
                "评分": None,
                "建议入场价": None,
                "止盈价": None,
                "止损价": None,
                "当前价": price,
                "涨跌幅": pct,
                "更新时间": now_beijing_str(),
                "动作建议": None,
                "动作原因": None,
                "提醒标记": None,
            }

            score, action, reason, tag = score_and_action(row)
            row["评分"] = score
            row["动作建议"] = action
            row["动作原因"] = reason
            row["提醒标记"] = tag

            rows.append(row)

    df = pd.DataFrame(rows, columns=COLUMNS)

    # 排序：评分高在前
    df = df.sort_values(["评分", "涨跌幅"], ascending=[False, False], na_position="last").reset_index(drop=True)

    # 格式化涨跌幅
    def fmt_pct(x):
        v = safe_float(x, None)
        return "" if v is None else f"{v:.2f}%"

    df["涨跌幅"] = df["涨跌幅"].apply(fmt_pct)
    df["当前价"] = df["当前价"].apply(lambda x: "" if x is None else round(float(x), 4))

    return df


# =========================
# 输出：docs/index.html（GitHub Pages）
# 同时输出 docs/data.csv docs/data.json
# =========================
def render_html(df: pd.DataFrame, title: str):
    # 简单可读的静态页面（不依赖外部JS，避免被墙/加载慢）
    updated = now_beijing_str()
    html_table = df.to_html(index=False, escape=False)

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif; padding: 16px; }}
    h1 {{ margin: 0 0 8px 0; font-size: 20px; }}
    .meta {{ color: #666; margin-bottom: 12px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #e5e5e5; padding: 8px; font-size: 13px; }}
    th {{ background: #fafafa; position: sticky; top: 0; }}
    tr:hover {{ background: #fcfcfc; }}
    .hint {{ margin-top: 12px; color: #666; font-size: 12px; }}
    .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; border:1px solid #ddd; font-size:12px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">更新时间（北京时间）：{updated} <span class="pill">A Top40</span> <span class="pill">HK Top20</span> <span class="pill">US Top40</span></div>
  {html_table}
  <div class="hint">
    说明：当前版本的“动作原因/评分”以技术面（趋势强弱）为主；后续会把“趋势突破（前高/均线/量能确认）”升级为更严格的规则，并增加基本面/消息面模块。
  </div>
</body>
</html>
"""
    return html


def main():
    # 你可以在这里改 TopN
    TOP_A = 40
    TOP_HK = 20
    TOP_US = 40

    # 你希望可容忍的“自动池抓取软超时”
    UNIVERSE_SOFT_TIMEOUT = 60

    print("====================================")
    print("MakeMoreMoney / Market Dashboard Run")
    print("Beijing time:", now_beijing_str())
    print("====================================")

    # 1) 自动股票池
    universe = build_universe_topn(TOP_A, TOP_HK, TOP_US, timeout_soft=UNIVERSE_SOFT_TIMEOUT)

    # 2) 评分与表格
    df = build_table(universe)

    # 3) 输出到 docs/ 给 GitHub Pages
    print("==> [3/4] Write outputs to docs/ (GitHub Pages)")
    mkdirp("docs")
    df.to_csv("docs/data.csv", index=False, encoding="utf-8-sig")
    df.to_json("docs/data.json", orient="records", force_ascii=False, indent=2)

    html = render_html(df, title="Market Top Dashboard（A/HK/US）")
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)

    # 4) 写一个运行状态文件，方便你判断“是否卡住/是否成功”
    print("==> [4/4] Write run_status.json")
    status = {
        "updated_beijing": now_beijing_str(),
        "count_A": len(universe.get("A", [])),
        "count_HK": len(universe.get("HK", [])),
        "count_US": len(universe.get("US", [])),
        "rows": len(df),
    }
    with open("docs/run_status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

    print("DONE. rows =", len(df))
    print("docs/index.html updated.")


if __name__ == "__main__":
    main()
