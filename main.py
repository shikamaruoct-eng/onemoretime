import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import akshare as ak

OUT_DIR = "docs"
OUT_HTML = f"{OUT_DIR}/index.html"
OUT_JSON = f"{OUT_DIR}/data.json"

TOP_A = 40
TOP_HK = 20
TOP_US = 40

# --------- 工具：字段清洗 ---------
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def to_num(s):
    try:
        if pd.isna(s):
            return np.nan
        s = str(s).replace("%", "").replace(",", "").strip()
        return float(s)
    except Exception:
        return np.nan

def zscore(x):
    x = pd.Series(x).astype(float)
    if x.std(ddof=0) == 0:
        return x * 0
    return (x - x.mean()) / x.std(ddof=0)

# --------- 评分与“自动原因（规则推理版）” ---------
def build_reason(row):
    parts = []
    parts.append(f"1D涨跌{row['涨跌幅(1D)']}")
    parts.append(f"量能{row['量能状态']}")
    parts.append(f"结构{row['趋势结构']}")
    parts.append(f"指标{row['指标摘要']}")
    parts.append(f"风险{row['风险提示']}")
    return "；".join(parts)

def decide_action(score, trend):
    # 简化动作：分数高+趋势强 => 入场；分数低+趋势弱 => 退出
    if score >= 75 and trend in ("多头", "强势震荡"):
        return "入场"
    if score <= 35 and trend in ("空头", "弱势震荡"):
        return "退出"
    if 60 <= score < 75:
        return "观察"
    return "观察"

def trend_label(price, ma20=None, ma60=None, chg=None):
    if ma20 is None or ma60 is None:
        # 没有均线就用涨跌幅粗判
        if chg is not None and chg >= 3:
            return "强势震荡"
        if chg is not None and chg <= -3:
            return "弱势震荡"
        return "震荡"
    if price >= ma20 >= ma60:
        return "多头"
    if price <= ma20 <= ma60:
        return "空头"
    # 夹在均线之间
    if price >= ma20 and ma20 < ma60:
        return "强势震荡"
    if price <= ma20 and ma20 > ma60:
        return "弱势震荡"
    return "震荡"

def volume_label(turnover):
    if np.isnan(turnover):
        return "未知"
    if turnover >= 5e8:
        return "放量/高流动性"
    if turnover >= 1e8:
        return "正常"
    return "偏低"

def indicator_summary(chg):
    # 先做“可运行版”，后续可升级为真RSI/MACD/KDJ（需要历史数据）
    if chg >= 5:
        return "动量强（短期过热可能）"
    if chg >= 2:
        return "动量偏强"
    if chg <= -5:
        return "动量弱（恐慌/超跌可能）"
    if chg <= -2:
        return "动量偏弱"
    return "中性"

def risk_hint(turnover, chg):
    hints = []
    if not np.isnan(turnover) and turnover < 8e7:
        hints.append("流动性偏低")
    if abs(chg) >= 8:
        hints.append("波动过大")
    return "、".join(hints) if hints else "常规"

def key_levels(price, chg):
    # 简化：用当日波动估算区间；后续可升级为支撑/阻力/ATR
    # 这里先给可用的“入场/目标/风控”三价位
    if chg >= 0:
        entry = price
        tp = price * 1.06
        stop = price * 0.94
    else:
        entry = price * 0.995
        tp = price * 1.05
        stop = price * 0.93
    return f"入场≈{entry:.2f}｜目标≈{tp:.2f}｜风控≈{stop:.2f}"

# --------- 数据抓取：A/HK/US ---------
def fetch_a_spot():
    # 避免单接口偶发分页/条数问题：拆沪/深/京更稳（AKShare教程列出了这些接口）:contentReference[oaicite:4]{index=4}
    dfs = []
    for fn in (ak.stock_sh_a_spot_em, ak.stock_sz_a_spot_em, ak.stock_bj_a_spot_em):
        try:
            dfs.append(fn())
        except Exception:
            continue
    if not dfs:
        raise RuntimeError("A股行情抓取失败")
    df = pd.concat(dfs, ignore_index=True)

    code_col = pick_col(df, ["代码", "股票代码"])
    name_col = pick_col(df, ["名称", "股票名称"])
    price_col = pick_col(df, ["最新价", "最新价(元)", "最新价/元"])
    chg_col = pick_col(df, ["涨跌幅", "涨跌幅(%)"])
    turn_col = pick_col(df, ["成交额", "成交额(元)"])

    out = pd.DataFrame({
        "市场": "A",
        "代码": df[code_col].astype(str),
        "名称": df[name_col].astype(str),
        "当前价": df[price_col].map(to_num),
        "涨跌幅(1D)_num": df[chg_col].map(to_num),
        "成交额_num": df[turn_col].map(to_num) if turn_col else np.nan,
    })
    return out.dropna(subset=["当前价", "涨跌幅(1D)_num"])

def fetch_hk_spot():
    # 港股实时行情（东财，延迟15分钟）:contentReference[oaicite:5]{index=5}
    df = ak.stock_hk_spot_em()
    code_col = pick_col(df, ["代码"])
    name_col = pick_col(df, ["名称"])
    price_col = pick_col(df, ["最新价", "最新价(港元)"])
    chg_col = pick_col(df, ["涨跌幅"])
    turn_col = pick_col(df, ["成交额"])

    out = pd.DataFrame({
        "市场": "HK",
        "代码": df[code_col].astype(str),
        "名称": df[name_col].astype(str),
        "当前价": df[price_col].map(to_num),
        "涨跌幅(1D)_num": df[chg_col].map(to_num),
        "成交额_num": df[turn_col].map(to_num) if turn_col else np.nan,
    })
    return out.dropna(subset=["当前价", "涨跌幅(1D)_num"])

def fetch_us_spot():
    # AKShare 教程列出了美股行情接口（Sina来源）:contentReference[oaicite:6]{index=6}
    df = ak.stock_us_spot()
    # 字段名可能随版本变化，做兼容
    code_col = pick_col(df, ["symbol", "代码", "股票代码"])
    name_col = pick_col(df, ["name", "名称", "股票名称"])
    price_col = pick_col(df, ["price", "最新价", "当前价"])
    chg_col = pick_col(df, ["changepercent", "涨跌幅", "涨跌幅(%)"])
    vol_col = pick_col(df, ["volume", "成交量"])
    # 美股用成交量估算流动性（成交额很多源拿不到）
    out = pd.DataFrame({
        "市场": "US",
        "代码": df[code_col].astype(str),
        "名称": df[name_col].astype(str),
        "当前价": df[price_col].map(to_num),
        "涨跌幅(1D)_num": df[chg_col].map(to_num),
        "成交额_num": df[vol_col].map(to_num) if vol_col else np.nan,
    })
    return out.dropna(subset=["当前价", "涨跌幅(1D)_num"])

# --------- 生成 TopN + 12列 ---------
def build_top(df, topn):
    df = df.copy()
    df["成交额_num"] = df["成交额_num"].astype(float)
    df["涨跌幅(1D)_num"] = df["涨跌幅(1D)_num"].astype(float)

    # 评分：动量 + 流动性（标准化后加权）
    score = 60 * zscore(df["涨跌幅(1D)_num"]).fillna(0) + 40 * zscore(df["成交额_num"]).fillna(0)
    # 压缩到 0-100
    score = (score - score.min()) / (score.max() - score.min() + 1e-9) * 100
    df["评分"] = score.round(0)

    # 趋势/量能/指标/风险（先用“可运行版”，后续再升级需要历史数据的版本）
    df["涨跌幅(1D)"] = df["涨跌幅(1D)_num"].map(lambda x: f"{x:.2f}%")
    df["量能状态"] = df["成交额_num"].map(volume_label)
    df["趋势结构"] = df.apply(lambda r: trend_label(r["当前价"], None, None, r["涨跌幅(1D)_num"]), axis=1)
    df["指标摘要"] = df["涨跌幅(1D)_num"].map(indicator_summary)
    df["风险提示"] = df.apply(lambda r: risk_hint(r["成交额_num"], r["涨跌幅(1D)_num"]), axis=1)
    df["关键价位"] = df.apply(lambda r: key_levels(r["当前价"], r["涨跌幅(1D)_num"]), axis=1)

    df["动作建议"] = df.apply(lambda r: decide_action(r["评分"], r["趋势结构"]), axis=1)
    df["AI原因"] = df.apply(build_reason, axis=1)

    # 选 TopN：按评分降序
    df = df.sort_values("评分", ascending=False).head(topn)

    # 输出 12 列（顺序固定）
    out = pd.DataFrame({
        "市场": df["市场"],
        "代码｜名称": df["代码"] + "｜" + df["名称"],
        "当前价": df["当前价"].map(lambda x: f"{x:.2f}"),
        "涨跌幅(1D)": df["涨跌幅(1D)"],
        "趋势结构": df["趋势结构"],
        "量能状态": df["量能状态"],
        "指标摘要": df["指标摘要"],
        "评分(0-100)": df["评分"].astype(int),
        "动作建议": df["动作建议"],
        "关键价位": df["关键价位"],
        "AI原因": df["AI原因"],
        "风险提示": df["风险提示"],
    })
    return out

def render_html(a_df, hk_df, us_df, generated_at):
    def table_html(df):
        return df.to_html(index=False, escape=False)

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>每日市场Top机会看板</title>
<style>
body{{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial; margin:24px;}}
h1{{margin:0 0 6px 0;}}
.meta{{color:#666;margin-bottom:18px;}}
section{{margin:18px 0 28px;}}
table{{border-collapse:collapse;width:100%; font-size:13px;}}
th,td{{border-bottom:1px solid #eee;padding:8px;text-align:left;vertical-align:top;}}
th{{position:sticky; top:0; background:#fff;}}
</style>
</head>
<body>
<h1>每日市场Top机会看板</h1>
<div class="meta">自动更新于：{generated_at}（UTC→本地已换算）</div>

<section>
<h2>A股 Top{TOP_A}</h2>
{table_html(a_df)}
</section>

<section>
<h2>港股 Top{TOP_HK}</h2>
{table_html(hk_df)}
</section>

<section>
<h2>美股 Top{TOP_US}</h2>
{table_html(us_df)}
</section>

</body>
</html>"""
    return html

def main():
    a = fetch_a_spot()
    hk = fetch_hk_spot()
    us = fetch_us_spot()

    a_top = build_top(a, TOP_A)
    hk_top = build_top(hk, TOP_HK)
    us_top = build_top(us, TOP_US)

    generated_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

    data = {
        "generated_at": generated_at,
        "a_top": a_top.to_dict(orient="records"),
        "hk_top": hk_top.to_dict(orient="records"),
        "us_top": us_top.to_dict(orient="records"),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    html = render_html(a_top, hk_top, us_top, generated_at)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print("OK")

if __name__ == "__main__":
    main()
