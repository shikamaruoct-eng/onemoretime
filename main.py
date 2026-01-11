from __future__ import annotations

import os
import json
from datetime import datetime
import yaml

from src.data_sources import fetch_market_spot, fetch_hist
from src.features import build_features
from src.filters import apply_filters
from src.scoring import score_breakout
from src.report import render_dashboard


def load_config(path: str = "config.yml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def now_str(tz_name: str) -> str:
    # 不依赖 pytz，使用系统时区信息（GitHub runner 支持 zoneinfo）
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def build_one_market(market: str, cfg: dict, updated_at: str) -> list[dict]:
    topn = cfg["topn"][market]
    spot = fetch_market_spot(market)

    # 统一必要字段并过滤
    spot = apply_filters(spot, market, cfg)

    # 为每只股票取历史数据 -> 计算指标 -> 打分
    rows = []
    for item in spot:
        code = item["code"]
        name = item["name"]
        px = item["price"]
        pct = item.get("pct_change")

        hist = fetch_hist(market, code)
        if hist is None or len(hist) < 80:
            continue

        feats = build_features(hist, lookback=cfg["breakout"]["lookback_days"])
        scored = score_breakout(item, feats, cfg)

        # 生成建议价位与原因
        action = scored["action"]
        entry = scored["entry"]
        stop = scored["stop"]
        take = scored["take"]
        reason = scored["reason"]

        rows.append({
            "公司名及代码": f"{name} ({code})",
            "所在市场": market,
            "评分": round(scored["score"], 2),
            "建议入场价": entry,
            "止盈价": take,
            "止损价": stop,
            "当前价": px,
            "涨跌幅": pct,
            "更新时间": updated_at,
            "动作建议": action,
            "动作原因": reason,
            "我的补充": ""
        })

    # 排序取 TopN
    rows.sort(key=lambda x: (x["评分"] if x["评分"] is not None else -1), reverse=True)
    return rows[:topn]


def main():
    cfg = load_config()
    updated_at = now_str(cfg.get("timezone", "Asia/Shanghai"))

    data = {}
    for market in ["A", "HK", "US"]:
        data[market] = build_one_market(market, cfg, updated_at)

    os.makedirs("docs", exist_ok=True)
    with open("docs/data.json", "w", encoding="utf-8") as f:
        json.dump({"updated_at": updated_at, "data": data}, f, ensure_ascii=False, indent=2)

    render_dashboard(data=data, updated_at=updated_at, out_path="docs/index.html")


if __name__ == "__main__":
    main()
