from __future__ import annotations


def apply_filters(spot: list[dict], market: str, cfg: dict) -> list[dict]:
    fcfg = cfg["filters"]
    min_turnover = fcfg["min_turnover"][market]
    max_abs = fcfg["max_abs_pct_change"][market]

    out = []
    for it in spot:
        turnover = it.get("turnover")
        pct = it.get("pct_change")

        # 流动性过滤
        if turnover is None or turnover < min_turnover:
            continue

        # 异常波动过滤（pct 可能为空，空则不拦）
        if pct is not None and abs(pct) > max_abs:
            continue

        out.append(it)

    return out
