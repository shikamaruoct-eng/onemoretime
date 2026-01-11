from __future__ import annotations

import math


def _fmt(x, nd=2):
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"


def score_breakout(spot_item: dict, feats: dict, cfg: dict) -> dict:
    w = cfg["weights"]
    bcfg = cfg["breakout"]
    tcfg = cfg["trade"]

    if not feats:
        return {"score": None, "action": "观望", "entry": None, "stop": None, "take": None, "reason": "历史数据不足，暂不评估。"}

    px = spot_item["price"]
    prev_high = feats["prev_high"]
    ma20 = feats["ma20"]
    ma60 = feats["ma60"]

    # 1) 突破强度
    confirm_level = prev_high * (1 + bcfg["confirm_buffer"])
    is_breakout = px >= confirm_level
    breakout_score = 0.0
    if is_breakout:
        # 突破越多分越高，但封顶
        breakout_score = min(1.0, (px / confirm_level - 1.0) / 0.03)  # 超过3%视为满分

    # 2) 趋势结构
    trend_score = 0.0
    if ma20 and ma60:
        if px > ma20 and ma20 > ma60:
            trend_score = 1.0
        elif px > ma20:
            trend_score = 0.6
        elif px > ma60:
            trend_score = 0.4
        else:
            trend_score = 0.0

    # 3) 放量确认（成交额/量）
    vol_today = feats.get("vol_today")
    vol20 = feats.get("vol20")
    volume_score = 0.0
    if vol_today is not None and vol20 is not None and vol20 > 0:
        volume_score = 1.0 if vol_today >= vol20 * bcfg["vol_confirm_mult"] else 0.4

    # 4) 风险惩罚：过热乖离
    risk_score = 1.0
    if ma20 and ma20 > 0:
        dev = (px / ma20) - 1.0
        if dev > bcfg["overheat_deviation"]:
            # 过热越多扣越多
            risk_score = max(0.0, 1.0 - (dev - bcfg["overheat_deviation"]) / 0.10)

    # 总分（0-100）
    score = (
        breakout_score * w["breakout"] +
        trend_score * w["trend"] +
        volume_score * w["volume"] +
        risk_score * w["risk"]
    )

    # 动作建议（趋势突破偏好）
    if score >= 75 and is_breakout:
        action = "关注/可分批试仓"
    elif score >= 60:
        action = "观察（等待确认）"
    else:
        action = "观望"

    # 建议入场/止损/止盈
    entry = round(confirm_level, 4)

    # 止损：max(MA20, 突破位) 再下方 buffer
    ref = None
    if tcfg["stop_mode"] == "max(ma20, breakout_level)":
        ref = max(ma20, prev_high) if ma20 and prev_high else (ma20 or prev_high)

    stop = None
    take = None
    if ref:
        stop = ref * (1 - tcfg["stop_buffer"])
        risk = entry - stop
        take = entry + risk * tcfg["take_profit_rr"]

    # 机器原因（可核验、非胡编）
    reason_parts = []
    reason_parts.append(f"突破：当前价 {_fmt(px)}，参考前高 {_fmt(prev_high)}，确认位 {_fmt(confirm_level)}，{'已突破并确认' if is_breakout else '未确认突破'}。")
    reason_parts.append(f"趋势：MA20 {_fmt(ma20)} / MA60 {_fmt(ma60)}，结构{'偏强' if trend_score>=0.6 else '偏弱'}。")

    if vol_today is not None and vol20 is not None and vol20 > 0:
        reason_parts.append(f"量能：今日量/额 {_fmt(vol_today,0)}，20日均值 {_fmt(vol20,0)}，{'放量确认' if volume_score>=1.0 else '量能一般'}。")
    else:
        reason_parts.append("量能：缺少可靠量/额数据，量能确认不计入满分。")

    if ma20 and ma20 > 0:
        dev = (px / ma20) - 1.0
        reason_parts.append(f"风险：距MA20乖离 {_fmt(dev*100,1)}%，{'偏热（扣分）' if risk_score<1.0 else '可接受'}。")

    reason = " ".join(reason_parts)

    return {
        "score": score,
        "action": action,
        "entry": round(entry, 4) if entry else None,
        "stop": round(stop, 4) if stop else None,
        "take": round(take, 4) if take else None,
        "reason": reason
    }
