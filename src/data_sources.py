from __future__ import annotations

import math
from typing import Optional
import akshare as ak


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace("%", "").replace(",", "")
        if s in ("", "-", "None", "nan"):
            return None
        return float(s)
    except Exception:
        return None


def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _std_spot(df, market: str) -> list[dict]:
    # code / name
    code_col = _pick_col(df, ["代码", "symbol", "股票代码", "证券代码", "代码简称"])
    name_col = _pick_col(df, ["名称", "name", "股票名称", "证券简称"])

    # price
    price_col = _pick_col(df, ["最新价", "现价", "收盘", "price", "last", "最新"])
    # pct change
    pct_col = _pick_col(df, ["涨跌幅", "涨跌幅(%)", "涨跌幅%", "p_change", "change_percent", "pctChg", "pct_change"])

    # volume / amount
    vol_col = _pick_col(df, ["成交量", "volume", "成交股数", "成交量(股)"])
    amt_col = _pick_col(df, ["成交额", "amount", "成交额(元)", "成交额(港元)", "成交额(美元)"])

    out = []
    for _, r in df.iterrows():
        code = str(r.get(code_col, "")).strip()
        name = str(r.get(name_col, "")).strip()
        price = _to_float(r.get(price_col))
        pct = _to_float(r.get(pct_col))

        vol = _to_float(r.get(vol_col)) if vol_col else None
        amt = _to_float(r.get(amt_col)) if amt_col else None

        # 美股很多数据源不给成交额，估算：price * volume
        if amt is None and price is not None and vol is not None:
            amt = price * vol

        if not code or not name or price is None:
            continue

        out.append({
            "market": market,
            "code": code,
            "name": name,
            "price": price,
            "pct_change": pct,
            "turnover": amt,   # 统一叫 turnover
        })
    return out


def fetch_market_spot(market: str) -> list[dict]:
    """
    market: A / HK / US
    """
    if market == "A":
        # 东方财富A股实时
        df = ak.stock_zh_a_spot_em()
        return _std_spot(df, "A")

    if market == "HK":
        df = ak.stock_hk_spot_em()
        return _std_spot(df, "HK")

    if market == "US":
        # 不同版本 akshare 可能函数名不同，做兜底
        try:
            df = ak.stock_us_spot_em()
        except Exception:
            df = ak.stock_us_spot()
        return _std_spot(df, "US")

    raise ValueError(f"unknown market: {market}")


def fetch_hist(market: str, code: str):
    """
    返回 DataFrame，至少包含: 日期, 开盘, 收盘, 最高, 最低, 成交量/成交额
    """
    if market == "A":
        # A股日线（前复权）
        try:
            return ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
        except Exception:
            return None

    if market == "HK":
        try:
            return ak.stock_hk_hist(symbol=code, period="daily", adjust="qfq")
        except Exception:
            return None

    if market == "US":
        try:
            return ak.stock_us_hist(symbol=code, period="daily", adjust="qfq")
        except Exception:
            # 有些版本用 stock_us_daily
            try:
                return ak.stock_us_daily(symbol=code, adjust="qfq")
            except Exception:
                return None

    return None
