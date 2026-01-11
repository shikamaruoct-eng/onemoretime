from __future__ import annotations

import json
from html import escape


def render_dashboard(data: dict, updated_at: str, out_path: str):
    def table_for(market: str, rows: list[dict]) -> str:
        cols = ["公司名及代码","所在市场","评分","建议入场价","止盈价","止损价","当前价","涨跌幅","更新时间","动作建议","动作原因","我的补充"]
        th = "".join(f"<th>{escape(c)}</th>" for c in cols)
        trs = []
        for r in rows:
            tds = "".join(f"<td>{escape(str(r.get(c,'')))}</td>" for c in cols)
            trs.append(f"<tr>{tds}</tr>")
        return f"""
        <h2>{market}</h2>
        <table>
          <thead><tr>{th}</tr></thead>
          <tbody>{''.join(trs)}</tbody>
        </table>
        """

    html = f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Market Top Dashboard</title>
  <style>
    body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,"PingFang SC","Microsoft YaHei",sans-serif; margin: 20px; }}
    .meta {{ color:#555; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f6f6f6; position: sticky; top: 0; }}
    tr:hover {{ background: #fcfcfc; }}
    .note {{ background:#fffbe6; border:1px solid #ffe58f; padding:10px; border-radius:6px; }}
  </style>
</head>
<body>
  <h1>趋势突破 TopN 看板</h1>
  <div class="meta">更新时间：{escape(updated_at)}（北京时间口径）</div>
  <div class="note">
    数据口径：08:30 生成“截至上一交易日收盘/最新可得数据”的趋势突破筛选结果；非盘中实时信号。动作建议为规则化参考，请结合你的仓位与风控执行。
  </div>

  {table_for("A股 Top40", data.get("A", []))}
  {table_for("港股 Top20", data.get("HK", []))}
  {table_for("美股 Top40", data.get("US", []))}

  <p>原始数据：<a href="./data.json">data.json</a></p>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
