import pandas as pd
import numpy as np
from typing import List, Dict, Any

def _nav_df(nav_history: List[Dict[str, Any]]):
    df = pd.DataFrame(nav_history)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df.set_index("date", inplace=True)
    return df

def compute_rolling_returns(nav_history: List[Dict[str, Any]], window_years=1) -> Dict[str, float]:
    df = _nav_df(nav_history)
    window = int(252 * window_years)  # 252 trading days per year
    returns = df["nav"].pct_change().add(1).rolling(window).apply(np.prod, raw=True) - 1
    return returns.dropna().to_dict()

def compute_volatility(nav_history: List[Dict[str, Any]], window_years=1) -> Dict[str, float]:
    df = _nav_df(nav_history)
    window = int(252 * window_years)
    vol = df["nav"].pct_change().rolling(window).std() * np.sqrt(252)
    return vol.dropna().to_dict()

def compute_sharpe_ratio(nav_history: List[Dict[str, Any]], risk_free_rate=0.05, window_years=1) -> Dict[str, float]:
    df = _nav_df(nav_history)
    window = int(252 * window_years)
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    excess = df["nav"].pct_change() - daily_rf
    sharpe = excess.rolling(window).mean() / excess.rolling(window).std() * np.sqrt(252)
    return sharpe.dropna().to_dict()

def compute_max_drawdown(nav_history: List[Dict[str, Any]]) -> float:
    df = _nav_df(nav_history)
    roll_max = df["nav"].cummax()
    drawdown = (df["nav"] - roll_max) / roll_max
    return drawdown.min()

def compare_funds(fund_nav_histories: Dict[str, List[Dict]]) -> pd.DataFrame:
    rows = []
    for fund, nav_history in fund_nav_histories.items():
        try:
            rolling = compute_rolling_returns(nav_history)
            vol = compute_volatility(nav_history)
            sharpe = compute_sharpe_ratio(nav_history)
            drawdown = compute_max_drawdown(nav_history)
            rows.append({
                "fund": fund,
                "rolling_return": list(rolling.values())[-1] if rolling else None,
                "volatility": list(vol.values())[-1] if vol else None,
                "sharpe": list(sharpe.values())[-1] if sharpe else None,
                "drawdown": drawdown
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

def rank_funds(fund_nav_histories: Dict[str, List[Dict]], metric: str, top_n=5) -> pd.DataFrame:
    df = compare_funds(fund_nav_histories)
    return df.sort_values(metric, ascending=(metric=="drawdown")).head(top_n)

def scenario_lump_sum(nav_history: List[Dict], amount: float, start_date: str) -> pd.DataFrame:
    df = _nav_df(nav_history)
    df = df[df.index >= pd.to_datetime(start_date)]
    if df.empty:
        return pd.DataFrame()
    units = amount / df["nav"].iloc[0]
    df["investment_value"] = units * df["nav"]
    return df[["investment_value"]]

def scenario_sip(nav_history: List[Dict], amount: float, start_date: str) -> pd.DataFrame:
    df = _nav_df(nav_history)
    df = df[df.index >= pd.to_datetime(start_date)]
    if df.empty:
        return pd.DataFrame()
    units = 0
    values = []
    for i, (date, row) in enumerate(df.iterrows()):
        units += amount / row["nav"]
        values.append(units * row["nav"])
    df["sip_value"] = values
    return df[["sip_value"]]

def plot_comparison_chart(df: pd.DataFrame, metric: str) -> str:
    import matplotlib.pyplot as plt
    import io, base64
    plt.figure(figsize=(8,4))
    plt.bar(df["fund"], df[metric])
    plt.title(f"Fund Comparison: {metric}")
    plt.ylabel(metric)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def plot_scenario_chart(df: pd.DataFrame, value_col: str) -> str:
    import matplotlib.pyplot as plt
    import io, base64
    plt.figure(figsize=(8,4))
    plt.plot(df.index, df[value_col])
    plt.title(f"Scenario Analysis: {value_col}")
    plt.ylabel("Value (₹)")
    plt.xlabel("Date")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8") 