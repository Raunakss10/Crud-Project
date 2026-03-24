# %% [markdown]
# When EIA reports a big crude inventory draw/build, does the WTI front spread (CL1–CL2) systematically widen/tighten over the next few days?

# %%
#!pip install pandas numpy matplotlib requests nasdaq-data-link
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import nasdaqdatalink as ndl

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 140)


# %%
import yfinance as yf
import pandas as pd

cl = yf.download("CL=F", start="2015-01-01")

print(cl.columns)  # debug

df = cl[['Close']].rename(columns={'Close': 'price'})

df['spread'] = df['price'] - df['price'].shift(5)

df = df.reset_index()
df = df[['Date', 'spread']].rename(columns={'Date': 'date'})

df.to_csv("wti_calendar_spread.csv", index=False)

print(df.head())

# %%
# =========================
# 1. Imports
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import requests

plt.rcParams["figure.figsize"] = (12, 6)


# =========================
# 2. Config
# =========================
EIA_API_KEY = "QGtfNcfX3uB0ARVUsdDy14AUICsoY9Wj16nTdb6e"


# =========================
# 3. Load EIA data from API
# =========================
def load_eia_api(api_key, start_date="2010-01-01"):
    """
    Pull full weekly US crude oil inventories from EIA API.

    Returns:
        DataFrame with columns:
        - release_date
        - actual
    """
    url = (
        "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
        f"?api_key={api_key}"
        "&frequency=weekly"
        "&data[0]=value"
        "&facets[series][]=WCESTUS1"
        f"&start={start_date}"
        "&sort[0][column]=period"
        "&sort[0][direction]=asc"
        "&offset=0"
        "&length=5000"
    )

    r = requests.get(url, timeout=30)
    data = r.json()

    if "response" not in data or "data" not in data["response"]:
        raise ValueError(f"Unexpected EIA API response: {data}")

    df = pd.DataFrame(data["response"]["data"])

    if df.empty:
        raise ValueError("EIA API returned no data.")

    df = df.rename(columns={
        "period": "release_date",
        "value": "actual"
    })

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")

    df = (
        df[["release_date", "actual"]]
        .dropna()
        .sort_values("release_date")
        .reset_index(drop=True)
    )

    return df


# =========================
# 4. Load + clean spread file
# =========================
def load_spread_file(path="wti_calendar_spread.csv"):
    """
    Cleans spread csv.

    Expected final output:
    date, spread
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    possible_date_cols = [c for c in df.columns if "date" in c.lower()]
    possible_spread_cols = [c for c in df.columns if "spread" in c.lower()]

    if len(possible_date_cols) == 0:
        df = df.rename(columns={df.columns[0]: "date"})
    else:
        df = df.rename(columns={possible_date_cols[0]: "date"})

    if len(possible_spread_cols) == 0:
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[1]: "spread"})
        else:
            raise ValueError("Could not identify spread column in wti_calendar_spread.csv")
    else:
        df = df.rename(columns={possible_spread_cols[0]: "spread"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["spread"] = pd.to_numeric(df["spread"], errors="coerce")

    df = (
        df[["date", "spread"]]
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df


# =========================
# 5. Build surprise series
# =========================
def build_inventory_surprise(eia_df, lookback_expected=4, zscore_window=26):
    """
    expected = rolling mean of previous releases
    surprise = actual - expected
    surprise_z = rolling z-score of surprise
    """
    df = eia_df.copy().sort_values("release_date").reset_index(drop=True)

    df["expected"] = (
        df["actual"]
        .shift(1)
        .rolling(lookback_expected, min_periods=lookback_expected)
        .mean()
    )

    df["surprise"] = df["actual"] - df["expected"]

    rolling_mean = (
        df["surprise"]
        .shift(1)
        .rolling(zscore_window, min_periods=max(8, zscore_window // 3))
        .mean()
    )

    rolling_std = (
        df["surprise"]
        .shift(1)
        .rolling(zscore_window, min_periods=max(8, zscore_window // 3))
        .std()
    )

    df["surprise_z"] = (df["surprise"] - rolling_mean) / rolling_std.replace(0, np.nan)

    return df


# =========================
# 6. Align events to market
# =========================
def align_events_to_market(eia_df, spread_df):
    trading_days = pd.DataFrame({"date": spread_df["date"].sort_values().unique()})

    aligned = pd.merge_asof(
        eia_df.sort_values("release_date"),
        trading_days.sort_values("date"),
        left_on="release_date",
        right_on="date",
        direction="forward"
    )

    aligned = aligned.rename(columns={"date": "trade_date"})
    return aligned


# =========================
# 7. Generate signals
# =========================
def generate_signals(events_df, z_threshold=1.0):
    """
    Signal logic:
    - Large negative inventory surprise z-score => long spread
    - Large positive inventory surprise z-score => short spread

    Fallback:
    - If z-score is NaN early in history, use raw surprise sign
    """
    df = events_df.copy()
    df["signal"] = 0

    has_z = df["surprise_z"].notna()
    df.loc[has_z & (df["surprise_z"] <= -z_threshold), "signal"] = 1
    df.loc[has_z & (df["surprise_z"] >= z_threshold), "signal"] = -1

    no_z = df["surprise_z"].isna() & df["surprise"].notna()
    df.loc[no_z & (df["surprise"] < 0), "signal"] = 1
    df.loc[no_z & (df["surprise"] > 0), "signal"] = -1

    return df


# =========================
# 8. Build positions
# =========================
def build_positions(spread_df, events_df, hold_days=5):
    """
    Hold each event signal for N trading days.
    Clip overlapping signals to -1/0/1.
    """
    df = spread_df.copy().set_index("date")
    df["position"] = 0.0
    df["event_signal"] = 0.0

    valid_events = events_df.dropna(subset=["trade_date"]).copy()

    for _, row in valid_events.iterrows():
        signal = row["signal"]
        trade_date = pd.Timestamp(row["trade_date"])

        if signal == 0 or trade_date not in df.index:
            continue

        entry_idx = df.index.get_loc(trade_date)
        exit_idx = min(entry_idx + hold_days, len(df))

        df.iloc[entry_idx:exit_idx, df.columns.get_loc("position")] += signal
        df.iloc[entry_idx, df.columns.get_loc("event_signal")] = signal

    df["position"] = np.sign(df["position"])

    return df.reset_index()


# =========================
# 9. Backtest
# =========================
def run_backtest(position_df, transaction_cost_bps=5.0):
    """
    Daily strategy PnL = lagged position * daily spread change - transaction costs
    """
    df = position_df.copy().sort_values("date").reset_index(drop=True)

    df["spread_return"] = df["spread"].diff()
    df["position_lag"] = df["position"].shift(1).fillna(0)

    df["gross_pnl"] = df["position_lag"] * df["spread_return"].fillna(0)

    turnover = (df["position"] - df["position_lag"]).abs()
    df["transaction_cost"] = turnover * (transaction_cost_bps / 10000.0)

    df["net_pnl"] = df["gross_pnl"] - df["transaction_cost"]
    df["cum_pnl"] = df["net_pnl"].cumsum()

    return df


# =========================
# 10. Trade-level metrics
# =========================
def compute_trade_table(bt_df):
    """
    Build trade-level PnL table from daily backtest.
    """
    df = bt_df.copy().reset_index(drop=True)
    trades = []

    in_trade = False
    start_idx = None
    current_pos = 0

    for i in range(len(df)):
        pos = df.loc[i, "position"]

        if not in_trade and pos != 0:
            in_trade = True
            start_idx = i
            current_pos = pos

        elif in_trade and pos == 0:
            end_idx = i - 1
            trade_slice = df.loc[start_idx:end_idx].copy()

            trades.append({
                "entry_date": df.loc[start_idx, "date"],
                "exit_date": df.loc[end_idx, "date"],
                "position": current_pos,
                "holding_days": len(trade_slice),
                "trade_pnl": trade_slice["net_pnl"].sum(),
                "gross_trade_pnl": trade_slice["gross_pnl"].sum(),
            })

            in_trade = False
            start_idx = None
            current_pos = 0

        elif in_trade and pos != current_pos:
            end_idx = i - 1
            trade_slice = df.loc[start_idx:end_idx].copy()

            trades.append({
                "entry_date": df.loc[start_idx, "date"],
                "exit_date": df.loc[end_idx, "date"],
                "position": current_pos,
                "holding_days": len(trade_slice),
                "trade_pnl": trade_slice["net_pnl"].sum(),
                "gross_trade_pnl": trade_slice["gross_pnl"].sum(),
            })

            start_idx = i
            current_pos = pos

    if in_trade:
        end_idx = len(df) - 1
        trade_slice = df.loc[start_idx:end_idx].copy()

        trades.append({
            "entry_date": df.loc[start_idx, "date"],
            "exit_date": df.loc[end_idx, "date"],
            "position": current_pos,
            "holding_days": len(trade_slice),
            "trade_pnl": trade_slice["net_pnl"].sum(),
            "gross_trade_pnl": trade_slice["gross_pnl"].sum(),
        })

    return pd.DataFrame(trades)


# =========================
# 11. Metrics
# =========================
def compute_metrics(bt_df, annualization=252):
    pnl = bt_df["net_pnl"].dropna()

    trade_table = compute_trade_table(bt_df)

    if len(pnl) == 0 or pnl.std() == 0:
        return {
            "total_pnl": 0.0,
            "annualized_return": 0.0,
            "annualized_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "num_trading_days": int(len(pnl)),
            "num_nonzero_position_days": int((bt_df["position"] != 0).sum()),
            "num_trades": int(len(trade_table)),
            "trade_win_rate": 0.0,
            "avg_trade_pnl": 0.0,
        }, trade_table

    equity = pnl.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max

    trade_win_rate = 0.0
    avg_trade_pnl = 0.0
    if not trade_table.empty:
        trade_win_rate = float((trade_table["trade_pnl"] > 0).mean())
        avg_trade_pnl = float(trade_table["trade_pnl"].mean())

    metrics = {
        "total_pnl": float(pnl.sum()),
        "annualized_return": float(pnl.mean() * annualization),
        "annualized_vol": float(pnl.std() * np.sqrt(annualization)),
        "sharpe": float((pnl.mean() / pnl.std()) * np.sqrt(annualization)),
        "max_drawdown": float(drawdown.min()),
        "num_trading_days": int(len(pnl)),
        "num_nonzero_position_days": int((bt_df["position"] != 0).sum()),
        "num_trades": int(len(trade_table)),
        "trade_win_rate": trade_win_rate,
        "avg_trade_pnl": avg_trade_pnl,
    }

    return metrics, trade_table


# =========================
# 12. Event-study panel
# =========================
def build_event_study_panel(spread_df, events_df, pre_days=3, post_days=5):
    spread = spread_df.copy().set_index("date")
    rows = []

    for _, event in events_df.dropna(subset=["trade_date"]).iterrows():
        trade_date = pd.Timestamp(event["trade_date"])
        if trade_date not in spread.index:
            continue

        center_loc = spread.index.get_loc(trade_date)
        left = max(0, center_loc - pre_days)
        right = min(len(spread) - 1, center_loc + post_days)

        window = spread.iloc[left:right + 1].copy().reset_index()
        event_level = spread.loc[trade_date, "spread"]

        window["event_date"] = trade_date
        window["t"] = np.arange(left - center_loc, right - center_loc + 1)
        window["spread_change_from_event"] = window["spread"] - event_level
        window["signal"] = event.get("signal", 0)
        window["surprise_z"] = event.get("surprise_z", np.nan)

        rows.append(window)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# =========================
# 13. Plot helpers
# =========================
def get_active_window(bt_df, buffer_days=30):
    active = bt_df[bt_df["position"] != 0]
    if active.empty:
        return bt_df.copy()

    start = active["date"].min() - pd.Timedelta(days=buffer_days)
    end = active["date"].max() + pd.Timedelta(days=buffer_days)

    return bt_df[(bt_df["date"] >= start) & (bt_df["date"] <= end)].copy()


# =========================
# 14. Plots
# =========================
def plot_results(bt_df, events_df=None, event_panel_df=None):
    bt_plot = get_active_window(bt_df, buffer_days=30)

    plt.figure()
    plt.plot(bt_plot["date"], bt_plot["cum_pnl"], linewidth=2)
    plt.title("Cumulative Strategy PnL (Active Window)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.step(bt_plot["date"], bt_plot["position"], where="post")
    plt.title("Strategy Position (Active Window)")
    plt.xlabel("Date")
    plt.ylabel("Position")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(bt_plot["date"], bt_plot["spread"], linewidth=1.5)

    if events_df is not None:
        valid_events = events_df.dropna(subset=["trade_date"]).copy()
        valid_events = valid_events[
            (valid_events["trade_date"] >= bt_plot["date"].min()) &
            (valid_events["trade_date"] <= bt_plot["date"].max())
        ]

        spread_lookup = bt_plot.set_index("date")["spread"]

        long_events = valid_events[valid_events["signal"] == 1]
        short_events = valid_events[valid_events["signal"] == -1]

        if not long_events.empty:
            yvals = spread_lookup.reindex(long_events["trade_date"]).values
            plt.scatter(long_events["trade_date"], yvals, marker="^", s=80, label="Long Signal")

        if not short_events.empty:
            yvals = spread_lookup.reindex(short_events["trade_date"]).values
            plt.scatter(short_events["trade_date"], yvals, marker="v", s=80, label="Short Signal")

        if not valid_events.empty:
            plt.legend()

    plt.title("Spread Series with Event Signals")
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if event_panel_df is not None and not event_panel_df.empty:
        counts = event_panel_df["event_date"].nunique()
        avg_path = event_panel_df.groupby("t")["spread_change_from_event"].mean()

        plt.figure()
        plt.plot(avg_path.index, avg_path.values, linewidth=2)
        plt.axvline(0, linestyle="--")
        plt.title(f"Average Event-Study Spread Response ({counts} events)")
        plt.xlabel("Trading Days Relative to Event")
        plt.ylabel("Spread Change From Event")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# =========================
# 15. Save outputs
# =========================
def save_outputs(eia_df, events_df, spread_df, bt_df, event_panel_df, trade_table, output_dir="output"):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    eia_df.to_csv(output_path / "clean_eia_inventory.csv", index=False)
    events_df.to_csv(output_path / "event_signals.csv", index=False)
    spread_df.to_csv(output_path / "clean_spread_data.csv", index=False)
    bt_df.to_csv(output_path / "strategy_backtest.csv", index=False)
    trade_table.to_csv(output_path / "trade_table.csv", index=False)

    if event_panel_df is not None and not event_panel_df.empty:
        event_panel_df.to_csv(output_path / "event_study_panel.csv", index=False)


# =========================
# 16. Main
# =========================
def main(
    api_key,
    spread_file="wti_calendar_spread.csv",
    eia_start_date="2010-01-01",
    lookback_expected=4,
    zscore_window=26,
    z_threshold=1.0,
    hold_days=5,
    transaction_cost_bps=5.0
):
    print("Loading EIA data from API...")
    eia = load_eia_api(api_key, start_date=eia_start_date)
    print(eia.head())
    print(eia.tail())
    print(f"EIA observations: {len(eia)}")
    print()

    print("Loading spread data...")
    spread = load_spread_file(spread_file)
    print(spread.head())
    print(spread.tail())
    print(f"Spread observations: {len(spread)}")
    print()

    print("Building surprise series...")
    eia_surprise = build_inventory_surprise(
        eia,
        lookback_expected=lookback_expected,
        zscore_window=zscore_window
    )

    print("Aligning events to market...")
    events_aligned = align_events_to_market(eia_surprise, spread)

    print("Generating signals...")
    events = generate_signals(events_aligned, z_threshold=z_threshold)

    print(events[["release_date", "actual", "expected", "surprise", "surprise_z", "trade_date", "signal"]].tail(10))
    print()
    print("Signal counts:")
    print(events["signal"].value_counts(dropna=False))
    print()

    print("Building positions...")
    positioned = build_positions(spread, events, hold_days=hold_days)

    print("Running backtest...")
    bt = run_backtest(positioned, transaction_cost_bps=transaction_cost_bps)

    print("Computing metrics...")
    metrics, trade_table = compute_metrics(bt)

    print("Building event-study panel...")
    event_panel = build_event_study_panel(spread, events, pre_days=3, post_days=5)

    print("\nMetrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    if not trade_table.empty:
        print("\nTrade table preview:")
        print(trade_table.head())

    print("\nPlotting...")
    plot_results(bt, events_df=events, event_panel_df=event_panel)

    print("\nSaving outputs...")
    save_outputs(eia_surprise, events, spread, bt, event_panel, trade_table, output_dir="output")
    print("Saved files to ./output")

    return eia_surprise, events, spread, bt, event_panel, metrics, trade_table


# =========================
# 17. Run
# =========================
eia_surprise, events, spread, bt, event_panel, metrics, trade_table = main(
    api_key=EIA_API_KEY,
    spread_file="wti_calendar_spread.csv",
    eia_start_date="2010-01-01",
    lookback_expected=4,
    zscore_window=26,
    z_threshold=1.0,
    hold_days=5,
    transaction_cost_bps=5.0
)

# %%
import numpy as np
import pandas as pd

perf = bt.copy()

# use net_pnl as the strategy return series
perf['returns'] = perf['net_pnl']

# ===== PERFORMANCE METRICS =====
total_return = perf['returns'].sum()
annual_return = perf['returns'].mean() * 252
annual_vol = perf['returns'].std() * np.sqrt(252)

if perf['returns'].std() != 0:
    sharpe = (perf['returns'].mean() / perf['returns'].std()) * np.sqrt(252)
else:
    sharpe = 0.0

# MAX DRAWDOWN
cum_pnl = perf['returns'].cumsum()
rolling_max = cum_pnl.cummax()
drawdown = cum_pnl - rolling_max
max_dd = drawdown.min()

# DAILY WIN RATE
daily_win_rate = (perf['returns'] > 0).mean()

# NONZERO DAYS
nonzero_days = (perf['position'] != 0).sum()

# TRADE-LEVEL STATS
trade_df = trade_table.copy() if 'trade_table' in globals() else pd.DataFrame()

if not trade_df.empty:
    num_trades = len(trade_df)
    trade_win_rate = (trade_df['trade_pnl'] > 0).mean()
    avg_trade_pnl = trade_df['trade_pnl'].mean()
    median_trade_pnl = trade_df['trade_pnl'].median()
    best_trade = trade_df['trade_pnl'].max()
    worst_trade = trade_df['trade_pnl'].min()
    avg_holding_days = trade_df['holding_days'].mean()
else:
    num_trades = 0
    trade_win_rate = np.nan
    avg_trade_pnl = np.nan
    median_trade_pnl = np.nan
    best_trade = np.nan
    worst_trade = np.nan
    avg_holding_days = np.nan

print("===== STRATEGY METRICS =====")
print(f"Total Return: {total_return:.4f}")
print(f"Annualized Return: {annual_return:.4f}")
print(f"Annualized Volatility: {annual_vol:.4f}")
print(f"Sharpe Ratio: {sharpe:.4f}")
print(f"Max Drawdown: {max_dd:.4f}")
print(f"Daily Win Rate: {daily_win_rate:.2%}")
print(f"Non-zero Position Days: {nonzero_days}")

print("\n===== TRADE-LEVEL METRICS =====")
print(f"Number of Trades: {num_trades}")
print(f"Trade Win Rate: {trade_win_rate:.2%}" if pd.notna(trade_win_rate) else "Trade Win Rate: N/A")
print(f"Average Trade PnL: {avg_trade_pnl:.4f}" if pd.notna(avg_trade_pnl) else "Average Trade PnL: N/A")
print(f"Median Trade PnL: {median_trade_pnl:.4f}" if pd.notna(median_trade_pnl) else "Median Trade PnL: N/A")
print(f"Best Trade: {best_trade:.4f}" if pd.notna(best_trade) else "Best Trade: N/A")
print(f"Worst Trade: {worst_trade:.4f}" if pd.notna(worst_trade) else "Worst Trade: N/A")
print(f"Average Holding Days: {avg_holding_days:.2f}" if pd.notna(avg_holding_days) else "Average Holding Days: N/A")

# %%
if not event_panel.empty:
    event_avg = event_panel.groupby("t")["spread_change_from_event"].mean()

    t1 = event_avg.get(1, np.nan)
    t3 = event_avg.get(3, np.nan)
    t5 = event_avg.get(5, np.nan)

    # positive response rates at each horizon
    event_t1 = event_panel[event_panel["t"] == 1]["spread_change_from_event"]
    event_t3 = event_panel[event_panel["t"] == 3]["spread_change_from_event"]
    event_t5 = event_panel[event_panel["t"] == 5]["spread_change_from_event"]

    pos_t1 = (event_t1 > 0).mean() if len(event_t1) > 0 else np.nan
    pos_t3 = (event_t3 > 0).mean() if len(event_t3) > 0 else np.nan
    pos_t5 = (event_t5 > 0).mean() if len(event_t5) > 0 else np.nan

    peak_day = event_avg.abs().idxmax()
    peak_value = event_avg.loc[peak_day]

    print("\n===== EVENT STUDY METRICS =====")
    print(f"Average Spread Change t+1: {t1:.4f}" if pd.notna(t1) else "Average Spread Change t+1: N/A")
    print(f"Average Spread Change t+3: {t3:.4f}" if pd.notna(t3) else "Average Spread Change t+3: N/A")
    print(f"Average Spread Change t+5: {t5:.4f}" if pd.notna(t5) else "Average Spread Change t+5: N/A")
    print(f"Positive Response Rate t+1: {pos_t1:.2%}" if pd.notna(pos_t1) else "Positive Response Rate t+1: N/A")
    print(f"Positive Response Rate t+3: {pos_t3:.2%}" if pd.notna(pos_t3) else "Positive Response Rate t+3: N/A")
    print(f"Positive Response Rate t+5: {pos_t5:.2%}" if pd.notna(pos_t5) else "Positive Response Rate t+5: N/A")
    print(f"Peak Response Day: t={peak_day}")
    print(f"Peak Response Magnitude: {peak_value:.4f}")
else:
    print("Event panel is empty.")

# %%



