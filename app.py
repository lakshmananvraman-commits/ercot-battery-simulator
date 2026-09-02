"""
ERCOT STORAGE DESK
Battery Arbitrage & Dispatch Simulator
=======================================

WHAT THIS APP DOES, IN ONE PARAGRAPH
-------------------------------------
A grid-scale battery makes money by buying electricity when it is cheap and
selling it back when it is expensive -- the same idea as buying a stock low
and selling high, except the "stock" is 15-minute electricity prices and the
"warehouse" is a battery with a hard energy limit. This app pulls the last
few days of real Electric Reliability Council of Texas (ERCOT) settlement
prices, then replays a simple rule-based trading strategy against that real
price history to see what a battery of a given size, sitting at a given hub,
would have earned.

HOW TO RUN THIS
----------------
1. Install the dependencies:      pip install -r requirements.txt
2. Launch the app:                streamlit run app.py
3. Your browser opens automatically to a local address (usually
   http://localhost:8501). Everything runs on your machine -- the only
   outbound network calls are to ERCOT's public data feed via the
   `gridstatus` library.

DATA SOURCE
------------
Real-time Settlement Point Prices (SPP), 15-minute resolution, pulled live
via the open-source `gridstatus` library (https://github.com/gridstatus/gridstatus),
which wraps ERCOT's public reporting feeds. No API key is required for this
dataset. Real ERCOT battery telemetry (actual charge/discharge MW from live
storage units) is a separate, credentialed feed -- see the "Roadmap" note
near the bottom of the sidebar for why this app simulates a hypothetical
battery instead of replaying a real one.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# gridstatus is the one dependency that actually talks to ERCOT. Importing it
# defensively means a missing-dependency problem shows up as a clear message
# in the app instead of a raw Python traceback the first time this runs.
try:
    from gridstatus import Ercot

    GRIDSTATUS_AVAILABLE = True
except ImportError:
    GRIDSTATUS_AVAILABLE = False


# =============================================================================
# 1. PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="ERCOT Storage Desk",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# 2. DESIGN SYSTEM
# -----------------------------------------------------------------------------
# =============================================================================
COLORS = {
    "page_bg": "#F8FAFC",       # Light slate background
    "surface": "#FFFFFF",       # Pure white metric cards
    "surface_raised": "#F1F5F9",
    "border": "#E2E8F0",        # Soft gray borders
    "ink_primary": "#0F172A",   # Deep slate text (softer than pure black)
    "ink_secondary": "#475569",
    "ink_muted": "#94A3B8",
    "gridline": "#F1F5F9",
    "baseline": "#CBD5E1",
    "charge": "#0EA5E9",        # Vibrant energy blue
    "discharge": "#F59E0B",     # Vibrant amber
    "good": "#10B981",          # Emerald green for profit
    "critical": "#EF4444",      # Rose red for loss
}

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
    }
    .stApp {
        background-color: #F8FAFC;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F1F5F9;
        border-right: 1px solid #E2E8F0;
    }
    /* Metric cards: turn Streamlit's plain metric into a bordered tile */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }
    div[data-testid="stMetricLabel"] {
        color: #475569;
    }
    div[data-testid="stMetricValue"] {
        color: #0F172A;
    }
    /* Headings */
    h1, h2, h3 { color: #0F172A; letter-spacing: -0.01em; }
    .desk-tagline {
        color: #475569;
        font-size: 0.95rem;
        margin-top: -10px;
        margin-bottom: 1.2rem;
    }
    .context-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #0EA5E9;
        border-radius: 6px;
        padding: 14px 18px;
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.55;
        margin-bottom: 1rem;
    }
    .context-card b { color: #0F172A; }
    .footnote {
        color: #94A3B8;
        font-size: 0.78rem;
    }
    hr { border-color: #E2E8F0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# 3. CONSTANTS
# =============================================================================

# ERCOT's actual named trading hubs (the "Trading Hub" location_type in the
# settlement price data). Each hub is a volume-weighted average of several
# real settlement points in that zone, which is why traders quote "the ERCOT
# price" at a hub rather than at any single node.
ERCOT_HUBS = {
    "HB_NORTH": "North Hub (Dallas-Fort Worth) -- the most widely quoted ERCOT reference price",
    "HB_HOUSTON": "Houston Hub -- Gulf Coast industrial & petrochemical load",
    "HB_WEST": "West Hub -- Permian Basin, heaviest wind curtailment & price volatility",
    "HB_SOUTH": "South Hub -- Rio Grande Valley, growing wind & solar penetration",
}

# Curated market context, grounded in published 2026 research (cited in the
# caption below the card) rather than invented. This is intentionally NOT
# pulled live -- it is analyst commentary, shown alongside the live numbers
# so a reader understands what regime the live data sits inside.
MARKET_CONTEXT_MD = """
**Why this market is worth simulating right now.** ERCOT's battery fleet
roughly doubled over the past year, reaching about **15 GW installed by Q1
2026** -- and that growth is precisely why arbitrage has gotten *harder*:
day-ahead price spreads have compressed by roughly **50% year-over-year**
as more storage competes to capture the same daily price swing. Real-Time
Co-Optimization plus Batteries (RTC+B), live since December 2025, now
co-optimizes energy and ancillary service awards in the same dispatch pass.
On the demand side, ERCOT's own interconnection queue points to as much as
**35 GW of new data-center load by 2035** -- nearly half of today's system
peak -- which is expected to widen the evening ramp batteries are built to
monetize. Two-hour+ systems have out-earned one-hour systems by 15-80%
across the last two years, which is why duration is exposed as a first-class
control below, not a footnote.
"""

# Column names gridstatus is documented to return for Ercot.get_spp(). Kept
# as a list (not a single guess) so a minor library-version rename doesn't
# silently break the app -- see _resolve_column() below.
TIME_COLUMN_CANDIDATES = ["Time", "Interval Start", "Interval Start Local", "interval_start_local"]
PRICE_COLUMN_CANDIDATES = ["SPP", "Spp", "LMP", "Price", "price", "spp"]


def format_money(x: float) -> str:
    """$12,340 for everyday dashboard magnitudes; $1.85M once it's big enough
    that a trader would say it out loud in millions rather than thousands."""
    if pd.isna(x):
        return "--"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000:
        return f"{sign}${x/1_000_000:,.2f}M"
    return f"{sign}${x:,.0f}"


# =============================================================================
# 4. DATA LAYER
# -----------------------------------------------------------------------------
# One function talks to the network. It is cached so that dragging a battery
# slider (which never needs new price data, only a new simulation over the
# SAME price data) never re-fetches from ERCOT -- only changing the hub or
# the lookback window does. ttl=600 means a cached pull is reused for up to
# 10 minutes, which comfortably covers ERCOT's own 15-minute publication
# cadence without hammering the feed on every widget interaction.
# =============================================================================

def _resolve_column(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    """Find the real column name gridstatus returned, tolerating minor
    naming drift across library versions instead of hard-crashing on a
    KeyError the moment ERCOT or gridstatus renames something. Matches
    case-insensitively as a last resort."""
    for c in candidates:
        if c in df.columns:
            return c
    lowered = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    raise KeyError(
        f"Could not find a {what} column in the data gridstatus returned. "
        f"Columns present: {list(df.columns)}"
    )


@st.cache_data(ttl=600, show_spinner=False)
def fetch_ercot_prices(hub: str, lookback_days: int) -> pd.DataFrame:
    """
    Pull real-time 15-minute Settlement Point Prices (SPP) for one ERCOT
    trading hub over the last `lookback_days` days.

    SPP vs. LMP, briefly: ERCOT settles most market participants at the
    hub-level Settlement Point Price, which is itself a load-weighted blend
    of the five-minute nodal LMPs underneath that hub, averaged up to a
    15-minute settlement interval. SPP is what a battery sitting at a hub
    (rather than a specific substation) actually gets paid, so it's the
    right series for a hub-level arbitrage study.

    Returns a two-column DataFrame: ['time', 'price'], sorted ascending,
    with duplicate timestamps collapsed (ERCOT occasionally republishes a
    settlement interval; we keep the latest value for each timestamp).
    """
    if not GRIDSTATUS_AVAILABLE:
        raise RuntimeError(
            "The 'gridstatus' package is not installed. Run: "
            "pip install -r requirements.txt"
        )

    iso = Ercot()

    end_ts = pd.Timestamp.now(tz="US/Central")
    start_ts = end_ts - pd.Timedelta(days=lookback_days)

    raw = iso.get_spp(
        date=start_ts,
        end=end_ts,
        market="REAL_TIME_15_MIN",
        location_type="Trading Hub",
        locations=[hub],
        verbose=False,
    )

    if raw is None or len(raw) == 0:
        raise ValueError(
            f"ERCOT returned no Settlement Point Price data for {hub} over "
            f"the last {lookback_days} days. This can happen right after "
            f"ERCOT's publication window rolls over -- try again in a "
            f"few minutes, or shorten the lookback window."
        )

    # Defensive filter: `locations=[hub]` should already restrict the result
    # to this one hub, but if a future gridstatus version ever ignores that
    # and returns every hub, blindly de-duplicating by timestamp next would
    # silently splice together prices from different hubs. Filtering by the
    # Location column first (when present) rules that out.
    for loc_col in ("Location", "location"):
        if loc_col in raw.columns and raw[loc_col].nunique() > 1:
            raw = raw[raw[loc_col] == hub]
            break

    time_col = _resolve_column(raw, TIME_COLUMN_CANDIDATES, "timestamp")
    price_col = _resolve_column(raw, PRICE_COLUMN_CANDIDATES, "price")

    clean = pd.DataFrame({
        "time": pd.to_datetime(raw[time_col]),
        "price": pd.to_numeric(raw[price_col], errors="coerce"),
    })
    clean = clean.dropna(subset=["price"])
    clean = clean.sort_values("time").drop_duplicates(subset="time", keep="last")
    clean = clean.reset_index(drop=True)

    if len(clean) == 0:
        raise ValueError(
            f"ERCOT data for {hub} came back but every price value was "
            f"unparseable -- this points to a gridstatus column-format "
            f"change rather than a network issue."
        )

    return clean


# =============================================================================
# 5. SIMULATION ENGINE -- THE BATTERY MATH
# -----------------------------------------------------------------------------
# A battery is constrained by TWO independent limits at every instant, and a
# correct simulator has to respect both simultaneously:
#
#   1. POWER (MW)     -- how fast energy can move in or out right now.
#                        This is the size of the inverter/pipe.
#   2. ENERGY (MWh)    -- how much can be stored in total.
#                        This is the size of the tank, = Power x Duration.
#                        A "100 MW / 2-hour" battery has a 200 MWh tank.
#
# In any single time step, the battery can move at most (Power x step
# length in hours) MWh -- that's the power limit. But it can never charge
# past a full tank or discharge past an empty one -- that's the energy
# limit. Every real dispatch loop is just: take the SMALLER of what the
# power rating allows and what the remaining tank space (or remaining
# charge) allows. That single "min()" is the crux of the whole model.
#
# ROUND-TRIP EFFICIENCY: no battery gives back 100% of what you put in --
# some is lost to heat in the cells, inverter, and transformer. A modern
# grid-scale lithium-ion system's round-trip efficiency (electricity out /
# electricity in, one full cycle) typically runs 85-92%. This model applies
# that loss on the way OUT: every MWh drawn from the state of charge only
# delivers (efficiency) MWh to the grid meter -- and only the delivered MWh
# earns revenue. Applying it entirely on discharge (rather than splitting it
# across both legs) is a simplification, but it nets to the identical total
# loss per round trip and is far easier to reason about in plain English.
#
# DEGRADATION COST: every MWh cycled through a battery uses up a sliver of
# its finite cycle life, which is a real economic cost even though no
# invoice arrives for it. Treating it as a fixed $/MWh "toll" on every MWh
# charged or discharged (default $0, i.e. off) is the standard simplified
# way analysts fold battery wear into an otherwise price-only arbitrage
# model.
# =============================================================================

def run_battery_simulation(
    df: pd.DataFrame,
    power_mw: float,
    duration_hours: float,
    charge_threshold: float,
    discharge_threshold: float,
    round_trip_efficiency: float = 0.88,
    initial_soc_pct: float = 50.0,
    degradation_cost_per_mwh: float = 0.0,
):
    """
    Replays a threshold dispatch strategy over historical prices.

    Strategy, in plain English, evaluated independently at every price
    interval:
        - price is BELOW the charge threshold  -> buy (charge), up to
          whichever is smaller: the power limit or the remaining tank space.
        - price is ABOVE the discharge threshold -> sell (discharge), up to
          whichever is smaller: the power limit or the energy left in the
          tank.
        - otherwise -> idle. No trade, no cost, no revenue.

    Parameters
    ----------
    df : DataFrame with columns ['time', 'price'], sorted ascending.
    power_mw : inverter / grid connection limit, MW.
    duration_hours : how many hours at full power the battery can sustain;
        energy capacity (MWh) = power_mw * duration_hours.
    charge_threshold, discharge_threshold : $/MWh trigger prices.
    round_trip_efficiency : 0-1, fraction of stored energy recovered at
        discharge.
    initial_soc_pct : starting state of charge, 0-100.
    degradation_cost_per_mwh : $/MWh of throughput charged into cumulative
        profit as a wear cost; 0 disables it.

    Returns
    -------
    (enriched_df, summary_dict) -- enriched_df is `df` plus per-interval
    columns for the action taken, state of charge, and running profit;
    summary_dict is the head-line numbers used by the metric cards.
    """
    capacity_mwh = power_mw * duration_hours
    if capacity_mwh <= 0:
        raise ValueError("Battery capacity (power x duration) must be greater than zero.")
    if len(df) < 2:
        raise ValueError("Need at least two price observations to simulate a dispatch.")

    # The interval length is INFERRED from the data's own timestamps (the
    # median gap between consecutive rows) rather than hardcoded to 15
    # minutes. That keeps the power-limit math correct even if ERCOT's feed
    # cadence changes, or a gap in the data makes some intervals uneven.
    interval_hours = df["time"].diff().dropna().median().total_seconds() / 3600.0

    soc_mwh = capacity_mwh * (initial_soc_pct / 100.0)
    soc_mwh = min(max(soc_mwh, 0.0), capacity_mwh)

    cumulative_profit = 0.0
    n = len(df)
    actions = [None] * n
    soc_series = np.empty(n)
    profit_series = np.empty(n)
    interval_mwh_series = np.zeros(n)  # +charge (grid draw) / -discharge (grid delivery)

    total_charged_mwh = 0.0
    total_discharged_mwh = 0.0   # measured AFTER efficiency loss -- what actually reached the grid
    total_charge_cost = 0.0
    total_discharge_revenue = 0.0

    prices = df["price"].to_numpy()
    EPS = 1e-9  # floating-point slack so "SoC == capacity" doesn't get blocked by rounding dust

    for i in range(n):
        price = prices[i]

        if price < charge_threshold and soc_mwh < capacity_mwh - EPS:
            # --- CHARGE ---
            # Cap by whichever is tighter: the inverter's power rating, or
            # the space still free in the tank.
            room_left_mwh = capacity_mwh - soc_mwh
            power_limited_mwh = power_mw * interval_hours
            charge_mwh = min(power_limited_mwh, room_left_mwh)

            soc_mwh += charge_mwh
            cost = charge_mwh * price + degradation_cost_per_mwh * charge_mwh
            cumulative_profit -= cost

            total_charged_mwh += charge_mwh
            total_charge_cost += charge_mwh * price  # excludes degradation, so this stays a pure market price

            actions[i] = "CHARGE"
            interval_mwh_series[i] = charge_mwh

        elif price > discharge_threshold and soc_mwh > EPS:
            # --- DISCHARGE ---
            # Cap by whichever is tighter: the inverter's power rating, or
            # the energy actually sitting in the tank.
            power_limited_mwh = power_mw * interval_hours
            discharge_mwh_from_soc = min(power_limited_mwh, soc_mwh)

            # Only a fraction of what leaves the tank reaches the meter --
            # the rest is the round-trip loss.
            energy_delivered_mwh = discharge_mwh_from_soc * round_trip_efficiency

            soc_mwh -= discharge_mwh_from_soc
            revenue = energy_delivered_mwh * price - degradation_cost_per_mwh * discharge_mwh_from_soc
            cumulative_profit += revenue

            total_discharged_mwh += energy_delivered_mwh
            total_discharge_revenue += energy_delivered_mwh * price

            actions[i] = "DISCHARGE"
            interval_mwh_series[i] = -discharge_mwh_from_soc

        else:
            # --- IDLE --- price didn't clear either threshold, or the
            # battery is already full (can't charge more) / already empty
            # (can't discharge more). No trade this interval.
            actions[i] = "IDLE"

        soc_series[i] = soc_mwh
        profit_series[i] = cumulative_profit

    out = df.copy()
    out["action"] = actions
    out["soc_mwh"] = soc_series
    out["soc_pct"] = soc_series / capacity_mwh * 100.0
    out["cumulative_profit"] = profit_series
    out["interval_mwh"] = interval_mwh_series

    # "Cycles" is the standard battery-industry yardstick: total energy
    # discharged divided by the tank size, i.e. how many times over you
    # fully emptied the battery -- the number that drives warranty and
    # degradation planning, independent of how big the battery is.
    total_cycles = total_discharged_mwh / capacity_mwh

    # Realized price is REVENUE-WEIGHTED (total $ / total MWh), not a
    # simple average of the trigger prices -- that's the number a trader
    # actually realizes, and it's pulled toward whichever intervals moved
    # the most energy.
    avg_discharge_price = (total_discharge_revenue / total_discharged_mwh) if total_discharged_mwh > EPS else float("nan")
    avg_charge_price = (total_charge_cost / total_charged_mwh) if total_charged_mwh > EPS else float("nan")
    realized_spread = (
        avg_discharge_price - avg_charge_price
        if total_discharged_mwh > EPS and total_charged_mwh > EPS
        else float("nan")
    )

    summary = {
        "capacity_mwh": capacity_mwh,
        "interval_hours": interval_hours,
        "total_profit": cumulative_profit,
        "total_cycles": total_cycles,
        "avg_discharge_price": avg_discharge_price,
        "avg_charge_price": avg_charge_price,
        "realized_spread": realized_spread,
        "total_charged_mwh": total_charged_mwh,
        "total_discharged_mwh": total_discharged_mwh,
        "pct_negative_price": float((df["price"] < 0).mean() * 100.0),
    }
    return out, summary


# =============================================================================
# 6. CHART BUILDER
# -----------------------------------------------------------------------------
# Three stacked subplots sharing one x-axis (time) -- deliberately NOT one
# chart with two y-axes. A dual-axis chart lets two independently-scaled
# series (dollars and percent, here) draw whatever visual correlation the
# axis ranges happen to create, which is misleading rather than informative.
# Three panels on one shared time axis (price, state of charge, running P&L)
# tell the same story without inventing a relationship between the scales.
# =============================================================================

def build_dashboard_figure(sim_df: pd.DataFrame, hub_label: str,
                            charge_threshold: float, discharge_threshold: float) -> go.Figure:
    charges = sim_df[sim_df["action"] == "CHARGE"]
    discharges = sim_df[sim_df["action"] == "DISCHARGE"]
    final_profit = sim_df["cumulative_profit"].iloc[-1]
    pnl_color = COLORS["good"] if final_profit >= 0 else COLORS["critical"]
    x_range = [sim_df["time"].min(), sim_df["time"].max()]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            f"ERCOT Settlement Price -- {hub_label}",
            "Battery State of Charge",
            "Cumulative Trading Profit",
        ),
    )

    # --- Row 1: price line -----------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=sim_df["time"], y=sim_df["price"],
            mode="lines", name="ERCOT Price",
            line=dict(color=COLORS["ink_secondary"], width=2),
            hovertemplate="Price: $%{y:.2f}/MWh<extra></extra>",
        ),
        row=1, col=1,
    )
    # Threshold reference lines -- dashed on purpose: these mark the
    # decision boundary the strategy trades against, not a plain gridline.
    # Drawn as ordinary traces (not fig.add_hline) so they pick up a legend
    # entry the same way every other series here does.
    fig.add_trace(
        go.Scatter(
            x=x_range, y=[charge_threshold, charge_threshold],
            mode="lines", name=f"Charge trigger (${charge_threshold:.0f})",
            line=dict(color=COLORS["charge"], width=1, dash="dash"),
            hoverinfo="skip",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_range, y=[discharge_threshold, discharge_threshold],
            mode="lines", name=f"Discharge trigger (${discharge_threshold:.0f})",
            line=dict(color=COLORS["discharge"], width=1, dash="dash"),
            hoverinfo="skip",
        ),
        row=1, col=1,
    )
    # Charge / discharge markers -- the only "loud" marks on this panel;
    # everything else recedes so these pop.
    fig.add_trace(
        go.Scatter(
            x=charges["time"], y=charges["price"],
            mode="markers", name="Charge",
            marker=dict(color=COLORS["charge"], size=9, symbol="circle",
                        line=dict(color=COLORS["surface"], width=2)),
            customdata=charges["interval_mwh"],
            hovertemplate="Charged %{customdata:.1f} MWh @ $%{y:.2f}/MWh<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=discharges["time"], y=discharges["price"],
            mode="markers", name="Discharge",
            marker=dict(color=COLORS["discharge"], size=9, symbol="circle",
                        line=dict(color=COLORS["surface"], width=2)),
            customdata=-discharges["interval_mwh"],
            hovertemplate="Discharged %{customdata:.1f} MWh @ $%{y:.2f}/MWh<extra></extra>",
        ),
        row=1, col=1,
    )

    # --- Row 2: state of charge --------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=sim_df["time"], y=sim_df["soc_pct"],
            mode="lines", name="State of Charge",
            line=dict(color=COLORS["charge"], width=2),
            fill="tozeroy", fillcolor="rgba(14, 165, 233, 0.10)",
            showlegend=False,
            hovertemplate="SoC: %{y:.0f}%<extra></extra>",
        ),
        row=2, col=1,
    )

    # --- Row 3: cumulative profit -------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=sim_df["time"], y=sim_df["cumulative_profit"],
            mode="lines", name="Cumulative Profit",
            line=dict(color=pnl_color, width=2),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.10)" if final_profit >= 0 else "rgba(239, 68, 68, 0.10)",
            showlegend=False,
            hovertemplate="Cumulative P&L: $%{y:,.0f}<extra></extra>",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_range, y=[0, 0],
            mode="lines", name="Break-even",
            line=dict(color=COLORS["baseline"], width=1),
            showlegend=False, hoverinfo="skip",
        ),
        row=3, col=1,
    )

    # --- Shared styling -------------------------------------------------
    fig.update_layout(
        height=780,
        paper_bgcolor=COLORS["page_bg"],
        plot_bgcolor=COLORS["surface"],
        font=dict(family="Inter, Helvetica Neue, sans-serif", color=COLORS["ink_secondary"]),
        hovermode="x unified",
        margin=dict(l=60, r=30, t=60, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["ink_secondary"]),
        ),
    )
    fig.update_xaxes(
        gridcolor=COLORS["gridline"], linecolor=COLORS["baseline"],
        showspikes=True, spikemode="across", spikecolor=COLORS["ink_muted"], spikethickness=1,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"], linecolor=COLORS["baseline"], zeroline=False)
    fig.update_yaxes(title_text="$/MWh", row=1, col=1)
    fig.update_yaxes(title_text="% full", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="$ cumulative", row=3, col=1)
    fig.update_xaxes(title_text="Time (Central)", row=3, col=1)

    # Subplot titles were declared through make_subplots, but they render as
    # generic annotations -- restyle them to match the rest of the ink system.
    for annotation in fig["layout"]["annotations"]:
        if annotation["text"] in (
            f"ERCOT Settlement Price -- {hub_label}",
            "Battery State of Charge",
            "Cumulative Trading Profit",
        ):
            annotation["font"] = dict(size=14, color=COLORS["ink_primary"])
            annotation["x"] = 0
            annotation["xanchor"] = "left"

    return fig


# =============================================================================
# 7. SIDEBAR -- all user controls live here
# =============================================================================
with st.sidebar:
    st.markdown("## Configuration")

    st.markdown("#### Data source")
    hub = st.selectbox(
        "ERCOT trading hub", options=list(ERCOT_HUBS.keys()), index=0,
    )
    st.caption(ERCOT_HUBS[hub])
    lookback_days = st.slider("Lookback window (days)", min_value=2, max_value=14, value=2, step=1)
    if st.button("Refresh live data", use_container_width=True):
        fetch_ercot_prices.clear()
        st.rerun()

    st.divider()
    st.markdown("#### Battery configuration")
    power_mw = st.slider("Power capacity (MW)", min_value=10, max_value=200, value=100, step=10)
    duration_hours = st.slider("Duration (hours)", min_value=0.5, max_value=8.0, value=2.0, step=0.5)
    st.caption(f"= **{power_mw * duration_hours:,.0f} MWh** usable energy capacity (power × duration)")
    initial_soc_pct = st.slider("Starting state of charge (%)", min_value=0, max_value=100, value=50, step=5)

    st.divider()
    st.markdown("#### Trading strategy")
    charge_threshold = st.slider("Charge when price is below ($/MWh)", min_value=-50.0, max_value=60.0, value=15.0, step=1.0)
    discharge_threshold = st.slider("Discharge when price is above ($/MWh)", min_value=40.0, max_value=300.0, value=100.0, step=5.0)
    if charge_threshold >= discharge_threshold:
        st.warning(
            "Charge threshold is at or above the discharge threshold, so the "
            "battery would never trade. Lower the charge trigger or raise "
            "the discharge trigger."
        )

    with st.expander("Advanced: efficiency & degradation"):
        round_trip_efficiency_pct = st.slider("Round-trip efficiency (%)", min_value=75, max_value=98, value=88, step=1)
        st.caption("Modern grid-scale lithium-ion systems typically run 85-92% round-trip.")
        degradation_cost = st.slider("Degradation cost ($/MWh throughput)", min_value=0.0, max_value=15.0, value=0.0, step=0.5)
        st.caption("A wear-and-tear toll on every MWh charged or discharged. Typical LFP estimates run roughly $2-8/MWh; 0 treats the battery as wear-free.")

    st.divider()
    with st.expander("Roadmap: real battery telemetry"):
        st.markdown(
            "ERCOT also publishes real Energy Storage Resource (ESR) "
            "telemetry -- actual charge/discharge MW from live storage "
            "units, down to 4-second resolution -- through a separate, "
            "credentialed endpoint (`rptesr-m`) that requires its own API "
            "subscription beyond the public price feed this app uses. "
            "Wiring that in would let this dashboard line a *simulated* "
            "battery's dispatch up against what real fleets actually did "
            "in the same window."
        )


# =============================================================================
# 8. MAIN LAYOUT
# =============================================================================
st.title("ERCOT Storage Desk")
st.markdown(
    '<p class="desk-tagline">Battery arbitrage &amp; dispatch simulator — '
    "live 15-minute settlement point prices</p>",
    unsafe_allow_html=True,
)

with st.expander("Why this market is worth simulating right now"):
    st.markdown(MARKET_CONTEXT_MD)
    st.markdown(
        '<p class="footnote">Source: Modo Energy ERCOT battery storage research, 2026; '
        "ERCOT long-term load forecasts.</p>",
        unsafe_allow_html=True,
    )

# --- Load price data --------------------------------------------------------
try:
    with st.spinner(f"Pulling {lookback_days}d of real-time SPP for {hub} from ERCOT..."):
        price_df = fetch_ercot_prices(hub, lookback_days)
    st.session_state["data_fetched_at"] = pd.Timestamp.now(tz="US/Central")
except Exception as exc:
    st.error(
        f"Couldn't load ERCOT price data for {hub}: {exc}\n\n"
        "This is almost always transient -- ERCOT's feed can be briefly "
        "unavailable around its publication windows. Try **Refresh live "
        "data** in the sidebar in a minute, or shorten the lookback window."
    )
    st.stop()

if charge_threshold >= discharge_threshold:
    st.info("Fix the threshold warning in the sidebar to run the simulation.")
    st.stop()

# --- Run the dispatch simulation --------------------------------------------
try:
    sim_df, summary = run_battery_simulation(
        price_df,
        power_mw=power_mw,
        duration_hours=duration_hours,
        charge_threshold=charge_threshold,
        discharge_threshold=discharge_threshold,
        round_trip_efficiency=round_trip_efficiency_pct / 100.0,
        initial_soc_pct=initial_soc_pct,
        degradation_cost_per_mwh=degradation_cost,
    )
except Exception as exc:
    st.error(f"The simulation could not run: {exc}")
    st.stop()

actual_days = max((sim_df["time"].max() - sim_df["time"].min()).total_seconds() / 86400.0, 1e-9)
fetched_at = st.session_state.get("data_fetched_at")
if fetched_at is not None:
    st.caption(
        f"Live data as of {fetched_at.strftime('%b %d, %Y %I:%M %p %Z')} · "
        f"{len(price_df):,} price intervals loaded · "
        f"{summary['pct_negative_price']:.1f}% of intervals settled negative"
    )

# --- Metric cards ------------------------------------------------------------
row1 = st.columns(4)
with row1[0]:
    per_mw = summary["total_profit"] / power_mw
    st.metric(
        "Total estimated profit",
        format_money(summary["total_profit"]),
        delta=f"{format_money(per_mw)}/MW over {lookback_days}d",
    )
with row1[1]:
    st.metric(
        "Total cycles",
        f"{summary['total_cycles']:.2f}",
        delta=f"{summary['total_cycles'] / actual_days:.2f}/day",
        delta_color="off",
    )
with row1[2]:
    val = summary["avg_discharge_price"]
    st.metric("Avg. discharge price", f"${val:.0f}/MWh" if not np.isnan(val) else "--")
with row1[3]:
    val = summary["avg_charge_price"]
    st.metric("Avg. charge price", f"${val:.0f}/MWh" if not np.isnan(val) else "--")

row2 = st.columns(3)
with row2[0]:
    val = summary["realized_spread"]
    st.metric(
        "Realized spread",
        f"${val:.0f}/MWh" if not np.isnan(val) else "--",
        delta="discharge − charge, volume-weighted",
        delta_color="off",
    )
with row2[1]:
    st.metric("Round-trip efficiency applied", f"{round_trip_efficiency_pct}%")
with row2[2]:
    st.metric("Data window", f"{lookback_days}d · 15-min SPP · {hub}")

# --- Chart --------------------------------------------------------------
fig = build_dashboard_figure(sim_df, hub, charge_threshold, discharge_threshold)
st.plotly_chart(fig, use_container_width=True, theme=None)

# --- Table view (accessibility twin of the chart) + export ------------------
with st.expander("View underlying data"):
    display_df = sim_df[["time", "price", "action", "soc_pct", "cumulative_profit"]].copy()
    display_df.columns = ["Time", "Price ($/MWh)", "Action", "SoC (%)", "Cumulative Profit ($)"]
    display_df["Price ($/MWh)"] = display_df["Price ($/MWh)"].round(2)
    display_df["SoC (%)"] = display_df["SoC (%)"].round(1)
    display_df["Cumulative Profit ($)"] = display_df["Cumulative Profit ($)"].round(2)
    st.dataframe(display_df, use_container_width=True, height=320)
    st.download_button(
        "Download simulation as CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name=f"ercot_battery_sim_{hub}_{lookback_days}d.csv",
        mime="text/csv",
    )

with st.expander("How this simulation works"):
    st.markdown(
        f"""
Prices are real-time 15-minute Settlement Point Prices for **{hub}**, pulled
live via the `gridstatus` library, which wraps ERCOT's public reporting
feeds -- no API key required for this dataset.

At every 15-minute interval, the battery does exactly one of three things:
charge if the price is below **${charge_threshold:.0f}/MWh**, discharge if
it's above **${discharge_threshold:.0f}/MWh**, or sit idle otherwise. Each
charge or discharge is capped by whichever is smaller: the **{power_mw} MW**
power rating (how fast energy can move) or the remaining room in the
**{power_mw * duration_hours:,.0f} MWh** tank (how much energy is left to
move). A **{round_trip_efficiency_pct}%** round-trip efficiency is applied
on the way out, so every MWh drawn from storage delivers only
{round_trip_efficiency_pct}% of a MWh -- and only that delivered amount is
paid the market price.

"Total cycles" is total MWh discharged divided by tank size -- the standard
way the industry measures how hard a battery worked, independent of its
size. "Avg. discharge/charge price" and "Realized spread" are
revenue-weighted (total $ ÷ total MWh), not simple averages of the trigger
prices, because that's the number a trader actually realizes.

**What this model deliberately leaves out:** ERCOT's live market
(especially under RTC+B, its real-time co-optimization framework) lets a
battery earn from ancillary services alongside energy arbitrage, and real
operators often reserve some capacity for those products rather than
committing 100% of the tank to energy trades. This simulator isolates pure
energy arbitrage on purpose, to make the price-driven dispatch logic easy to
see -- it's a backtest against historical prices with a simple rule, not a
live trading signal or investment advice.
"""
    )

st.markdown(
    '<p class="footnote">Data: ERCOT public Settlement Point Prices via the '
    "gridstatus library. Market commentary: Modo Energy ERCOT battery "
    "storage research, 2026. Educational backtest, not investment "
    "advice.</p>",
    unsafe_allow_html=True,
)
