import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Nifty Spot Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("Data/NiftySpotPrice.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M:%S")
    df["ExpiryDate"] = pd.to_datetime(df["ExpiryDate"], format="%d-%m-%Y", errors="coerce").dt.date
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["date"] = df["timestamp"].dt.date
    return df


@st.cache_data
def build_daily(df, open_time_str="9:15"):
    _open_t = pd.Timestamp("09:16").time() if open_time_str == "9:16" else pd.Timestamp("09:15").time()
    _df = df[df["timestamp"].dt.time >= _open_t] if open_time_str != "9:15" else df
    return (
        _df.groupby("date")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            expiry_date=("ExpiryDate", "first"),
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )


@st.cache_data
def compute_expiry_map(daily):
    """
    Classify each trading date relative to its weekly expiry.
    Expiry = expiry day itself, Expiry-1 = 1 trading day before, etc.
    """
    result = {}
    for _, grp in daily.groupby("expiry_date"):
        dates = sorted(grp["date"].tolist())
        n = len(dates)
        for i, d in enumerate(dates):
            offset = n - 1 - i
            result[d] = "Expiry" if offset == 0 else f"Expiry-{offset}"
    return result


df = load_data()
daily = build_daily(df)
expiry_map = compute_expiry_map(daily)
daily["expiry_type"] = daily["date"].map(expiry_map).fillna("Other")
all_dates = daily["date"].tolist()  # sorted ascending

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 Nifty Spot")
    st.caption("Session: Nifty Spot Prediction")
    st.divider()

    DAY_TYPES = ["All", "Expiry", "Expiry-1", "Expiry-2", "Expiry-3", "Expiry-4"]
    day_type = st.selectbox("1️⃣  Day Type (Expiry Filter)", DAY_TYPES)

    filtered_dates = (
        all_dates
        if day_type == "All"
        else daily[daily["expiry_type"] == day_type]["date"].tolist()
    )

    if not filtered_dates:
        st.error("No dates found for this filter.")
        st.stop()

    selected_date = st.selectbox(
        "2️⃣  Select Date",
        sorted(filtered_dates, reverse=True),
        format_func=lambda d: d.strftime("%d %b %Y  (%a)"),
    )

    st.divider()
    n_segments = st.slider("📌 Line Segments", min_value=2, max_value=20, value=8, step=1,
                           help="Fixed number of segments drawn each day — consistent across all days")

    st.divider()
    open_time_opt = st.selectbox(
        "3️⃣  Opening Candle",
        ["9:15", "9:16"],
        help="Reference candle for Open price, gap calc, and all chart start times",
    )

chart_daily = build_daily(df, open_time_opt)

# ── Derived values ─────────────────────────────────────────────────────────────

idx = all_dates.index(selected_date)
day_row  = daily.iloc[idx]
prev_date = all_dates[idx - 1] if idx > 0 else None
prev_row  = daily.iloc[idx - 1] if idx > 0 else None
day_data = df[df["date"] == selected_date]

_market_open  = pd.Timestamp("09:16").time() if open_time_opt == "9:16" else pd.Timestamp("09:15").time()
_market_close = pd.Timestamp("15:30").time()
_market_data = day_data[
    (day_data["timestamp"].dt.time >= _market_open) &
    (day_data["timestamp"].dt.time <= _market_close)
]
expiry_tag = expiry_map.get(selected_date, "Other")

open_price = float(_market_data["open"].iloc[0]) if len(_market_data) > 0 else float(day_row["open"])

if prev_row is not None:
    gap_pts = round(open_price - float(prev_row["close"]), 2)
    gap_pct = round(gap_pts / float(prev_row["close"]) * 100, 2)
    gap_arrow = "▲" if gap_pts > 0 else ("▼" if gap_pts < 0 else "—")
    gap_label = "Gap Up" if gap_pts > 0 else ("Gap Down" if gap_pts < 0 else "Flat")
else:
    gap_pts = gap_pct = 0
    gap_arrow = "—"
    gap_label = "Flat"

day_pts = round(float(day_row["close"]) - open_price, 2)
day_pct = round(day_pts / open_price * 100, 2)

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown(f"## Nifty Spot  —  {selected_date.strftime('%A, %d %B %Y')}")

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    badge_color = {
        "Expiry": "#7c3aed",
        "Expiry-1": "#2563eb",
        "Expiry-2": "#0891b2",
        "Expiry-3": "#059669",
        "Expiry-4": "#d97706",
    }.get(expiry_tag, "#6b7280")
    st.markdown(
        f"""<div style='background:{badge_color};border-radius:8px;padding:14px 10px;text-align:center'>
            <div style='font-size:11px;color:#ddd;margin-bottom:4px'>Expiry Flag</div>
            <div style='font-size:20px;font-weight:700;color:white'>{expiry_tag}</div>
        </div>""",
        unsafe_allow_html=True,
    )

with c2:
    st.metric(
        f"Open  {gap_arrow} {gap_label}",
        f"{open_price:,.2f}",
        delta=f"{gap_pts:+.2f} pts  ({gap_pct:+.2f}%)" if prev_row is not None else None,
    )

with c3:
    st.metric("High", f"{day_row['high']:,.2f}")

with c4:
    st.metric("Low", f"{day_row['low']:,.2f}")

with c5:
    st.metric(
        "Close",
        f"{day_row['close']:,.2f}",
        delta=f"{day_pts:+.2f} pts  ({day_pct:+.2f}%)",
    )

with c6:
    if prev_row is not None:
        st.metric("Prev Close", f"{prev_row['close']:,.2f}")
    else:
        st.metric("Prev Close", "—")

st.divider()

# ── Historical mini charts ─────────────────────────────────────────────────────

st.subheader("Historical Movement  (prior to selected date)")

past = [d for d in all_dates if d < selected_date]
sel_ts = pd.Timestamp(selected_date)


def get_period_dates(label: str) -> list:
    if label == "1 Day":
        return past[-1:] if past else []
    if label == "1 Week":
        cutoff = (sel_ts - pd.Timedelta(weeks=1)).date()
        return [d for d in past if d >= cutoff]
    if label == "1 Month":
        cutoff = (sel_ts - pd.DateOffset(months=1)).date()
        return [d for d in past if d >= cutoff]
    return []


cols = st.columns(3)

for col, label in zip(cols, ["1 Day", "1 Week", "1 Month"]):
    period_dates = get_period_dates(label)

    with col:
        if not period_dates:
            st.markdown(f"**{label}**")
            st.caption("No data")
            continue

        # 1 Day: resample 1-min data of previous day into 15-min candles
        if label == "1 Day":
            prev_day = period_dates[-1]
            raw = df[df["date"] == prev_day].copy()
            raw = raw.set_index("timestamp")
            raw = raw[raw.index.time >= _market_open]
            pdata = raw.resample("15min").agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
            ).dropna().reset_index()
            p_open  = float(pdata["open"].iloc[0])
            p_close = float(pdata["close"].iloc[-1])
            x_vals  = pdata["timestamp"]
            caption = f"{prev_day.strftime('%d %b %Y')}  (15-min candles)"
        else:
            pdata   = chart_daily[chart_daily["date"].isin(period_dates)].copy()
            p_open  = float(pdata["open"].iloc[0])
            p_close = float(pdata["close"].iloc[-1])
            x_vals  = pdata["date"].astype(str)
            caption = (
                f"{period_dates[0].strftime('%d %b')} → {period_dates[-1].strftime('%d %b %Y')}"
                f"  ({len(period_dates)} days)"
            )

        m_pts = round(p_close - p_open, 2)
        m_pct = round(m_pts / p_open * 100, 2)
        color = "#26a69a" if m_pts >= 0 else "#ef5350"
        arrow = "▲" if m_pts >= 0 else "▼"

        st.markdown(
            f"<span style='font-size:14px;font-weight:700'>{label}</span>"
            f"<span style='color:{color};font-size:13px;font-weight:600'>"
            f" ({arrow} {m_pts:+.2f} pts &nbsp;|&nbsp; {m_pct:+.2f}%)</span>",
            unsafe_allow_html=True,
        )

        mini = go.Figure(
            go.Candlestick(
                x=x_vals,
                open=pdata["open"],
                high=pdata["high"],
                low=pdata["low"],
                close=pdata["close"],
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
                showlegend=False,
            )
        )
        mini.update_layout(
            xaxis_rangeslider_visible=False,
            height=160,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=True, showgrid=False, tickfont=dict(size=9)),
        )
        st.plotly_chart(mini, width="stretch", key=f"mini_{label}")
        st.caption(caption)

st.divider()

# ── Main intraday candlestick ──────────────────────────────────────────────────

st.subheader(f"Intraday 1-min Chart  —  [{expiry_tag}]")

fig = go.Figure(
    go.Candlestick(
        x=_market_data["timestamp"],
        open=_market_data["open"],
        high=_market_data["high"],
        low=_market_data["low"],
        close=_market_data["close"],
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        name="NIFTY Spot",
    )
)

if prev_row is not None:
    fig.add_hline(
        y=float(prev_row["close"]),
        line_dash="dash",
        line_color="#ffa726",
        line_width=1.5,
        annotation_text=f"Prev Close  {prev_row['close']:.2f}",
        annotation_position="top left",
        annotation_font_color="#ffa726",
    )

fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=560,
    template="plotly_dark",
    margin=dict(l=0, r=0, t=36, b=0),
    xaxis=dict(
        title=None,
        automargin=True,
        range=[_market_data["timestamp"].iloc[0], _market_data["timestamp"].iloc[-1] + pd.Timedelta(minutes=1)],
    ),
    yaxis=dict(title=None, side="left", automargin=True),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.06,
        xanchor="right",
        x=1,
        font=dict(size=11),
        bgcolor="rgba(0,0,0,0)",
    ),
)

# ── Price path (DC segments) ───────────────────────────────────────────────────

def _rdp_vertical(closes, epsilon):
    """Ramer-Douglas-Peucker on a 1-D price series using vertical distance."""
    n = len(closes)
    if n < 3:
        return list(range(n))
    keep = [False] * n
    keep[0] = True
    keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue
        y0, y1 = closes[start], closes[end]
        max_dist, max_idx = 0.0, start
        for i in range(start + 1, end):
            t = (i - start) / (end - start)
            dist = abs(closes[i] - (y0 + t * (y1 - y0)))
            if dist > max_dist:
                max_dist, max_idx = dist, i
        if max_dist > epsilon:
            keep[max_idx] = True
            stack.append((start, max_idx))
            stack.append((max_idx, end))
    return [i for i, k in enumerate(keep) if k]


def _epsilon_for_n_segments(closes, target):
    lo, hi = 0.0, float(closes.max() - closes.min()) + 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if len(_rdp_vertical(closes, mid)) - 1 <= target:
            hi = mid
        else:
            lo = mid
    return hi


import numpy as _np


def _reduce_to_n(idx, closes, target):
    idx = list(idx)
    while len(idx) - 1 > target and len(idx) >= 3:
        min_dist, min_pos = float("inf"), None
        for pos in range(1, len(idx) - 1):
            pi, ci, ni = idx[pos - 1], idx[pos], idx[pos + 1]
            t    = (ci - pi) / (ni - pi) if ni != pi else 0.5
            dist = abs(float(closes[ci]) - (float(closes[pi]) + t * (float(closes[ni]) - float(closes[pi]))))
            if dist < min_dist:
                min_dist, min_pos = dist, pos
        if min_pos is None:
            break
        idx.pop(min_pos)
    return idx


def _dc_run(closes, theta):
    pts = [(0, float(closes[0]))]
    ext, ext_i, direction = float(closes[0]), 0, None
    for i in range(1, len(closes)):
        p = float(closes[i])
        if direction is None:
            if (p - ext) / ext >= theta:
                direction = "UP";   ext, ext_i = p, i
            elif (ext - p) / ext >= theta:
                direction = "DOWN"; ext, ext_i = p, i
        elif direction == "UP":
            if p > ext: ext, ext_i = p, i
            elif (ext - p) / ext >= theta:
                pts.append((ext_i, ext)); direction = "DOWN"; ext, ext_i = p, i
        else:
            if p < ext: ext, ext_i = p, i
            elif (p - ext) / ext >= theta:
                pts.append((ext_i, ext)); direction = "UP";   ext, ext_i = p, i
    pts.append((len(closes) - 1, float(closes[-1])))
    return pts


def _dc_idx(closes, target):
    lo, hi = 1e-6, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if len(_dc_run(closes, mid)) - 1 <= target:
            hi = mid
        else:
            lo = mid
    pts = _dc_run(closes, hi)
    idx = [p[0] for p in pts]
    return _reduce_to_n(idx, closes, target)


_closes  = _market_data["close"].values.astype(float)
_closes[0] = open_price  # anchor first point to opening candle's open, not its close
_ts_mkt  = _market_data["timestamp"].values
_eps     = _epsilon_for_n_segments(_closes, n_segments)
_rdp_idx   = _rdp_vertical(_closes, _eps)
_main_idx  = _dc_idx(_closes, n_segments)   # DC segments for main chart
_highs_mkt = _market_data["high"].values.astype(float)
_lows_mkt  = _market_data["low"].values.astype(float)

# ── Day-level summary stats (model-independent) ────────────────────────────────
_day_close_val = float(_market_data["close"].iloc[-1])
_day_net       = _day_close_val - open_price
_day_dir       = "Up" if _day_net >= 0 else "Down"
_day_disp      = abs(_day_net)
_day_mins      = (pd.Timestamp(_ts_mkt[-1]) - pd.Timestamp(_ts_mkt[0])).total_seconds() / 60
_day_slope     = round(_day_net / _day_mins, 3) if _day_mins > 0 else 0.0
if _day_net >= 0:
    _day_mfe = round(float(_highs_mkt.max()) - open_price, 2)
    _day_mae = round(open_price - float(_lows_mkt.min()), 2)
else:
    _day_mfe = round(open_price - float(_lows_mkt.min()), 2)
    _day_mae = round(float(_highs_mkt.max()) - open_price, 2)

def _day_summary_hover(path_len):
    er = round(_day_disp / path_len, 3) if path_len > 0 else 1.0
    return (
        f"<b>Day Open → Close</b><br>"
        f"Dir: {_day_dir}  ({_day_net:+.2f} pts)<br>"
        f"Displacement: {_day_disp:.2f} pts<br>"
        f"Path (seg sum): {path_len:.2f} pts<br>"
        f"ER: {er:.3f}<br>"
        f"Slope: {_day_slope:+.3f} pts/min<br>"
        f"MFE: {_day_mfe:.2f} pts<br>"
        f"MAE: {_day_mae:.2f} pts"
    )

_main_path = float(sum(abs(_closes[_main_idx[k+1]] - _closes[_main_idx[k]])
                        for k in range(len(_main_idx) - 1)))

fig.add_trace(go.Scatter(
    x=_ts_mkt[_main_idx],
    y=_closes[_main_idx],
    mode="lines+markers",
    line=dict(color="#f0c040", width=2),
    marker=dict(size=7, color="#f0c040", symbol="circle"),
    name="Price Path (DC)",
))

# Global high and low markers with price labels
_gh = _market_data.loc[_market_data["high"].idxmax()]
_gl = _market_data.loc[_market_data["low"].idxmin()]

fig.add_trace(go.Scatter(
    x=[_gh["timestamp"]], y=[_gh["high"]],
    mode="markers+text",
    marker=dict(size=11, color="#26a69a", symbol="triangle-up"),
    text=[f"H {_gh['high']:.2f}"],
    textposition="top center",
    textfont=dict(color="#26a69a", size=11),
    name="Day High",
))

fig.add_trace(go.Scatter(
    x=[_gl["timestamp"]], y=[_gl["low"]],
    mode="markers+text",
    marker=dict(size=11, color="#ef5350", symbol="triangle-down"),
    text=[f"L {_gl['low']:.2f}"],
    textposition="bottom center",
    textfont=dict(color="#ef5350", size=11),
    name="Day Low",
))

# ── Efficiency Ratio per segment (computed before chart render) ────────────────

_er_rows = []
for k in range(len(_main_idx) - 1):
    si, ei = _main_idx[k], _main_idx[k + 1]
    seg  = _closes[si : ei + 1]
    net  = float(seg[-1]) - float(seg[0])
    path = float(sum(abs(seg[j + 1] - seg[j]) for j in range(len(seg) - 1)))
    er   = round(abs(net) / path, 3) if path > 0 else 1.0
    _er_rows.append({
        "Seg": k + 1,
        "From": pd.Timestamp(_ts_mkt[si]).strftime("%H:%M"),
        "To":   pd.Timestamp(_ts_mkt[ei]).strftime("%H:%M"),
        "Start": round(float(seg[0]), 2),
        "End":   round(float(seg[-1]), 2),
        "Dir":   "↑" if net > 0 else "↓",
        "Net (pts)":  round(net, 2),
        "Path (pts)": round(path, 2),
        "ER":   er,
    })

# Segment number labels at midpoint of each segment, hover shows ER
if _er_rows:
    _lx, _ly, _lt, _lh = [], [], [], []
    for row in _er_rows:
        si, ei = _main_idx[row["Seg"] - 1], _main_idx[row["Seg"]]
        t_mid = pd.Timestamp(_ts_mkt[si]) + (pd.Timestamp(_ts_mkt[ei]) - pd.Timestamp(_ts_mkt[si])) / 2
        p_mid = (_closes[si] + _closes[ei]) / 2
        _lx.append(t_mid)
        _ly.append(p_mid)
        _lt.append(str(row["Seg"]))
        _lh.append(
            f"<b>Seg {row['Seg']}</b>  {row['From']} → {row['To']}<br>"
            f"Dir: {row['Dir']}  |  Net: {row['Net (pts)']:+.2f} pts<br>"
            f"Path: {row['Path (pts)']:.2f} pts<br>"
            f"<b>ER: {row['ER']:.3f}</b>"
        )
    fig.add_trace(go.Scatter(
        x=_lx, y=_ly,
        mode="markers+text",
        marker=dict(size=22, color="rgba(240,192,64,0.18)", symbol="circle",
                    line=dict(color="rgba(240,192,64,0.6)", width=1)),
        text=_lt,
        textfont=dict(color="#f0c040", size=10, family="monospace"),
        textposition="middle center",
        hovertext=_lh,
        hoverinfo="text",
        showlegend=False,
        name="Segments",
    ))

_oc_hover = _day_summary_hover(_main_path)
_t0_ts = pd.Timestamp(_ts_mkt[0])
_t1_ts = pd.Timestamp(_ts_mkt[-1])
_mid_ts = _t0_ts + (_t1_ts - _t0_ts) / 2
_mid_y  = (open_price + _day_close_val) / 2

fig.add_trace(go.Scatter(
    x=[_t0_ts, _t1_ts],
    y=[open_price, _day_close_val],
    mode="lines",
    line=dict(color="#00e5ff", width=2, dash="dot"),
    name="Open → Close",
    hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=[_t0_ts, _mid_ts, _t1_ts],
    y=[open_price, _mid_y, _day_close_val],
    mode="markers",
    marker=dict(size=10, color="#00e5ff", symbol="diamond",
                line=dict(color="#ffffff", width=1)),
    name="Day Summary",
    hovertext=[_oc_hover, _oc_hover, _oc_hover],
    hoverinfo="text",
))

st.plotly_chart(fig, width="stretch")

# ── 1-Week / 1-Month context charts ───────────────────────────────────────────

def _build_context_chart(window_dates, title, height=360):
    GAP_UP = "#00c853"
    GAP_DN = "#f44336"
    # Synthetic Mon–Fri anchor: each valid trading day gets its own weekday slot
    # so all sessions get identical x-width via rangebreaks.
    _SYN_BASE = pd.Timestamp("2000-01-03")  # Monday

    def _syn_date(slot):
        """Map slot index to a synthetic weekday (Mon–Fri cycling)."""
        w, r = divmod(slot, 5)
        return _SYN_BASE + pd.Timedelta(weeks=w, days=r)

    def _to_syn(raw_ts, syn_d):
        t = pd.Timestamp(raw_ts)
        return syn_d + pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)

    cfig = go.Figure()
    _open_t  = pd.Timestamp("09:16").time() if open_time_opt == "9:16" else pd.Timestamp("09:15").time()
    _close_t = pd.Timestamp("15:30").time()
    prev_close_ctx = None
    prev_last_syn  = None   # synthetic ts of previous session's last candle
    day_slot       = 0
    tick_vals      = []
    tick_text      = []

    for d in window_dates:
        day_mkt = df[df["date"] == d]
        day_mkt = day_mkt[
            (day_mkt["timestamp"].dt.time >= _open_t) &
            (day_mkt["timestamp"].dt.time <= _close_t)
        ]
        if len(day_mkt) < 2:
            # No data for this date — skip entirely, do not advance slot
            prev_close_ctx = None
            prev_last_syn  = None
            continue

        syn_d   = _syn_date(day_slot)
        ts      = day_mkt["timestamp"].values
        cl      = day_mkt["close"].values.astype(float)
        hi      = day_mkt["high"].values.astype(float)
        lo      = day_mkt["low"].values.astype(float)
        d_open  = float(day_mkt["open"].iloc[0])
        d_close = float(cl[-1])
        cl_plot = cl.copy(); cl_plot[0] = d_open

        is_sel  = (d == selected_date)

        # ── Path: reuse main-chart precomputed values for selected day so
        #    Path (DC) and ER are bit-identical to the main intraday chart.
        if is_sel:
            d_idx = _main_idx
            path  = _main_path
        else:
            d_idx = _dc_idx(cl_plot, n_segments)
            path  = float(sum(abs(cl_plot[d_idx[k+1]] - cl_plot[d_idx[k]])
                              for k in range(len(d_idx) - 1)))

        net   = d_close - d_open
        disp  = abs(net)
        mins  = (pd.Timestamp(ts[-1]) - pd.Timestamp(ts[0])).total_seconds() / 60
        er    = round(disp / path, 3) if path > 0 else 1.0
        slope = round(net / mins, 3) if mins > 0 else 0.0
        if net >= 0:
            mfe = round(float(hi.max()) - d_open, 2)
            mae = round(d_open - float(lo.min()), 2)
        else:
            mfe = round(d_open - float(lo.min()), 2)
            mae = round(float(hi.max()) - d_open, 2)

        lw_dc   = 1.8 if is_sel else 1.2
        lw_oc   = 1.5 if is_sel else 0.9
        mk_sz   = 6   if is_sel else 4
        hl_sz   = 10  if is_sel else 7
        day_lbl = d.strftime("%d %b") + (" ★" if is_sel else "")

        hover_txt = (
            f"<b>{d.strftime('%d %b %Y')}{' ★' if is_sel else ''}</b><br>"
            f"Dir: {'Up ▲' if net >= 0 else 'Down ▼'}  ({net:+.2f} pts)<br>"
            f"Open: {d_open:.2f}  →  Close: {d_close:.2f}<br>"
            f"Displacement: {disp:.2f} pts<br>"
            f"Path (DC): {path:.2f} pts<br>"
            f"ER: {er:.3f}<br>"
            f"Slope: {slope:+.3f} pts/min<br>"
            f"MFE: {mfe:.2f} pts<br>"
            f"MAE: {mae:.2f} pts"
        )

        # Map all timestamps to synthetic x-axis
        ts_syn = [_to_syn(t, syn_d) for t in ts]
        t0c    = ts_syn[0]
        t1c    = ts_syn[-1]
        tick_vals.append(syn_d + pd.Timedelta(hours=12))
        tick_text.append(d.strftime("%d %b"))

        # ── After-market gap visualisation ─────────────────────────────────────
        if prev_close_ctx is not None and prev_last_syn is not None:
            gap = d_open - prev_close_ctx
            if abs(gap) > 0.05:
                gc      = GAP_UP if gap > 0 else GAP_DN
                gc_fill = "rgba(0,200,83,0.18)" if gap > 0 else "rgba(244,67,54,0.18)"
                gt = (f"<b>Gap {'Up ▲' if gap > 0 else 'Down ▼'}</b><br>"
                      f"{gap:+.2f} pts  |  prev close {prev_close_ctx:.2f} → open {d_open:.2f}")
                cfig.add_shape(
                    type="rect", xref="x", yref="y",
                    x0=prev_last_syn, x1=t0c,
                    y0=min(prev_close_ctx, d_open),
                    y1=max(prev_close_ctx, d_open),
                    fillcolor=gc_fill, line=dict(width=0), layer="below",
                )
                cfig.add_trace(go.Scatter(
                    x=[t0c, t0c], y=[prev_close_ctx, d_open],
                    mode="lines",
                    line=dict(color=gc, width=2, dash="dash"),
                    showlegend=False, hoverinfo="skip", legendgroup=day_lbl,
                ))
                cfig.add_trace(go.Scatter(
                    x=[t0c], y=[d_open],
                    mode="markers",
                    marker=dict(size=13, color=gc,
                                symbol="triangle-up" if gap > 0 else "triangle-down",
                                line=dict(color="#ffffff", width=1)),
                    hovertext=[gt], hoverinfo="text",
                    showlegend=False, legendgroup=day_lbl,
                ))

        # ── Raw intraday close line (background, green/red) ────────────────────
        line_color = "#26a69a" if net >= 0 else "#ef5350"
        cfig.add_trace(go.Scatter(
            x=ts_syn,
            y=cl_plot,
            mode="lines",
            line=dict(color=line_color, width=1.2 if is_sel else 0.7),
            opacity=0.45,
            showlegend=False, legendgroup=day_lbl, hoverinfo="skip",
        ))

        # ── DC segment line (gold) ──────────────────────────────────────────────
        cfig.add_trace(go.Scatter(
            x=[ts_syn[i] for i in d_idx],
            y=cl_plot[d_idx],
            mode="lines+markers",
            line=dict(color="#f0c040", width=lw_dc),
            marker=dict(size=mk_sz, color="#f0c040", symbol="circle"),
            name=day_lbl, legendgroup=day_lbl,
            hovertext=[hover_txt] * len(d_idx),
            hoverinfo="text",
        ))

        # ── Open→Close dotted line + diamond hover markers ──────────────────────
        mid_tc = t0c + (t1c - t0c) / 2
        mid_yc = (d_open + d_close) / 2
        cfig.add_trace(go.Scatter(
            x=[t0c, t1c], y=[d_open, d_close],
            mode="lines",
            line=dict(color="#00e5ff", width=lw_oc, dash="dot"),
            showlegend=False, hoverinfo="skip", legendgroup=day_lbl,
        ))
        cfig.add_trace(go.Scatter(
            x=[t0c, mid_tc, t1c], y=[d_open, mid_yc, d_close],
            mode="markers",
            marker=dict(size=6 if not is_sel else 9, color="#00e5ff",
                        symbol="diamond", line=dict(color="#ffffff", width=1)),
            hovertext=[hover_txt] * 3, hoverinfo="text",
            showlegend=False, legendgroup=day_lbl,
        ))

        # Thin vertical separator between sessions
        if prev_last_syn is not None:
            cfig.add_shape(
                type="line", xref="x", yref="paper",
                x0=t0c, x1=t0c, y0=0, y1=1,
                line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
            )

        prev_close_ctx = d_close
        prev_last_syn  = t1c
        day_slot += 1

    cfig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis_rangeslider_visible=False,
        height=height,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(
            showgrid=False, title=None,
            tickvals=tick_vals,
            ticktext=tick_text,
            tickfont=dict(size=9),
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(bounds=[15.55, 9.1], pattern="hour"),
            ],
        ),
        yaxis=dict(showgrid=False, automargin=True, title=None),
        hovermode="closest",
        legend=dict(orientation="h", y=1.1, x=0,
                    font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
    )
    return cfig


_ctx_week_dates  = (past[-4:] if len(past) >= 4 else past) + [selected_date]
_ctx_month_dates = (past[-21:] if len(past) >= 21 else past) + [selected_date]

st.subheader("1-Week Context")
st.plotly_chart(_build_context_chart(_ctx_week_dates,  "1 Week  (5 days)"),  width="stretch", key="ctx_week")
st.subheader("1-Month Context")
st.plotly_chart(_build_context_chart(_ctx_month_dates, "1 Month  (~22 days)", height=420), width="stretch", key="ctx_month")

# ── Tabs: Model Comparison + Segment Features ──────────────────────────────────

st.divider()
_tab1, _tab2, _tab3 = st.tabs(["📊 Model Comparison", "📐 Segment Features", "🏆 Model Backtest"])

def _comp_chart(title, path_x, path_y, color, day_hover=""):
    cf = go.Figure(go.Candlestick(
        x=_market_data["timestamp"],
        open=_market_data["open"], high=_market_data["high"],
        low=_market_data["low"],  close=_market_data["close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=False,
    ))
    if prev_row is not None:
        cf.add_hline(y=float(prev_row["close"]), line_dash="dash",
                     line_color="#ffa726", line_width=1)
    cf.add_trace(go.Scatter(
        x=path_x, y=path_y,
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=5, color=color),
        showlegend=False,
    ))
    _ct0 = pd.Timestamp(_ts_mkt[0])
    _ct1 = pd.Timestamp(_ts_mkt[-1])
    _cmid_ts = _ct0 + (_ct1 - _ct0) / 2
    _cmid_y  = (open_price + _day_close_val) / 2
    cf.add_trace(go.Scatter(
        x=[_ct0, _ct1],
        y=[open_price, _day_close_val],
        mode="lines",
        line=dict(color="#00e5ff", width=1.5, dash="dot"),
        hoverinfo="skip",
        showlegend=False,
    ))
    cf.add_trace(go.Scatter(
        x=[_ct0, _cmid_ts, _ct1],
        y=[open_price, _cmid_y, _day_close_val],
        mode="markers",
        marker=dict(size=7, color="#00e5ff", symbol="diamond",
                    line=dict(color="#ffffff", width=1)),
        hovertext=[day_hover, day_hover, day_hover],
        hoverinfo="text",
        showlegend=False,
    ))
    cf.update_layout(
        title=dict(text=title, font=dict(size=12)),
        xaxis_rangeslider_visible=False,
        height=300, template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showgrid=False, automargin=True),
    )
    return cf


_methods = []

# ── 0. RDP (current model) ──
_methods.append(("0. RDP  (current model)", True,
                 _ts_mkt[_rdp_idx], _closes[_rdp_idx], "#f0c040", list(_rdp_idx)))

# ── 1. Ruptures — Binseg linear ──
try:
    import ruptures as rpt
    _sig  = _closes.reshape(-1, 1)
    _bkps = rpt.Binseg(model="linear").fit(_sig).predict(n_bkps=min(n_segments - 1, len(_closes) - 2))
    _m1i  = sorted(set([0] + [b - 1 for b in _bkps]))
    _methods.append(("1. Ruptures  (Binseg · linear)", True,
                     _ts_mkt[_m1i], _closes[_m1i], "#64b5f6", list(_m1i)))
except ImportError:
    _methods.append(("1. Ruptures  (Binseg · linear)", False,
                     None, None, "pip install ruptures", None))

# ── 2. ZigZag + ATR ──
try:
    from scipy.signal import find_peaks as _fp

    def _zigzag_idx(closes, target):
        lo, hi = 0.0, float(closes.max() - closes.min()) + 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            p, _ = _fp(closes,  prominence=mid)
            t, _ = _fp(-closes, prominence=mid)
            if len(p) + len(t) <= target - 1:
                hi = mid
            else:
                lo = mid
        p, _ = _fp(closes,  prominence=hi)
        t, _ = _fp(-closes, prominence=hi)
        idx = sorted(set([0] + list(p) + list(t) + [len(closes) - 1]))
        return _reduce_to_n(idx, closes, target)

    _m2i = _zigzag_idx(_closes, n_segments)
    _methods.append(("2. ZigZag + ATR", True,
                     _ts_mkt[_m2i], _closes[_m2i], "#ba68c8", list(_m2i)))
except Exception as e:
    _methods.append(("2. ZigZag + ATR", False, None, None, str(e), None))

# ── 3. Directional Change (DC) ──
_m3i = _dc_idx(_closes, n_segments)
_methods.append(("3. Directional Change (DC)", True,
                 _ts_mkt[_m3i], _closes[_m3i], "#ff8a65", list(_m3i)))

# ── 4. L1 Trend Filter ──
try:
    import cvxpy as cp

    def _l1_idx(closes, target):
        n   = len(closes)
        xv  = cp.Variable(n)
        lo, hi = 0.0, float(closes.max() - closes.min()) * 200
        for _ in range(20):
            lam  = (lo + hi) / 2
            prob = cp.Problem(cp.Minimize(cp.sum_squares(closes - xv) + lam * cp.norm1(cp.diff(xv, 2))))
            prob.solve(solver=cp.CLARABEL, verbose=False)
            if xv.value is None:
                lo = lam; continue
            sc = _np.where(_np.diff(_np.sign(_np.diff(xv.value))) != 0)[0] + 1
            if len(sc) <= target - 1:
                hi = lam
            else:
                lo = lam
        prob = cp.Problem(cp.Minimize(cp.sum_squares(closes - xv) + hi * cp.norm1(cp.diff(xv, 2))))
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if xv.value is not None:
            sc  = _np.where(_np.diff(_np.sign(_np.diff(xv.value))) != 0)[0] + 1
            idx = sorted(set([0] + list(sc) + [n - 1]))
        else:
            idx = [0, n - 1]
        return _reduce_to_n(idx, closes, target)

    _m4i = _l1_idx(_closes, n_segments)
    _methods.append(("4. L1 Trend Filter", True,
                     _ts_mkt[_m4i], _closes[_m4i], "#a5d6a7", list(_m4i)))
except ImportError:
    _methods.append(("4. L1 Trend Filter", False, None, None, "pip install cvxpy", None))

# ── 5. HMM → reduce to n_segments ──
try:
    from hmmlearn import hmm as _hmm

    _ret  = _np.diff(_closes) / (_closes[:-1] + 1e-8)
    _feat = _np.column_stack([_ret, _np.abs(_ret)])
    _hm   = _hmm.GaussianHMM(n_components=3, covariance_type="diag",
                              n_iter=100, random_state=42, min_covar=1e-3)
    _hm.fit(_feat)
    _sts  = _hm.predict(_feat)
    _m5i  = [0] + [i for i in range(1, len(_sts)) if _sts[i] != _sts[i - 1]] + [len(_closes) - 1]
    _m5i  = _reduce_to_n(sorted(set(_m5i)), _closes, n_segments)
    _methods.append(("5. HMM  (3-state → reduced)", True,
                     _ts_mkt[_m5i], _closes[_m5i], "#fff176", list(_m5i)))
except ImportError:
    _methods.append(("5. HMM  (3-state → reduced)", False,
                     None, None, "pip install hmmlearn", None))

@st.cache_data(show_spinner=False)
def _run_backtest(_df, dates, n_seg, open_time_str="9:15"):
    """Loop every date × 6 models, return per-model aggregated metrics."""
    from scipy.stats import linregress as _lr_bt
    from scipy.signal import find_peaks as _fp_bt

    _mkt_open  = pd.Timestamp("09:16").time() if open_time_str == "9:16" else pd.Timestamp("09:15").time()
    _mkt_close = pd.Timestamp("15:30").time()

    _model_keys = ["0. RDP", "1. Ruptures", "2. ZigZag", "3. DC", "4. L1", "5. HMM"]
    _seg_lists   = {k: [] for k in _model_keys}
    _count_lists = {k: [] for k in _model_keys}

    for _d in dates:
        _md = _df[_df["date"] == _d]
        _md = _md[(_md["timestamp"].dt.time >= _mkt_open) &
                  (_md["timestamp"].dt.time <= _mkt_close)]
        if len(_md) < n_seg + 2:
            continue
        _cl = _md["close"].values.astype(float)
        _cl[0] = float(_md["open"].iloc[0])  # anchor to opening candle's open
        _hi = _md["high"].values.astype(float)
        _lo = _md["low"].values.astype(float)
        _ts = _md["timestamp"].values

        _idx_map = {}

        try:
            _e = _epsilon_for_n_segments(_cl, n_seg)
            _idx_map["0. RDP"] = list(_rdp_vertical(_cl, _e))
        except Exception:
            pass

        try:
            import ruptures as _rpt_bt
            _bkps = _rpt_bt.Binseg(model="linear").fit(_cl.reshape(-1,1)).predict(
                        n_bkps=min(n_seg - 1, len(_cl) - 2))
            _idx_map["1. Ruptures"] = _reduce_to_n(
                sorted(set([0] + [b - 1 for b in _bkps])), _cl, n_seg)
        except Exception:
            pass

        try:
            lo_z, hi_z = 0.0, float(_cl.max() - _cl.min()) + 1.0
            for _ in range(60):
                mid_z = (lo_z + hi_z) / 2
                p_z, _ = _fp_bt(_cl,  prominence=mid_z)
                t_z, _ = _fp_bt(-_cl, prominence=mid_z)
                if len(p_z) + len(t_z) <= n_seg - 1:
                    hi_z = mid_z
                else:
                    lo_z = mid_z
            p_z, _ = _fp_bt(_cl,  prominence=hi_z)
            t_z, _ = _fp_bt(-_cl, prominence=hi_z)
            _idx_map["2. ZigZag"] = _reduce_to_n(
                sorted(set([0] + list(p_z) + list(t_z) + [len(_cl)-1])), _cl, n_seg)
        except Exception:
            pass

        try:
            _idx_map["3. DC"] = _dc_idx(_cl, n_seg)
        except Exception:
            pass

        try:
            import cvxpy as _cp_bt
            _n_bt = len(_cl)
            _xv_bt = _cp_bt.Variable(_n_bt)
            _lo_l, _hi_l = 0.0, float(_cl.max() - _cl.min()) * 200
            for _ in range(20):
                _lam = (_lo_l + _hi_l) / 2
                _pb  = _cp_bt.Problem(_cp_bt.Minimize(
                    _cp_bt.sum_squares(_cl - _xv_bt) + _lam * _cp_bt.norm1(_cp_bt.diff(_xv_bt, 2))))
                _pb.solve(solver=_cp_bt.CLARABEL, verbose=False)
                if _xv_bt.value is None:
                    _lo_l = _lam; continue
                _sc_l = _np.where(_np.diff(_np.sign(_np.diff(_xv_bt.value))) != 0)[0] + 1
                if len(_sc_l) <= n_seg - 1:
                    _hi_l = _lam
                else:
                    _lo_l = _lam
            _pb2 = _cp_bt.Problem(_cp_bt.Minimize(
                _cp_bt.sum_squares(_cl - _xv_bt) + _hi_l * _cp_bt.norm1(_cp_bt.diff(_xv_bt, 2))))
            _pb2.solve(solver=_cp_bt.CLARABEL, verbose=False)
            if _xv_bt.value is not None:
                _sc_l = _np.where(_np.diff(_np.sign(_np.diff(_xv_bt.value))) != 0)[0] + 1
                _idx_map["4. L1"] = _reduce_to_n(
                    sorted(set([0] + list(_sc_l) + [_n_bt-1])), _cl, n_seg)
        except Exception:
            pass

        try:
            from hmmlearn import hmm as _hmm_bt
            _ret_bt  = _np.diff(_cl) / (_cl[:-1] + 1e-8)
            _feat_bt = _np.column_stack([_ret_bt, _np.abs(_ret_bt)])
            _hm_bt   = _hmm_bt.GaussianHMM(n_components=3, covariance_type="diag",
                                            n_iter=100, random_state=42, min_covar=1e-3)
            _hm_bt.fit(_feat_bt)
            _sts_bt  = _hm_bt.predict(_feat_bt)
            _m5_bt   = [0] + [i for i in range(1, len(_sts_bt)) if _sts_bt[i] != _sts_bt[i-1]] + [len(_cl)-1]
            _idx_map["5. HMM"] = _reduce_to_n(sorted(set(_m5_bt)), _cl, n_seg)
        except Exception:
            pass

        for _mname, _midx in _idx_map.items():
            _count_lists[_mname].append(len(_midx) - 1)
            for _k in range(len(_midx) - 1):
                _si, _ei = _midx[_k], _midx[_k + 1]
                _sc  = _cl[_si:_ei + 1]
                _sh  = _hi[_si + 1:_ei + 1]
                _sl  = _lo[_si + 1:_ei + 1]
                _sts = _ts[_si:_ei + 1]

                _t0  = pd.Timestamp(_sts[0])
                _t1  = pd.Timestamp(_sts[-1])
                _dur = (_t1 - _t0).total_seconds() / 60
                _net = float(_sc[-1]) - float(_sc[0])
                _pth = float(sum(abs(_sc[j+1] - _sc[j]) for j in range(len(_sc)-1)))
                _dsp = abs(_net)
                _er  = abs(_net) / _pth if _pth > 0 else 1.0
                _rvol = float(_np.std(_np.diff(_sc) / (_sc[:-1] + 1e-8)) * 100) if len(_sc) > 1 else 0.0

                _op = float(_sc[0])
                if len(_sh) == 0:
                    _mfe_bt = _mae_bt = 0.0
                elif _net >= 0:
                    _mfe_bt = float(_np.max(_sh)) - _op
                    _mae_bt = _op - float(_np.min(_sl))
                else:
                    _mfe_bt = _op - float(_np.min(_sl))
                    _mae_bt = float(_np.max(_sh)) - _op

                if len(_sc) >= 3:
                    _slp_bt, _, _, _, _se_bt = _lr_bt(_np.arange(len(_sc)), _sc)
                    _tstat_bt = abs(float(_slp_bt / _se_bt)) if _se_bt > 0 else 0.0
                else:
                    _tstat_bt = 0.0

                _capture = _dsp / _mfe_bt if _mfe_bt > 0 else 1.0
                _pain    = _mae_bt / _dsp  if _dsp  > 0 else 0.0

                _seg_lists[_mname].append({
                    "er": _er, "slope_t": _tstat_bt,
                    "capture": min(_capture, 1.0), "pain": _pain, "rvol": _rvol,
                })

    _agg = {}
    for _mname in _model_keys:
        _segs = _seg_lists[_mname]
        if not _segs:
            continue
        _counts = _count_lists[_mname]
        _agg[_mname] = {
            "mean_er":      round(float(_np.mean([s["er"]      for s in _segs])), 3),
            "mean_slope_t": round(float(_np.mean([s["slope_t"] for s in _segs])), 2),
            "mean_capture": round(float(_np.mean([s["capture"] for s in _segs])), 3),
            "mean_pain":    round(float(_np.mean([s["pain"]    for s in _segs])), 3),
            "mean_rvol":    round(float(_np.mean([s["rvol"]    for s in _segs])), 4),
            "seg_std":      round(float(_np.std(_counts)), 2),
            "n_days":       len(_counts),
        }
    return _agg


with _tab1:
    st.subheader("Method Comparison  (same day, same slider)")
    for row in range(3):
        _c1, _c2 = st.columns(2)
        for col_i, _col in enumerate([_c1, _c2]):
            _idx = row * 2 + col_i
            if _idx >= len(_methods):
                break
            _title, _ok, _mx, _my, _color, _ridx = _methods[_idx]
            with _col:
                if _ok:
                    _ridx_path = float(sum(
                        abs(_closes[_ridx[k+1]] - _closes[_ridx[k]])
                        for k in range(len(_ridx) - 1)
                    )) if _ridx else 0.0
                    st.plotly_chart(
                        _comp_chart(_title, _mx, _my, _color, _day_summary_hover(_ridx_path)),
                        width="stretch", key=f"cmp_{_idx}"
                    )
                else:
                    st.markdown(f"**{_title}**")
                    st.warning(f"Not available — {_color}")

with _tab2:
    from scipy.stats import linregress as _linreg

    _model_names = [m[0] for m in _methods]
    _sel_model   = st.selectbox("Select Model", _model_names, key="tab2_model")
    _sel_m       = _methods[_model_names.index(_sel_model)]
    _t2, _ok2, _mx2, _my2, _color2, _ridx2 = _sel_m

    if not _ok2 or _ridx2 is None or len(_ridx2) < 2:
        st.warning("Model not available or has fewer than 2 points.")
    else:
        # ── Chart with segment labels ──
        _fig2 = go.Figure(go.Candlestick(
            x=_market_data["timestamp"],
            open=_market_data["open"], high=_market_data["high"],
            low=_market_data["low"],  close=_market_data["close"],
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            name="NIFTY Spot",
        ))
        if prev_row is not None:
            _fig2.add_hline(y=float(prev_row["close"]), line_dash="dash",
                            line_color="#ffa726", line_width=1)
        _fig2.add_trace(go.Scatter(
            x=_ts_mkt[_ridx2], y=_closes[_ridx2],
            mode="lines+markers",
            line=dict(color=_color2, width=2),
            marker=dict(size=7, color=_color2),
            name="Path",
        ))

        _lx2, _ly2, _lt2, _lh2 = [], [], [], []
        for _k2 in range(len(_ridx2) - 1):
            _si2, _ei2 = _ridx2[_k2], _ridx2[_k2 + 1]
            _tmid2 = (pd.Timestamp(_ts_mkt[_si2])
                      + (pd.Timestamp(_ts_mkt[_ei2]) - pd.Timestamp(_ts_mkt[_si2])) / 2)
            _pmid2 = (_closes[_si2] + _closes[_ei2]) / 2
            _net2  = float(_closes[_ei2]) - float(_closes[_si2])
            _lx2.append(_tmid2)
            _ly2.append(_pmid2)
            _lt2.append(str(_k2 + 1))
            _lh2.append(
                f"<b>Seg {_k2+1}</b>  "
                f"{pd.Timestamp(_ts_mkt[_si2]).strftime('%H:%M')} → "
                f"{pd.Timestamp(_ts_mkt[_ei2]).strftime('%H:%M')}<br>"
                f"Dir: {'↑' if _net2 > 0 else '↓'}  |  Net: {_net2:+.2f} pts"
            )
        _fig2.add_trace(go.Scatter(
            x=_lx2, y=_ly2,
            mode="markers+text",
            marker=dict(size=22, color="rgba(255,255,255,0.12)", symbol="circle",
                        line=dict(color=_color2, width=1)),
            text=_lt2,
            textfont=dict(color=_color2, size=10, family="monospace"),
            textposition="middle center",
            hovertext=_lh2,
            hoverinfo="text",
            showlegend=False,
            name="Segments",
        ))
        _fig2.update_layout(
            xaxis_rangeslider_visible=False,
            height=420, template="plotly_dark",
            margin=dict(l=0, r=0, t=36, b=0),
            title=dict(text=_t2, font=dict(size=13)),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False, automargin=True),
        )
        st.plotly_chart(_fig2, width="stretch", key="tab2_chart")

        # ── Feature table ──
        _feat_rows = []
        for _k2 in range(len(_ridx2) - 1):
            _si2, _ei2 = _ridx2[_k2], _ridx2[_k2 + 1]
            _sc  = _closes[_si2:_ei2 + 1]
            # skip first candle: its high/low happened before close[si] (our entry)
            _sh  = _highs_mkt[_si2 + 1:_ei2 + 1]
            _sl  = _lows_mkt[_si2 + 1:_ei2 + 1]
            _sts = _ts_mkt[_si2:_ei2 + 1]

            _t0  = pd.Timestamp(_sts[0])
            _t1  = pd.Timestamp(_sts[-1])
            _dur = (_t1 - _t0).total_seconds() / 60
            _net = float(_sc[-1]) - float(_sc[0])
            _pth = float(sum(abs(_sc[j + 1] - _sc[j]) for j in range(len(_sc) - 1)))
            _dsp = abs(_net)
            _spd = _net / _dur if _dur > 0 else 0.0
            _er  = abs(_net) / _pth if _pth > 0 else 1.0

            _rvol = float(_np.std(_np.diff(_sc) / (_sc[:-1] + 1e-8)) * 100) if len(_sc) > 1 else 0.0

            _op = float(_sc[0])
            if len(_sh) == 0:
                _mfe = _mae = 0.0
            elif _net >= 0:
                _mfe = float(_np.max(_sh)) - _op
                _mae = _op - float(_np.min(_sl))
            else:
                _mfe = _op - float(_np.min(_sl))
                _mae = float(_np.max(_sh)) - _op

            if len(_sc) >= 3:
                _slp, _, _, _, _se = _linreg(_np.arange(len(_sc)), _sc)
                _tstat = float(_slp / _se) if _se > 0 else 0.0
            else:
                _tstat = 0.0

            _hr  = _t0.hour
            _tod = "Morning" if _hr < 11 else ("Midday" if _hr < 13 else "Afternoon")

            _feat_rows.append({
                "Seg":          _k2 + 1,
                "From":         _t0.strftime("%H:%M"),
                "To":           _t1.strftime("%H:%M"),
                "ToD":          _tod,
                "Dir":          "↑" if _net > 0 else "↓",
                "Disp (pts)":   round(_dsp, 2),
                "Speed (pt/m)": round(_spd, 3),
                "Path (pts)":   round(_pth, 2),
                "ER":           round(_er, 3),
                "Slope t":      round(_tstat, 2),
                "RVol (%)":     round(_rvol, 4),
                "MFE (pts)":    round(_mfe, 2),
                "MAE (pts)":    round(_mae, 2),
            })

        _df_feat = pd.DataFrame(_feat_rows)

        def _er_color(val):
            g = int(min(max(val, 0.0), 1.0) * 200)
            return f"background-color: rgb({200 - g},{g + 55},80); color: black"

        st.dataframe(
            _df_feat.style.map(_er_color, subset=["ER"]),
            width="stretch",
            hide_index=True,
        )

with _tab3:
    st.subheader("Model Backtest Scorecard")
    st.caption(
        f"All 6 models  ·  {len(filtered_dates)} days  ·  "
        f"{n_segments} segments  ·  Day filter: {day_type}"
    )

    st.markdown("**Scoring weights** — adjust to match your trading style")
    _wc = st.columns(5)
    _w_er   = _wc[0].slider("ER",        0.0, 1.0, 0.30, 0.05, key="w_er",
                             help="Higher ER = cleaner, more efficient segments")
    _w_slp  = _wc[1].slider("|Slope t|", 0.0, 1.0, 0.25, 0.05, key="w_slp",
                             help="Higher = statistically stronger trends")
    _w_cap  = _wc[2].slider("Capture",   0.0, 1.0, 0.20, 0.05, key="w_cap",
                             help="Disp÷MFE — closer to 1 = segment ends near its peak")
    _w_pain = _wc[3].slider("Pain ↓",    0.0, 1.0, 0.15, 0.05, key="w_pain",
                             help="MAE÷Disp — lower = less drawdown vs net move (inverted: lower is better)")
    _w_rvol = _wc[4].slider("RVol ↓",    0.0, 1.0, 0.10, 0.05, key="w_rvol",
                             help="Tick-to-tick volatility — lower = smoother segments (inverted: lower is better)")

    if st.button(f"▶  Run Backtest  ({len(filtered_dates)} days)", key="bt_run"):
        with st.spinner("Computing all 6 models across all days…  (L1 may take a minute)"):
            _bt_agg = _run_backtest(df, tuple(filtered_dates), n_segments, open_time_opt)
            st.session_state["bt_agg"]      = _bt_agg
            st.session_state["bt_day_type"] = day_type
            st.session_state["bt_n_seg"]    = n_segments

    if "bt_agg" in st.session_state:
        _bt = st.session_state["bt_agg"]

        if not _bt:
            st.warning("No results — not enough data for the current filter.")
        else:
            # ── Weighted scorecard ──────────────────────────────────────────
            _all_er   = [v["mean_er"]      for v in _bt.values()]
            _all_slp  = [v["mean_slope_t"] for v in _bt.values()]
            _all_cap  = [v["mean_capture"] for v in _bt.values()]
            _all_pain = [v["mean_pain"]    for v in _bt.values()]
            _all_rvol = [v["mean_rvol"]    for v in _bt.values()]

            def _norm01(val, vals, invert=False):
                mn, mx = min(vals), max(vals)
                if mx == mn:
                    return 0.5
                n = (val - mn) / (mx - mn)
                return 1.0 - n if invert else n

            _score_rows = []
            for _mname, _v in _bt.items():
                _s = (
                    _w_er   * _norm01(_v["mean_er"],      _all_er)            +
                    _w_slp  * _norm01(_v["mean_slope_t"], _all_slp)           +
                    _w_cap  * _norm01(_v["mean_capture"], _all_cap)           +
                    _w_pain * _norm01(_v["mean_pain"],    _all_pain, True)    +
                    _w_rvol * _norm01(_v["mean_rvol"],    _all_rvol, True)
                )
                _score_rows.append({
                    "Model":     _mname,
                    "Days":      _v["n_days"],
                    "Mean ER":   _v["mean_er"],
                    "|Slope t|": _v["mean_slope_t"],
                    "Capture":   _v["mean_capture"],
                    "Pain":      _v["mean_pain"],
                    "RVol (%)":  _v["mean_rvol"],
                    "Seg σ":     _v["seg_std"],
                    "Score":     round(_s, 3),
                })

            _df_score = (pd.DataFrame(_score_rows)
                         .sort_values("Score", ascending=False)
                         .reset_index(drop=True))

            _max_score = _df_score["Score"].max()

            def _score_color(val):
                g = int(min(val / (_max_score + 1e-9), 1.0) * 200)
                return f"background-color: rgb({200-g},{g+55},80); color: black"

            st.dataframe(
                _df_score.style.map(_score_color, subset=["Score"]),
                width="stretch",
                hide_index=True,
            )

            st.caption(
                f"Backtest run on: {st.session_state.get('bt_day_type','—')}  ·  "
                f"{st.session_state.get('bt_n_seg','—')} segments  ·  "
                "Re-run button to refresh after changing sidebar filters."
            )

            # ── Per-metric bar charts (2-column grid) ───────────────────────
            st.markdown("**Per-metric breakdown**")
            _models_sorted = _df_score["Model"].tolist()

            def _bar_chart(title, values, lower_better):
                _color = "#ef5350" if lower_better else "#26a69a"
                _bf = go.Figure(go.Bar(
                    x=_models_sorted,
                    y=values,
                    marker_color=_color,
                    text=[f"{v:.3f}" for v in values],
                    textposition="outside",
                ))
                _bf.update_layout(
                    title=dict(
                        text=f"{title}  ({'↓ lower better' if lower_better else '↑ higher better'})",
                        font=dict(size=11)),
                    height=230, template="plotly_dark",
                    margin=dict(l=0, r=0, t=36, b=0),
                    xaxis=dict(showgrid=False, tickfont=dict(size=9)),
                    yaxis=dict(showgrid=False),
                    showlegend=False,
                )
                return _bf

            _sorted_vals = lambda col: [
                _df_score.loc[_df_score["Model"] == m, col].values[0]
                for m in _models_sorted
            ]

            _charts = [
                ("Mean ER",    "Mean ER",   False),
                ("|Slope t|",  "|Slope t|", False),
                ("Capture",    "Capture",   False),
                ("Pain",       "Pain",      True),
                ("RVol (%)",   "RVol (%)",  True),
            ]

            for i in range(0, len(_charts), 2):
                _bc1, _bc2 = st.columns(2)
                for _col, (_ctitle, _ckey, _inv) in zip([_bc1, _bc2], _charts[i:i+2]):
                    _col.plotly_chart(
                        _bar_chart(_ctitle, _sorted_vals(_ckey), _inv),
                        width="stretch", key=f"bt_bar_{_ckey}",
                    )
