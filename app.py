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
    df = pd.read_csv("Data/NiftySpotPrice.csv", parse_dates=["timestamp"])
    df["ExpiryDate"] = pd.to_datetime(df["ExpiryDate"]).dt.date
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["date"] = df["timestamp"].dt.date
    return df


@st.cache_data
def build_daily(df):
    return (
        df.groupby("date")
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

# ── Derived values ─────────────────────────────────────────────────────────────

idx = all_dates.index(selected_date)
day_row  = daily.iloc[idx]
prev_date = all_dates[idx - 1] if idx > 0 else None
prev_row  = daily.iloc[idx - 1] if idx > 0 else None
day_data = df[df["date"] == selected_date]

_market_open  = pd.Timestamp("09:15").time()
_market_close = pd.Timestamp("15:30").time()
_market_data = day_data[
    (day_data["timestamp"].dt.time >= _market_open) &
    (day_data["timestamp"].dt.time <= _market_close)
]
expiry_tag = expiry_map.get(selected_date, "Other")

if prev_row is not None:
    gap_pts = round(float(day_row["open"]) - float(prev_row["close"]), 2)
    gap_pct = round(gap_pts / float(prev_row["close"]) * 100, 2)
    gap_label = "Gap Up ▲" if gap_pts > 0 else ("Gap Down ▼" if gap_pts < 0 else "Flat Open")
else:
    gap_pts = gap_pct = 0
    gap_label = "Gap"

day_pts = round(float(day_row["close"]) - float(day_row["open"]), 2)
day_pct = round(day_pts / float(day_row["open"]) * 100, 2)

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
    st.metric("Open", f"{day_row['open']:,.2f}")

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
        gap_dir = "Gap Up ▲" if gap_pts > 0 else ("Gap Down ▼" if gap_pts < 0 else "Flat Open")
        st.metric(
            f"Prev Close  ({gap_dir})",
            f"{prev_row['close']:,.2f}",
            delta=f"{gap_pts:+.2f} pts  ({gap_pct:+.2f}%)",
        )
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
    if label == "5 Days":
        return past[-5:] if len(past) >= 5 else past
    if label == "1 Week":
        cutoff = (sel_ts - pd.Timedelta(weeks=1)).date()
        return [d for d in past if d >= cutoff]
    if label == "1 Month":
        cutoff = (sel_ts - pd.DateOffset(months=1)).date()
        return [d for d in past if d >= cutoff]
    return []


cols = st.columns(4)

for col, label in zip(cols, ["1 Day", "5 Days", "1 Week", "1 Month"]):
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
            pdata   = daily[daily["date"].isin(period_dates)].copy()
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
        x=day_data["timestamp"],
        open=day_data["open"],
        high=day_data["high"],
        low=day_data["low"],
        close=day_data["close"],
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
        range=[day_data["timestamp"].iloc[0], day_data["timestamp"].iloc[-1] + pd.Timedelta(minutes=1)],
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

# ── RDP price path ─────────────────────────────────────────────────────────────

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


_closes  = _market_data["close"].values.astype(float)
_ts_mkt  = _market_data["timestamp"].values
_eps     = _epsilon_for_n_segments(_closes, n_segments)
_rdp_idx = _rdp_vertical(_closes, _eps)

fig.add_trace(go.Scatter(
    x=_ts_mkt[_rdp_idx],
    y=_closes[_rdp_idx],
    mode="lines+markers",
    line=dict(color="#f0c040", width=2),
    marker=dict(size=7, color="#f0c040", symbol="circle"),
    name="Price Path",
))

# Global high and low markers with price labels
_gh = day_data.loc[day_data["high"].idxmax()]
_gl = day_data.loc[day_data["low"].idxmin()]

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
for k in range(len(_rdp_idx) - 1):
    si, ei = _rdp_idx[k], _rdp_idx[k + 1]
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
        si, ei = _rdp_idx[row["Seg"] - 1], _rdp_idx[row["Seg"]]
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

st.plotly_chart(fig, width="stretch")

# ── Method Comparison ──────────────────────────────────────────────────────────

st.divider()
st.subheader("Method Comparison  (same day, same slider)")

def _comp_chart(title, path_x, path_y, color):
    cf = go.Figure(go.Candlestick(
        x=day_data["timestamp"],
        open=day_data["open"], high=day_data["high"],
        low=day_data["low"],  close=day_data["close"],
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
    cf.update_layout(
        title=dict(text=title, font=dict(size=12)),
        xaxis_rangeslider_visible=False,
        height=300, template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showgrid=False, automargin=True),
    )
    return cf


import numpy as _np

def _reduce_to_n(idx, closes, target):
    """Greedily remove the least-significant interior point until len(idx)-1 == target."""
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


_methods = []

# ── 0. RDP (current model) ──
_methods.append(("0. RDP  (current model)", True,
                 _ts_mkt[_rdp_idx], _closes[_rdp_idx], "#f0c040"))

# ── 1. Ruptures — Binseg linear ──
try:
    import ruptures as rpt
    _sig  = _closes.reshape(-1, 1)
    _bkps = rpt.Binseg(model="linear").fit(_sig).predict(n_bkps=min(n_segments - 1, len(_closes) - 2))
    _m1i  = sorted(set([0] + [b - 1 for b in _bkps]))
    _methods.append(("1. Ruptures  (Binseg · linear)", True,
                     _ts_mkt[_m1i], _closes[_m1i], "#64b5f6"))
except ImportError:
    _methods.append(("1. Ruptures  (Binseg · linear)", False,
                     None, None, "pip install ruptures"))

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
                     _ts_mkt[_m2i], _closes[_m2i], "#ba68c8"))
except Exception as e:
    _methods.append(("2. ZigZag + ATR", False, None, None, str(e)))

# ── 3. Directional Change (DC) ──
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

_m3i = _dc_idx(_closes, n_segments)
_methods.append(("3. Directional Change (DC)", True,
                 _ts_mkt[_m3i], _closes[_m3i], "#ff8a65"))

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
            prob.solve(solver=cp.OSQP, verbose=False)
            if xv.value is None:
                lo = lam; continue
            sc = _np.where(_np.diff(_np.sign(_np.diff(xv.value))) != 0)[0] + 1
            if len(sc) <= target - 1:
                hi = lam
            else:
                lo = lam
        prob = cp.Problem(cp.Minimize(cp.sum_squares(closes - xv) + hi * cp.norm1(cp.diff(xv, 2))))
        prob.solve(solver=cp.OSQP, verbose=False)
        if xv.value is not None:
            sc  = _np.where(_np.diff(_np.sign(_np.diff(xv.value))) != 0)[0] + 1
            idx = sorted(set([0] + list(sc) + [n - 1]))
        else:
            idx = [0, n - 1]
        return _reduce_to_n(idx, closes, target)

    _m4i = _l1_idx(_closes, n_segments)
    _methods.append(("4. L1 Trend Filter", True,
                     _ts_mkt[_m4i], _closes[_m4i], "#a5d6a7"))
except ImportError:
    _methods.append(("4. L1 Trend Filter", False, None, None, "pip install cvxpy"))

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
                     _ts_mkt[_m5i], _closes[_m5i], "#fff176"))
except ImportError:
    _methods.append(("5. HMM  (3-state → reduced)", False,
                     None, None, "pip install hmmlearn"))

# Display in 3-row × 2-column grid
for row in range(3):
    _c1, _c2 = st.columns(2)
    for col_i, _col in enumerate([_c1, _c2]):
        _idx = row * 2 + col_i
        if _idx >= len(_methods):
            break
        _title, _ok, _mx, _my, *_rest = _methods[_idx]
        _color = _rest[0] if _ok else None
        _err   = _rest[0] if not _ok else None
        with _col:
            if _ok:
                st.plotly_chart(_comp_chart(_title, _mx, _my, _color),
                                width="stretch", key=f"cmp_{_idx}")
            else:
                st.markdown(f"**{_title}**")
                st.warning(f"Not available — {_err}")

