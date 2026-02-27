import streamlit as st
import pickle, requests, pandas as pd, numpy as np
from datetime import date, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="CBB Picks", page_icon="🏀", layout="centered")

# ─── Load saved model assets ──────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    with open("cbb_model_assets.pkl", "rb") as f:
        return pickle.load(f)

try:
    assets = load_assets()
except FileNotFoundError:
    st.error("cbb_model_assets.pkl not found — run the notebook through §9a first.")
    st.stop()

model              = assets["model"]
FEATURE_COLS       = assets["feature_cols"]
medians            = pd.Series(assets["medians"])
adj_ratings        = assets["adj_ratings"]
team_stats         = assets["team_stats"]
latest_form_roll   = assets["latest_form_roll"]
latest_form_season = assets["latest_form_season"]
latest_season      = assets["latest_season"]
CURRENT_SEASON     = assets["current_season"]

API_KEY  = st.secrets.get("upIJ74md21T7tdxEJop2P8O3UP0ydqHQLzm0H1VB0hZSOmeyhJsxA5qO2F6Aft8U", "")
BASE_URL = "https://api.collegebasketballdata.com"

# ─── Lookup helpers ───────────────────────────────────────────────────────────
def lookup_adj(team, season):
    m = adj_ratings[(adj_ratings["team"] == team) & (adj_ratings["season"] == season)]
    if m.empty:
        m = adj_ratings[(adj_ratings["team"] == team) & (adj_ratings["season"] == season - 1)]
    return m.iloc[0] if not m.empty else pd.Series(dtype=float)

def lookup_stats(team, season):
    m = team_stats[(team_stats["team"] == team) & (team_stats["season"] == season)]
    if m.empty:
        m = team_stats[(team_stats["team"] == team) & (team_stats["season"] == season - 1)]
    return m.iloc[0] if not m.empty else pd.Series(dtype=float)

# ─── Spread / ATS utilities ───────────────────────────────────────────────────
def spread_to_implied_prob(home_spread):
    """Home spread (negative = home favored) → implied home win probability."""
    if pd.isna(home_spread):
        return np.nan
    return 1.0 / (1.0 + np.exp(home_spread * 0.12))

def prob_to_spread(prob):
    """Home win probability → implied home spread (negative = home favored)."""
    if pd.isna(prob) or prob <= 0.01 or prob >= 0.99:
        return np.nan
    return -np.log(prob / (1.0 - prob)) / 0.12

def flatten_lines(raw):
    """
    Flatten /lines API response → dict {game_id: home_spread}.
    home_spread < 0 means home team is favored.
    """
    result = {}
    if not raw:
        return result
    for game in raw:
        gid = game.get("id")
        if gid is None:
            continue
        lines_list = game.get("lines") or []
        for line in lines_list:
            s = line.get("spread")
            if s is not None:
                result[int(gid)] = {
                    "home_spread": float(s),
                    "over_under" : line.get("overUnder"),
                    "provider"   : line.get("provider", ""),
                }
                break   # use first provider that has a spread
    return result

def fmt_spread_label(home_spread, home_team, away_team):
    """e.g. 'Duke -7.0'  or  'Kentucky -3.0'  or  'PK'"""
    if pd.isna(home_spread):
        return None
    if home_spread < 0:
        return f"{home_team} {home_spread:.1f}"
    elif home_spread > 0:
        return f"{away_team} -{home_spread:.1f}"
    return "PK"

def ats_lean(home_prob, home_spread, home_team, away_team):
    """
    Return (lean_str, spread_edge) for display.
    spread_edge < 0  →  model likes home more than line (take home to cover)
    spread_edge > 0  →  model likes away more than line (take away to cover)
    """
    if pd.isna(home_spread) or pd.isna(home_prob):
        return None, np.nan
    model_spd  = prob_to_spread(home_prob)
    if pd.isna(model_spd):
        return None, np.nan
    edge = model_spd - home_spread          # negative = model likes home to cover
    abs_edge = abs(edge)
    if edge < -2.5:
        return f"ATS lean: **{home_team}** covers  ({abs_edge:.1f} pt edge)", edge
    elif edge > 2.5:
        return f"ATS lean: **{away_team}** covers  ({abs_edge:.1f} pt edge)", edge
    return "ATS: no clear edge vs spread", edge

# ─── API helpers ──────────────────────────────────────────────────────────────
def _get(path, params):
    r = requests.get(
        f"{BASE_URL}/{path}",
        headers={"Authorization": f"Bearer {API_KEY}"},
        params=params,
        timeout=25,
    )
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=1800)
def fetch_schedule(date_str):
    if not API_KEY:
        return pd.DataFrame()
    try:
        data = _get("games", {
            "startDateRange": f"{date_str}T00:00:00Z",
            "endDateRange"  : f"{date_str}T23:59:59Z",
        })
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.error(f"Schedule API error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def fetch_lines_date(date_str):
    if not API_KEY:
        return {}
    try:
        data = _get("lines", {
            "startDateRange": f"{date_str}T00:00:00Z",
            "endDateRange"  : f"{date_str}T23:59:59Z",
        })
        return flatten_lines(data)
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def fetch_games_range(start_str, end_str):
    if not API_KEY:
        return pd.DataFrame()
    try:
        data = _get("games", {
            "startDateRange": f"{start_str}T00:00:00Z",
            "endDateRange"  : f"{end_str}T23:59:59Z",
            "status"        : "final",
        })
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.error(f"Backtest games API error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_lines_range(start_str, end_str):
    if not API_KEY:
        return {}
    try:
        data = _get("lines", {
            "startDateRange": f"{start_str}T00:00:00Z",
            "endDateRange"  : f"{end_str}T23:59:59Z",
        })
        return flatten_lines(data)
    except Exception:
        return {}

# ─── Feature builder & predictor ─────────────────────────────────────────────
def build_features(sched_df):
    rows = []
    for _, g in sched_df.iterrows():
        ht = g.get("homeTeam", "")
        at = g.get("awayTeam", "")
        s  = g.get("season", CURRENT_SEASON)

        h_adj  = lookup_adj(ht, s)
        a_adj  = lookup_adj(at, s)
        h_stat = lookup_stats(ht, s)
        a_stat = lookup_stats(at, s)

        h_r10 = latest_form_roll.get(ht, 0.5)
        a_r10 = latest_form_roll.get(at, 0.5)
        h_swp = latest_form_season.get(ht, 0.5)
        a_swp = latest_form_season.get(at, 0.5)

        h_elo = pd.to_numeric(g.get("homeTeamEloStart"), errors="coerce")
        a_elo = pd.to_numeric(g.get("awayTeamEloStart"), errors="coerce")
        neutral = 1 if g.get("neutralSite", False) else 0

        rows.append({
            "elo_diff"         : (h_elo - a_elo) if (pd.notna(h_elo) and pd.notna(a_elo)) else np.nan,
            "home_advantage"   : 1 - neutral,
            "conf_game"        : 1 if g.get("conferenceGame", False) else 0,
            "seed_diff"        : np.nan,
            "adj_off_diff"     : h_adj.get("offensiveRating", np.nan) - a_adj.get("offensiveRating", np.nan),
            "adj_def_diff"     : a_adj.get("defensiveRating", np.nan) - h_adj.get("defensiveRating", np.nan),
            "net_rating_diff"  : h_adj.get("netRating", np.nan) - a_adj.get("netRating", np.nan),
            "efg_diff"         : h_stat.get("efg_pct", np.nan) - a_stat.get("efg_pct", np.nan),
            "to_diff"          : a_stat.get("to_ratio", np.nan) - h_stat.get("to_ratio", np.nan),
            "orb_diff"         : h_stat.get("orb_pct", np.nan) - a_stat.get("orb_pct", np.nan),
            "ftr_diff"         : h_stat.get("ft_rate", np.nan) - a_stat.get("ft_rate", np.nan),
            "fg3_diff"         : h_stat.get("fg3_pct", np.nan) - a_stat.get("fg3_pct", np.nan),
            "ft_pct_diff"      : h_stat.get("ft_pct", np.nan) - a_stat.get("ft_pct", np.nan),
            "pace_diff"        : h_stat.get("pace", np.nan) - a_stat.get("pace", np.nan),
            "form_roll10_diff" : h_r10 - a_r10,
            "form_season_diff" : h_swp - a_swp,
        })

    return pd.DataFrame(rows, columns=FEATURE_COLS).fillna(medians)

def predict(sched_df):
    probs = model.predict_proba(build_features(sched_df).values.astype("float32"))[:, 1]
    out = sched_df.copy().reset_index(drop=True)
    out["home_prob"]  = probs
    out["away_prob"]  = 1 - probs
    out["winner"]     = np.where(probs >= 0.5, out["homeTeam"], out["awayTeam"])
    out["confidence"] = np.maximum(probs, 1 - probs)
    return out.sort_values("confidence", ascending=False).reset_index(drop=True)

def attach_lines(results_df, lines_map):
    """Join lines_map onto results_df by game id."""
    if lines_map and "id" in results_df.columns:
        results_df["home_spread"] = results_df["id"].apply(
            lambda x: lines_map.get(int(x), {}).get("home_spread") if pd.notna(x) else np.nan
        )
    else:
        results_df["home_spread"] = np.nan
    return results_df

# ─── Page ─────────────────────────────────────────────────────────────────────
st.markdown("# 🏀 CBB Game Predictions")
st.markdown("*XGBoost · Adjusted Efficiency · Four Factors · ELO · Rolling Form*")
st.divider()

tab_picks, tab_backtest = st.tabs(["📅 Today's Picks", "📊 Backtest"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PICKS
# ══════════════════════════════════════════════════════════════════════════════
def show_picks_tab():
    c1, c2 = st.columns([4, 1])
    with c1:
        sel_date = st.date_input("Date", value=date.today(),
                                 label_visibility="collapsed", key="picks_date")
    with c2:
        if st.button("Today", use_container_width=True, key="today_btn"):
            sel_date = date.today()

    st.markdown(f"### {sel_date.strftime('%A, %B %d, %Y')}")

    with st.spinner("Fetching schedule, lines & running model..."):
        schedule  = fetch_schedule(str(sel_date))
        lines_map = fetch_lines_date(str(sel_date))

    if schedule.empty or "homeTeam" not in schedule.columns:
        st.info("No games found for this date.")
        return

    if "status" in schedule.columns:
        schedule = schedule[schedule["status"].isin(
            ["scheduled", "in_progress", "final"])].copy()

    if schedule.empty:
        st.info("No active games for this date.")
        return

    results = predict(schedule)
    results = attach_lines(results, lines_map)
    n_lines = results["home_spread"].notna().sum()

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown(f"**{len(results)} game{'s' if len(results)!=1 else ''}"
                f"{f' · {n_lines} with lines' if n_lines else ''}**")

    tbl = {
        "Away"      : results["awayTeam"],
        "Away %"    : (results["away_prob"]  * 100).round(1).astype(str) + "%",
        "Home %"    : (results["home_prob"]  * 100).round(1).astype(str) + "%",
        "Home"      : results["homeTeam"],
        "Pick"      : results["winner"],
        "Conf"      : (results["confidence"] * 100).round(1).astype(str) + "%",
    }
    if n_lines:
        tbl["Line"] = [
            fmt_spread_label(r["home_spread"], r["homeTeam"], r["awayTeam"]) or "N/A"
            for _, r in results.iterrows()
        ]
        tbl["ATS Lean"] = [
            ats_lean(r["home_prob"], r["home_spread"], r["homeTeam"], r["awayTeam"])[0] or "—"
            for _, r in results.iterrows()
        ]
    st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)
    st.divider()

    # ── Game cards ────────────────────────────────────────────────────────────
    for _, g in results.iterrows():
        hp, ap   = g["home_prob"], g["away_prob"]
        conf     = g["confidence"]
        is_home  = hp >= 0.5
        ht, at   = g["homeTeam"], g["awayTeam"]
        s        = g.get("season", CURRENT_SEASON)
        hs       = g.get("home_spread", np.nan)

        tier = ("🔥 High confidence" if conf >= 0.72
                else "📊 Moderate edge" if conf >= 0.62
                else "🪙 Coin flip")

        h_adj = lookup_adj(ht, s)
        a_adj = lookup_adj(at, s)
        h_net = h_adj.get("netRating", np.nan)
        a_net = a_adj.get("netRating", np.nan)
        h_elo = pd.to_numeric(g.get("homeTeamEloStart"), errors="coerce")
        a_elo = pd.to_numeric(g.get("awayTeamEloStart"), errors="coerce")

        spread_label = fmt_spread_label(hs, ht, at)
        lean_str, _  = ats_lean(hp, hs, ht, at)

        with st.container(border=True):
            col_a, col_mid, col_h = st.columns([5, 1, 5])

            with col_a:
                arrow = " ← PICK" if not is_home else ""
                st.markdown(f"**🔴 {at}**{arrow}")
                caps = []
                if pd.notna(a_net): caps.append(f"Net Rtg: {a_net:+.1f}")
                if pd.notna(a_elo): caps.append(f"ELO: {a_elo:.0f}")
                if caps: st.caption("  ·  ".join(caps))
                st.markdown(f"### {ap:.1%}")

            with col_mid:
                st.markdown("<br>**@**", unsafe_allow_html=True)

            with col_h:
                arrow = " ← PICK" if is_home else ""
                neutral = "  *(Neutral)*" if g.get("neutralSite", False) else ""
                st.markdown(f"**🔵 {ht}**{arrow}{neutral}")
                caps = []
                if pd.notna(h_net): caps.append(f"Net Rtg: {h_net:+.1f}")
                if pd.notna(h_elo): caps.append(f"ELO: {h_elo:.0f}")
                if caps: st.caption("  ·  ".join(caps))
                st.markdown(f"### {hp:.1%}")

            st.markdown(
                f'<div style="height:12px;border-radius:6px;margin:2px 0 4px 0;'
                f'background:linear-gradient(to right,'
                f'#d32f2f {ap*100:.0f}%,#1565c0 {ap*100:.0f}%)"></div>',
                unsafe_allow_html=True,
            )

            bottom = [f"Pick: **{g['winner']}**", tier, f"Conf: {conf:.1%}"]
            if spread_label:
                bottom.append(f"Line: {spread_label}")
            st.caption("  ·  ".join(bottom))

            if lean_str:
                icon = "🟢" if "lean" in lean_str else "⚪"
                st.caption(f"{icon} {lean_str}")

    st.caption(
        f"Model trained on {latest_season} seasons · "
        "CollegeBasketballData.com · Schedule refreshes every 30 min"
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
def show_backtest_tab():
    st.markdown("### Model Backtest — Current Season")
    st.caption(
        "All 2025-26 games are out-of-sample (model trained through 2024-25). "
        "Ratings are season-to-date as of today — slight lookahead vs true real-time."
    )

    bc1, bc2 = st.columns(2)
    with bc1:
        bt_start = st.date_input("From", value=date.today() - timedelta(days=30),
                                 key="bt_start")
    with bc2:
        bt_end = st.date_input("To", value=date.today() - timedelta(days=1),
                               key="bt_end")

    if bt_start >= bt_end:
        st.warning("Start date must be before end date.")
        return
    if (bt_end - bt_start).days > 60:
        st.warning("Range capped at 60 days.")
        bt_end = bt_start + timedelta(days=60)

    with st.spinner(f"Fetching {bt_start} → {bt_end}..."):
        bt_games = fetch_games_range(str(bt_start), str(bt_end))
        bt_lines = fetch_lines_range(str(bt_start), str(bt_end))

    if bt_games.empty or "homeTeam" not in bt_games.columns:
        st.info("No completed games in this range.")
        return

    # Keep only finals with a declared winner
    bt_games = bt_games[
        (bt_games["status"] == "final") & bt_games["homeWinner"].notna()
    ].copy()

    if bt_games.empty:
        st.info("No finalized games found.")
        return

    with st.spinner(f"Running model on {len(bt_games)} games..."):
        bt_res = predict(bt_games)

    bt_res = attach_lines(bt_res, bt_lines)

    # Actual outcomes
    bt_res["actual_home_win"] = bt_games["homeWinner"].astype(int).values
    bt_res["correct"] = (
        (bt_res["home_prob"] >= 0.5) == (bt_res["actual_home_win"] == 1)
    )
    bt_res["actual_winner"] = np.where(
        bt_res["actual_home_win"] == 1, bt_res["homeTeam"], bt_res["awayTeam"]
    )

    # Home margin for ATS
    bt_res["home_margin"] = (
        pd.to_numeric(bt_games["homePoints"], errors="coerce").values -
        pd.to_numeric(bt_games["awayPoints"], errors="coerce").values
    )

    # ATS: home covers when home_margin + home_spread > 0
    # (e.g., home -7: need home_margin > 7, i.e., home_margin + (-7) > 0)
    has_line = bt_res["home_spread"].notna() & bt_res["home_margin"].notna()
    bt_res["home_covers"]   = np.where(has_line,
        bt_res["home_margin"] + bt_res["home_spread"] > 0, np.nan)
    bt_res["model_spread"]  = bt_res["home_prob"].apply(prob_to_spread)
    # model likes home to cover when model_spread < home_spread
    # (model sees home as bigger favorite than the line)
    bt_res["ats_pick_home"] = np.where(has_line,
        bt_res["model_spread"] < bt_res["home_spread"], np.nan)
    bt_res["ats_correct"]   = np.where(has_line,
        bt_res["ats_pick_home"] == bt_res["home_covers"], np.nan)

    # ── Metrics ───────────────────────────────────────────────────────────────
    n_games   = len(bt_res)
    n_correct = int(bt_res["correct"].sum())
    acc       = n_correct / n_games if n_games else 0

    ats_mask   = has_line & bt_res["ats_correct"].notna()
    n_ats      = int(ats_mask.sum())
    n_ats_w    = int(bt_res.loc[ats_mask, "ats_correct"].sum())
    ats_acc    = n_ats_w / n_ats if n_ats else 0

    hc         = bt_res[bt_res["confidence"] >= 0.72]
    hc_acc     = hc["correct"].mean() if len(hc) else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("SU Record",    f"{n_correct}-{n_games - n_correct}")
    m2.metric("SU Accuracy",  f"{acc:.1%}")
    if n_ats:
        m3.metric("ATS Record",   f"{n_ats_w}-{n_ats - n_ats_w}")
        m4.metric("ATS Accuracy", f"{ats_acc:.1%}")
    else:
        m3.metric("ATS",          "No lines data")
        m4.metric("High Conf SU", f"{hc_acc:.1%}" if len(hc) else "N/A",
                  f"{len(hc)} games")

    if len(hc) and n_ats:
        hc_ats = bt_res.loc[(bt_res["confidence"] >= 0.72) & ats_mask, "ats_correct"]
        st.caption(
            f"High confidence (>=72%): {int(hc['correct'].sum())}-"
            f"{len(hc)-int(hc['correct'].sum())} SU ({hc_acc:.1%})"
        )

    st.divider()

    # ── Cumulative accuracy chart ─────────────────────────────────────────────
    if n_games >= 10:
        chart_df = bt_res.copy()
        if "startDate" in chart_df.columns:
            chart_df = chart_df.sort_values(
                pd.to_datetime(chart_df["startDate"], utc=True, errors="coerce")
            )
        chart_df["cum_acc"] = chart_df["correct"].expanding().mean()
        chart_df["game_n"]  = range(1, len(chart_df) + 1)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(chart_df["game_n"], chart_df["cum_acc"],
                color="#1565c0", lw=2, label="SU accuracy")
        if n_ats >= 10:
            ats_df = chart_df[ats_mask.values].copy()
            ats_df["cum_ats"] = ats_df["ats_correct"].astype(float).expanding().mean()
            ats_df["ats_n"]   = range(1, len(ats_df) + 1)
            ax.plot(ats_df["ats_n"], ats_df["cum_ats"],
                    color="#e67e22", lw=2, linestyle="--", label="ATS accuracy")
        ax.axhline(0.5,  color="gray",   ls="--", alpha=0.4, lw=1)
        ax.axhline(acc,  color="#1565c0", ls=":",  alpha=0.6, lw=1,
                   label=f"SU final {acc:.1%}")
        ax.set_xlabel("Game number (sorted by date)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Cumulative Model Accuracy — Current Season Backtest")
        ax.set_ylim(0.35, 1.0)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.divider()

    # ── Game-by-game table ────────────────────────────────────────────────────
    st.markdown("#### Game-by-Game Results")

    tbl2 = {
        "Date"       : (pd.to_datetime(bt_res["startDate"], utc=True, errors="coerce")
                        .dt.strftime("%m/%d")
                        if "startDate" in bt_res.columns
                        else bt_res.index.astype(str)),
        "Away"       : bt_res["awayTeam"],
        "Home"       : bt_res["homeTeam"],
        "Pick"       : bt_res["winner"],
        "Actual"     : bt_res["actual_winner"],
        "Conf"       : (bt_res["confidence"] * 100).round(1).astype(str) + "%",
        "SU"         : bt_res["correct"].map({True: "W", False: "L"}),
    }
    if n_ats:
        tbl2["Spread"] = bt_res["home_spread"].apply(
            lambda x: f"{x:+.1f}" if pd.notna(x) else "—"
        )
        tbl2["ATS"] = bt_res["ats_correct"].map(
            {True: "W", False: "L", 1.0: "W", 0.0: "L"}
        ).fillna("—")

    st.dataframe(pd.DataFrame(tbl2), use_container_width=True, hide_index=True)

    st.caption(
        f"{n_games} games · {n_ats} with lines · "
        "Caveat: ratings used are season-to-date, not game-date snapshots"
    )

# ─── Render tabs ──────────────────────────────────────────────────────────────
with tab_picks:
    show_picks_tab()

with tab_backtest:
    show_backtest_tab()
