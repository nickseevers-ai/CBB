import streamlit as st
import pickle, requests, pandas as pd, numpy as np
from datetime import date

st.set_page_config(page_title="CBB Picks", page_icon="🏀", layout="centered")

# ─── Load saved model assets ──────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    with open("cbb_model_assets.pkl", "rb") as f:
        return pickle.load(f)

try:
    assets = load_assets()
except FileNotFoundError:
    st.error("⚠️ cbb_model_assets.pkl not found — run the notebook through §9a first.")
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

API_KEY = st.secrets.get("CFBD_API_KEY", "")
BASE_URL = "https://api.collegebasketballdata.com"

# ─── Helper functions ─────────────────────────────────────────────────────────
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

@st.cache_data(ttl=1800)
def fetch_schedule(date_str):
    if not API_KEY:
        st.error("Set CFBD_API_KEY in Streamlit secrets (Settings → Secrets).")
        return pd.DataFrame()
    try:
        r = requests.get(
            f"{BASE_URL}/games",
            headers={"Authorization": f"Bearer {API_KEY}"},
            params={
                "startDateRange": f"{date_str}T00:00:00Z",
                "endDateRange"  : f"{date_str}T23:59:59Z",
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.error(f"API error: {e}")
        return pd.DataFrame()

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

    feat_df = pd.DataFrame(rows, columns=FEATURE_COLS)
    return feat_df.fillna(medians)

def predict(sched_df):
    feat_df = build_features(sched_df)
    probs   = model.predict_proba(feat_df.values.astype("float32"))[:, 1]
    out = sched_df.copy().reset_index(drop=True)
    out["home_prob"]  = probs
    out["away_prob"]  = 1 - probs
    out["winner"]     = np.where(probs >= 0.5, out["homeTeam"], out["awayTeam"])
    out["confidence"] = np.maximum(probs, 1 - probs)
    return out.sort_values("confidence", ascending=False).reset_index(drop=True)

# ─── Page UI ──────────────────────────────────────────────────────────────────
st.markdown("# 🏀 CBB Game Predictions")
st.markdown("*XGBoost · Adjusted Efficiency · Four Factors · ELO · Rolling Form*")
st.divider()

col1, col2 = st.columns([4, 1])
with col1:
    sel_date = st.date_input("Select date", value=date.today(), label_visibility="collapsed")
with col2:
    if st.button("📅 Today", use_container_width=True):
        sel_date = date.today()

st.markdown(f"### {sel_date.strftime('%A, %B %d, %Y')}")

with st.spinner("Fetching schedule & running model..."):
    schedule = fetch_schedule(str(sel_date))

if schedule.empty or "homeTeam" not in schedule.columns:
    st.info("No games found for this date — try another day or check back during the season.")
    st.stop()

# Filter to scheduled/final only (skip postponed/cancelled)
if "status" in schedule.columns:
    schedule = schedule[schedule["status"].isin(["scheduled", "in_progress", "final"])].copy()

if schedule.empty:
    st.info("No active games for this date.")
    st.stop()

results = predict(schedule)

# ─── Summary table ────────────────────────────────────────────────────────────
st.markdown(f"**{len(results)} game{'s' if len(results) != 1 else ''} today**")

tbl = results[["awayTeam", "away_prob", "home_prob", "homeTeam", "winner", "confidence"]].copy()
tbl.columns = ["Away", "Away %", "Home %", "Home", "Pick", "Confidence"]
tbl["Away %"]     = (tbl["Away %"]     * 100).round(1).astype(str) + "%"
tbl["Home %"]     = (tbl["Home %"]     * 100).round(1).astype(str) + "%"
tbl["Confidence"] = (tbl["Confidence"] * 100).round(1).astype(str) + "%"
st.dataframe(tbl, use_container_width=True, hide_index=True)

st.divider()

# ─── Game cards ───────────────────────────────────────────────────────────────
for _, g in results.iterrows():
    hp, ap = g["home_prob"], g["away_prob"]
    conf   = g["confidence"]
    winner_is_home = hp >= 0.5

    if conf >= 0.72:
        tier = "🔥 High confidence"
    elif conf >= 0.62:
        tier = "📊 Moderate edge"
    else:
        tier = "🪙 Coin flip"

    away_pick = " ← PICK" if not winner_is_home else ""
    home_pick = " ← PICK" if winner_is_home     else ""
    neutral_tag = "  *(Neutral)*" if g.get("neutralSite", False) else ""

    # Pull adj efficiency for display
    s  = g.get("season", CURRENT_SEASON)
    ht, at = g["homeTeam"], g["awayTeam"]
    h_adj = lookup_adj(ht, s)
    a_adj = lookup_adj(at, s)
    h_net = h_adj.get("netRating", None)
    a_net = a_adj.get("netRating", None)
    h_net_str = f"Net Rtg: {h_net:+.1f}" if pd.notna(h_net) else ""
    a_net_str = f"Net Rtg: {a_net:+.1f}" if pd.notna(a_net) else ""

    h_elo = pd.to_numeric(g.get("homeTeamEloStart"), errors="coerce")
    a_elo = pd.to_numeric(g.get("awayTeamEloStart"), errors="coerce")
    h_elo_str = f"ELO: {h_elo:.0f}" if pd.notna(h_elo) else ""
    a_elo_str = f"ELO: {a_elo:.0f}" if pd.notna(a_elo) else ""

    with st.container(border=True):
        col_a, col_mid, col_h = st.columns([5, 1, 5])

        with col_a:
            st.markdown(f"**🔴 {at}**{away_pick}")
            if a_net_str:
                st.caption(a_net_str + (f"  ·  {a_elo_str}" if a_elo_str else ""))
            st.markdown(f"### {ap:.1%}")

        with col_mid:
            st.markdown("<br>**@**", unsafe_allow_html=True)

        with col_h:
            st.markdown(f"**🔵 {ht}**{home_pick}{neutral_tag}")
            if h_net_str:
                st.caption(h_net_str + (f"  ·  {h_elo_str}" if h_elo_str else ""))
            st.markdown(f"### {hp:.1%}")

        # Gradient probability bar
        st.markdown(
            f'<div style="height:12px;border-radius:6px;margin:2px 0 6px 0;'
            f'background:linear-gradient(to right,#d32f2f {ap*100:.0f}%,#1565c0 {ap*100:.0f}%)">'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Pick: **{g['winner']}** · {tier} · Confidence: {conf:.1%}")

st.caption(
    f"Model trained on {latest_season} seasons · "
    "Data: CollegeBasketballData.com API · "
    "Schedule refreshes every 30 min"
)
