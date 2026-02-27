"""
build_cbb_nb.py  —  generates cbb_game_winner.ipynb
Run:  python build_cbb_nb.py
"""
import json, textwrap

def code_cell(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(src).lstrip(),
    }

def md_cell(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(src).lstrip(),
    }

cells = []

# ── §0 Config ─────────────────────────────────────────────────────────────────
cells.append(md_cell("## §0 · Config & Imports"))
cells.append(code_cell("""
    import requests, pandas as pd, numpy as np, time, pickle, warnings
    from datetime import datetime, date
    warnings.filterwarnings("ignore")

    API_KEY  = "YOUR_API_KEY_HERE"   # ← paste your CBBD API key here
    BASE_URL = "https://api.collegebasketballdata.com"
    HEADERS  = {"Authorization": f"Bearer {API_KEY}"}

    # Seasons to train on (season int = year season ENDS, e.g. 2024 = 2023-24)
    SEASONS = list(range(2016, 2026))    # 10 seasons: 2015-16 through 2024-25
    CURRENT_SEASON = 2026               # 2025-26 (for predictions)

    print(f"Training seasons: {SEASONS[0]}–{SEASONS[-1]}")
    print(f"Prediction season: {CURRENT_SEASON}")
"""))

# ── §1 Fetch Historical Games ─────────────────────────────────────────────────
cells.append(md_cell("## §1 · Fetch Historical Games"))
cells.append(code_cell("""
    def fetch_games(season, season_type="regular"):
        url = f"{BASE_URL}/games"
        params = {"season": season, "seasonType": season_type, "status": "final"}
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    all_games = []
    for s in SEASONS:
        for stype in ["regular", "postseason"]:
            g = fetch_games(s, stype)
            all_games.extend(g)
            print(f"Season {s} {stype:12s}: {len(g):5d} games")
            time.sleep(0.25)

    games_raw = pd.DataFrame(all_games)
    print(f"\\nTotal raw games : {len(games_raw)}")
    print(f"Columns         : {games_raw.columns.tolist()}")
"""))

# ── §2 Clean & Filter Games ───────────────────────────────────────────────────
cells.append(md_cell("## §2 · Clean & Filter Games"))
cells.append(code_cell("""
    games = games_raw[games_raw["status"] == "final"].copy()
    games = games.dropna(subset=["homePoints", "awayPoints"]).copy()

    # Parse date
    games["startDate"] = pd.to_datetime(games["startDate"], utc=True)
    games["game_date"] = games["startDate"].dt.date

    # Target
    games["home_win"] = games["homeWinner"].astype(int)

    # Context flags
    games["neutral"]        = games["neutralSite"].fillna(False).astype(int)
    games["home_advantage"] = 1 - games["neutral"]
    games["conf_game"]      = games["conferenceGame"].fillna(False).astype(int)

    # Seeds (postseason / tournament games)
    games["homeSeed"] = pd.to_numeric(games.get("homeSeed", np.nan), errors="coerce")
    games["awaySeed"] = pd.to_numeric(games.get("awaySeed", np.nan), errors="coerce")
    games["seed_diff"] = games["homeSeed"] - games["awaySeed"]

    # ELO embedded in game data — pre-game values
    games["elo_diff"] = (
        pd.to_numeric(games["homeTeamEloStart"], errors="coerce") -
        pd.to_numeric(games["awayTeamEloStart"], errors="coerce")
    )

    games_sorted = games.sort_values("startDate").reset_index(drop=True)

    print(f"Final games    : {len(games_sorted)}")
    print(f"Home win rate  : {games_sorted.home_win.mean():.3f}")
    print(f"Neutral-site   : {games_sorted.neutral.sum()} ({games_sorted.neutral.mean():.2%})")
    games_sorted[["homeTeam","awayTeam","homePoints","awayPoints",
                  "home_win","neutral","elo_diff","season"]].head()
"""))

# ── §3 Rolling Win% Form ──────────────────────────────────────────────────────
cells.append(md_cell("## §3 · Rolling Win% Form (no data leakage)"))
cells.append(code_cell("""
    # Build per-team record table
    # Use rename() so column count never mismatches (avoids ValueError if extra cols exist)
    home_rec = (
        games_sorted[["startDate","homeTeam","home_win","season"]]
        .rename(columns={"homeTeam": "team", "home_win": "win"})
        .copy()
    )

    away_rec = games_sorted[["startDate","awayTeam","home_win","season"]].copy()
    away_rec["win"] = 1 - away_rec["home_win"]
    # Select only the 4 columns we want BEFORE renaming (drops the 'home_win' column)
    away_rec = (
        away_rec[["startDate","awayTeam","win","season"]]
        .rename(columns={"awayTeam": "team"})
    )

    all_rec = pd.concat([home_rec, away_rec]).sort_values("startDate").reset_index(drop=True)

    # shift(1) prevents using the current game's result
    all_rec["roll_win10"] = (
        all_rec.groupby("team")["win"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )
    all_rec["season_winpct"] = (
        all_rec.groupby(["team","season"])["win"]
        .transform(lambda x: x.shift(1).expanding(min_periods=5).mean())
    )

    # Build lookup dict: (startDate_timestamp, team) -> {roll_win10, season_winpct}
    form_map = (
        all_rec.set_index(["startDate","team"])
        [["roll_win10","season_winpct"]]
        .to_dict("index")
    )

    print(f"Form map entries : {len(form_map)}")
    print(all_rec[["team","win","roll_win10","season_winpct"]].dropna().tail(5).to_string())
"""))

# ── §4 Fetch Adjusted Efficiency Ratings ─────────────────────────────────────
cells.append(md_cell("## §4 · Adjusted Efficiency Ratings (KenPom-style)"))
cells.append(code_cell("""
    def fetch_adj_ratings(season):
        r = requests.get(f"{BASE_URL}/ratings/adjusted",
                         headers=HEADERS, params={"season": season}, timeout=30)
        r.raise_for_status()
        rows = r.json()
        for row in rows:
            row["season"] = season
        return rows

    adj_list = []
    for s in SEASONS:
        data = fetch_adj_ratings(s)
        adj_list.extend(data)
        print(f"Season {s}: {len(data)} teams")
        time.sleep(0.25)

    adj_ratings = pd.DataFrame(adj_list)
    print(f"\\nTotal rows : {len(adj_ratings)}")
    print(f"Columns    : {adj_ratings.columns.tolist()}")
    adj_ratings[["season","team","offensiveRating","defensiveRating","netRating"]].head(5)
"""))

# ── §5 Fetch Team Season Stats ────────────────────────────────────────────────
cells.append(md_cell("## §5 · Team Season Stats (Four Factors + Pace)"))
cells.append(code_cell("""
    def fetch_team_stats(season):
        r = requests.get(f"{BASE_URL}/stats/team/season",
                         headers=HEADERS, params={"season": season}, timeout=30)
        r.raise_for_status()
        return r.json()

    def flatten_stats(row):
        ts  = row.get("teamStats", {}) or {}
        ff  = ts.get("fourFactors", {}) or {}
        fg3 = ts.get("threePointFieldGoals", {}) or {}
        ft  = ts.get("freeThrows", {}) or {}
        rb  = ts.get("rebounds", {}) or {}
        fg  = ts.get("fieldGoals", {}) or {}
        return {
            "team"    : row.get("team"),
            "season"  : row.get("season"),
            "games"   : row.get("games", np.nan),
            "wins"    : row.get("wins", np.nan),
            "pace"    : row.get("pace", np.nan),
            "fg_pct"  : fg.get("pct", np.nan),
            "fg3_pct" : fg3.get("pct", np.nan),
            "ft_pct"  : ft.get("pct", np.nan),
            # Dean Oliver's Four Factors
            "efg_pct" : ff.get("effectiveFieldGoalPct", np.nan),
            "to_ratio": ff.get("turnoverRatio", np.nan),
            "orb_pct" : ff.get("offensiveReboundPct", np.nan),
            "ft_rate" : ff.get("freeThrowRate", np.nan),
            # Additional
            "orb_raw" : rb.get("offensive", np.nan),
            "drb_raw" : rb.get("defensive", np.nan),
        }

    stats_list = []
    for s in SEASONS:
        raw = fetch_team_stats(s)
        for row in raw:
            row["season"] = s
        stats_list.extend(raw)
        print(f"Season {s}: {len(raw)} teams")
        time.sleep(0.25)

    team_stats = pd.DataFrame([flatten_stats(r) for r in stats_list])
    print(f"\\nFlattened rows : {len(team_stats)}")
    team_stats.head(3)
"""))

# ── §6 Feature Assembly ───────────────────────────────────────────────────────
cells.append(md_cell("## §6 · Feature Assembly"))
cells.append(code_cell("""
    def lookup_adj(team, season, df=None):
        if df is None:
            df = adj_ratings
        m = df[(df["team"] == team) & (df["season"] == season)]
        if m.empty:
            m = df[(df["team"] == team) & (df["season"] == season - 1)]
        return m.iloc[0] if not m.empty else pd.Series(dtype=float)

    def lookup_stats(team, season, df=None):
        if df is None:
            df = team_stats
        m = df[(df["team"] == team) & (df["season"] == season)]
        if m.empty:
            m = df[(df["team"] == team) & (df["season"] == season - 1)]
        return m.iloc[0] if not m.empty else pd.Series(dtype=float)

    def get_form(team, ts):
        val = form_map.get((ts, team), {})
        return val.get("roll_win10", np.nan), val.get("season_winpct", np.nan)

    feature_rows = []
    for _, g in games_sorted.iterrows():
        ht = g["homeTeam"]
        at = g["awayTeam"]
        s  = g["season"]
        ts = g["startDate"]

        h_adj  = lookup_adj(ht, s)
        a_adj  = lookup_adj(at, s)
        h_stat = lookup_stats(ht, s)
        a_stat = lookup_stats(at, s)
        h_r10, h_swp = get_form(ht, ts)
        a_r10, a_swp = get_form(at, ts)

        feat = {
            # ELO (pre-game, from API)
            "elo_diff"         : g.get("elo_diff", np.nan),
            # Game context
            "home_advantage"   : g.get("home_advantage", 1),
            "conf_game"        : g.get("conf_game", 0),
            "seed_diff"        : g.get("seed_diff", np.nan),
            # Adjusted efficiency (KenPom-style)
            "adj_off_diff"     : h_adj.get("offensiveRating", np.nan) - a_adj.get("offensiveRating", np.nan),
            "adj_def_diff"     : a_adj.get("defensiveRating", np.nan) - h_adj.get("defensiveRating", np.nan),
            "net_rating_diff"  : h_adj.get("netRating", np.nan) - a_adj.get("netRating", np.nan),
            # Four Factors diffs (positive = home advantage)
            "efg_diff"         : h_stat.get("efg_pct", np.nan) - a_stat.get("efg_pct", np.nan),
            "to_diff"          : a_stat.get("to_ratio", np.nan) - h_stat.get("to_ratio", np.nan),
            "orb_diff"         : h_stat.get("orb_pct", np.nan) - a_stat.get("orb_pct", np.nan),
            "ftr_diff"         : h_stat.get("ft_rate", np.nan) - a_stat.get("ft_rate", np.nan),
            # Shooting
            "fg3_diff"         : h_stat.get("fg3_pct", np.nan) - a_stat.get("fg3_pct", np.nan),
            "ft_pct_diff"      : h_stat.get("ft_pct", np.nan) - a_stat.get("ft_pct", np.nan),
            # Pace
            "pace_diff"        : h_stat.get("pace", np.nan) - a_stat.get("pace", np.nan),
            # Form
            "form_roll10_diff" : (h_r10 - a_r10) if (pd.notna(h_r10) and pd.notna(a_r10)) else np.nan,
            "form_season_diff" : (h_swp - a_swp) if (pd.notna(h_swp) and pd.notna(a_swp)) else np.nan,
            # Target
            "home_win"         : g["home_win"],
        }
        feature_rows.append(feat)

    feat_df = pd.DataFrame(feature_rows)
    print(f"Feature rows  : {len(feat_df)}")
    print(f"Home win rate : {feat_df.home_win.mean():.3f}")
    print(f"NaN counts:\\n{feat_df.isna().sum().sort_values(ascending=False).head(10)}")
    feat_df.describe()
"""))

# ── §7 Dataset Prep ───────────────────────────────────────────────────────────
cells.append(md_cell("## §7 · Dataset Prep"))
cells.append(code_cell("""
    from sklearn.model_selection import train_test_split

    FEATURE_COLS = [c for c in feat_df.columns if c != "home_win"]
    X = feat_df[FEATURE_COLS].copy()
    y = feat_df["home_win"].copy()

    # Drop rows where ALL key efficiency features are NaN
    key_feats = ["elo_diff", "net_rating_diff", "adj_off_diff"]
    mask = X[key_feats].isna().all(axis=1)
    X = X[~mask].reset_index(drop=True)
    y = y[~mask].reset_index(drop=True)
    print(f"Dropped {mask.sum()} rows missing all key features")

    medians = X.median()
    X = X.fillna(medians)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Train : {len(X_train):,}")
    print(f"Test  : {len(X_test):,}")
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
"""))

# ── §8 XGBoost Cross-Validation ───────────────────────────────────────────────
cells.append(md_cell("## §8 · XGBoost Cross-Validation"))
cells.append(code_cell("""
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

    xgb_model = xgb.XGBClassifier(
        n_estimators     = 500,
        max_depth        = 4,
        learning_rate    = 0.025,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 8,
        gamma            = 0.2,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        use_label_encoder= False,
        eval_metric      = "logloss",
        random_state     = 42,
        n_jobs           = -1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc   = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring="accuracy")
    auc   = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring="roc_auc")
    brier = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring="neg_brier_score")

    print("=== 5-Fold Cross-Validation ===")
    print(f"Accuracy : {acc.mean():.4f}  ±{acc.std():.4f}")
    print(f"ROC-AUC  : {auc.mean():.4f}  ±{auc.std():.4f}")
    print(f"Brier    : {(-brier).mean():.4f}  ±{brier.std():.4f}")
"""))

# ── §9 Calibrated Model ───────────────────────────────────────────────────────
cells.append(md_cell("## §9 · Calibrated Model (Isotonic Regression)"))
cells.append(code_cell("""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss

    xgb_model.fit(X_train, y_train)

    cal_model = CalibratedClassifierCV(xgb_model, method="isotonic", cv=5)
    cal_model.fit(X_train, y_train)

    y_proba = cal_model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    print("=== Test Set Performance ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Brier    : {brier_score_loss(y_test, y_proba):.4f}")
"""))

# ── §9a Save Assets ───────────────────────────────────────────────────────────
cells.append(md_cell("## §9a · Save Model Assets  ← run this before launching cbb_app.py"))
cells.append(code_cell("""
    # Fetch current-season data for live predictions
    print(f"Fetching {CURRENT_SEASON} adjusted ratings...")
    curr_adj_raw = fetch_adj_ratings(CURRENT_SEASON)
    curr_adj_df  = pd.DataFrame(curr_adj_raw)
    print(f"  {len(curr_adj_df)} teams")

    print(f"Fetching {CURRENT_SEASON} team stats...")
    curr_stats_raw = fetch_team_stats(CURRENT_SEASON)
    for r in curr_stats_raw:
        r["season"] = CURRENT_SEASON
    curr_stats_df = pd.DataFrame([flatten_stats(r) for r in curr_stats_raw])
    print(f"  {len(curr_stats_df)} teams")

    # Combine historical + current season
    all_adj   = pd.concat([adj_ratings, curr_adj_df], ignore_index=True)
    all_stats = pd.concat([team_stats,  curr_stats_df], ignore_index=True)

    # Latest rolling form per team (for live predictions)
    latest_form_roll   = all_rec.groupby("team")["roll_win10"].last().dropna().to_dict()
    latest_form_season = all_rec.groupby("team")["season_winpct"].last().dropna().to_dict()

    assets = {
        "model"              : cal_model,
        "feature_cols"       : FEATURE_COLS,
        "medians"            : medians.to_dict(),
        "adj_ratings"        : all_adj,
        "team_stats"         : all_stats,
        "latest_form_roll"   : latest_form_roll,
        "latest_form_season" : latest_form_season,
        "latest_season"      : max(SEASONS),
        "current_season"     : CURRENT_SEASON,
    }

    with open("cbb_model_assets.pkl", "wb") as f:
        pickle.dump(assets, f)

    print("\\n✅ cbb_model_assets.pkl saved!")
    print(f"  adj_ratings rows  : {len(all_adj)}")
    print(f"  team_stats rows   : {len(all_stats)}")
    print(f"  form teams (roll) : {len(latest_form_roll)}")
"""))

# ── §10 Feature Importance ────────────────────────────────────────────────────
cells.append(md_cell("## §10 · Feature Importance"))
cells.append(code_cell("""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fi = (pd.Series(xgb_model.feature_importances_, index=FEATURE_COLS)
          .sort_values(ascending=True))

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#e74c3c" if "elo" in i or "adj" in i or "net" in i else "#3498db" for i in fi.index]
    fi.plot.barh(ax=ax, color=colors)
    ax.set_title("XGBoost Feature Importances — CBB Win Probability Model", fontsize=13)
    ax.set_xlabel("Importance (gain)")
    ax.axvline(fi.mean(), color="gray", linestyle="--", alpha=0.6, label="Mean")
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\\nTop 5 features:")
    print(fi.sort_values(ascending=False).head(5).to_string())
"""))

# ── §11 Today's Predictions ───────────────────────────────────────────────────
cells.append(md_cell("## §11 · Today's Predictions"))
cells.append(code_cell("""
    def fetch_today():
        today = date.today().isoformat()
        r = requests.get(f"{BASE_URL}/games", headers=HEADERS, params={
            "startDateRange": f"{today}T00:00:00Z",
            "endDateRange"  : f"{today}T23:59:59Z",
        }, timeout=30)
        r.raise_for_status()
        return pd.DataFrame(r.json())

    def predict_games(sched_df, adj_df, stats_df, form_roll, form_seas):
        rows = []
        for _, g in sched_df.iterrows():
            ht = g.get("homeTeam", "")
            at = g.get("awayTeam", "")
            s  = g.get("season", CURRENT_SEASON)

            h_adj  = lookup_adj(ht, s, adj_df)
            a_adj  = lookup_adj(at, s, adj_df)
            h_stat = lookup_stats(ht, s, stats_df)
            a_stat = lookup_stats(at, s, stats_df)
            h_r10  = form_roll.get(ht, 0.5)
            a_r10  = form_roll.get(at, 0.5)
            h_swp  = form_seas.get(ht, 0.5)
            a_swp  = form_seas.get(at, 0.5)

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

        feat = pd.DataFrame(rows, columns=FEATURE_COLS).fillna(pd.Series(medians))
        probs = cal_model.predict_proba(feat.values.astype("float32"))[:, 1]

        out = sched_df[["homeTeam","awayTeam","neutralSite"]].copy().reset_index(drop=True)
        out["home_prob"]  = probs
        out["away_prob"]  = 1 - probs
        out["winner"]     = np.where(probs >= 0.5, out["homeTeam"], out["awayTeam"])
        out["confidence"] = np.maximum(probs, 1 - probs)
        return out.sort_values("confidence", ascending=False)

    sched = fetch_today()
    print(f"Games today: {len(sched)}")

    if not sched.empty and "homeTeam" in sched.columns:
        results = predict_games(sched, all_adj, all_stats,
                                latest_form_roll, latest_form_season)
        print()
        print(results[["awayTeam","homeTeam","away_prob","home_prob",
                        "winner","confidence"]].to_string(index=False))
    else:
        print("No games today — try another date or check back during the season.")
"""))

# ── §12 Calibration Curve ─────────────────────────────────────────────────────
cells.append(md_cell("## §12 · Calibration Curve"))
cells.append(code_cell("""
    from sklearn.calibration import calibration_curve

    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_pred, frac_pos, "s-b", label="CBB Model (calibrated)")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (home wins)")
    ax.set_title("Calibration Curve — CBB Win Probability")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
"""))

# ── §13 Notes ─────────────────────────────────────────────────────────────────
cells.append(md_cell("""\
## §13 · Notes

### Data Sources
| Source | Endpoint | Used For |
|--------|----------|----------|
| CBBD API `/games` | Game results + ELO | Target, ELO diff |
| CBBD API `/ratings/adjusted` | Adj. off/def efficiency | KenPom-style features |
| CBBD API `/stats/team/season` | Season stats | Four factors, pace, shooting |

### Key Features
| Feature | Description |
|---------|-------------|
| `elo_diff` | Pre-game ELO difference (home − away) — embedded in game data |
| `net_rating_diff` | Adjusted net efficiency home − away |
| `adj_off_diff` / `adj_def_diff` | Offensive / defensive efficiency diffs |
| `efg_diff` | Effective FG% diff (Dean Oliver Factor 1) |
| `to_diff` | Turnover ratio diff (Factor 2) |
| `orb_diff` | Offensive rebound % diff (Factor 3) |
| `ftr_diff` | Free throw rate diff (Factor 4) |
| `home_advantage` | 1 = home court, 0 = neutral site |
| `form_roll10_diff` | Last 10-game win% diff |

### Season Convention
The API uses `season = year the season ENDS`
e.g., `season=2024` → 2023–24 season
"""))

# ── Write notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

with open("cbb_game_winner.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("OK  cbb_game_winner.ipynb written successfully!")
print(f"    {len(cells)} cells")
