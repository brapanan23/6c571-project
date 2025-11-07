#!/usr/bin/env python3
"""
PyBaseball Data Export Script for Hitters and Pitchers
with:
- MLBAM ID–based merges for Statcast/BRef
- Bat tracking from Baseball Savant
- Baseball-Reference WAR via bwar_bat / bwar_pitch

Hitters (2023–2024):
  - BRef batting stats (mlbID)
  - Statcast exit velo / barrels
  - Statcast expected stats
  - Statcast sprint speed
  - Bat tracking (bat speed, blast rate, swing length, etc.)
  - BRef WAR (from bwar_bat)

Pitchers (2015–current year):
  - BRef pitching stats
  - Statcast pitcher EV/barrels allowed
  - Statcast pitcher expected stats against
  - BRef WAR (from bwar_pitch)

Outputs:
  - Hitters_2023-24_byYear.csv
  - Hitters_2023-24_Total.csv
  - Pitchers_2015-<current>_byYear.csv
  - Pitchers_2015-<current>_Total.csv
"""

import pandas as pd
import requests
from io import StringIO
from datetime import datetime

from pybaseball import (
    batting_stats_bref,
    pitching_stats_bref,
    bwar_bat,
    bwar_pitch,
    statcast_batter_exitvelo_barrels,
    statcast_batter_expected_stats,
    statcast_sprint_speed,
    statcast_pitcher_exitvelo_barrels,
    statcast_pitcher_expected_stats,
    cache,
)

cache.enable()

# ---------------------------------------------------------------------
# Helpers: normalize name / ID columns
# ---------------------------------------------------------------------

def normalize_name_column(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Ensure there is a 'Name' column for readability.
    Does NOT control join keys; joins are done on mlbID where possible.
    """
    df = df.copy()
    if "Name" in df.columns:
        return df

    for cand in ["last_name, first_name", "player_name", "name_common", "name"]:
        if cand in df.columns:
            df.rename(columns={cand: "Name"}, inplace=True)
            print(f"[Normalize] Renamed '{cand}' -> 'Name' for {label}")
            return df

    print(f"[Normalize] WARNING: no obvious name column for {label}. Columns: {list(df.columns)}")
    return df


def normalize_bref_ids(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Normalize BRef dataframes:
      - ensure 'mlbID' exists and is string
      - ensure 'Name' exists
    """
    df = df.copy()
    if "mlbID" not in df.columns:
        raise KeyError(f"[IDs] 'mlbID' not found in {label}. Columns: {list(df.columns)}")

    df["mlbID"] = df["mlbID"].astype(str)
    df = normalize_name_column(df, label)
    return df


def normalize_statcast_ids(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Normalize Statcast-style dataframes:
      - map 'player_id' / 'id' to 'mlbID'
      - cast to string
      - ensure 'Name' exists when possible
    """
    df = df.copy()

    id_col = None
    for cand in ["player_id", "playerid", "id", "mlbID"]:
        if cand in df.columns:
            id_col = cand
            break

    if id_col is None:
        raise KeyError(f"[IDs] No id column found in {label}. Columns: {list(df.columns)}")

    if id_col != "mlbID":
        df.rename(columns={id_col: "mlbID"}, inplace=True)
        print(f"[IDs] Renamed '{id_col}' -> 'mlbID' for {label}")

    df["mlbID"] = df["mlbID"].astype(str)
    df = normalize_name_column(df, label)
    return df


# ---------------------------------------------------------------------
# WAR helpers (bwar_bat / bwar_pitch)
# ---------------------------------------------------------------------

def get_bwar_bat_year(year: int) -> pd.DataFrame:
    """
    Get Baseball-Reference batting WAR for a given year using bwar_bat,
    and return a DataFrame with columns: ['Name', 'Season', 'WAR'].
    """
    print(f"[bWAR Bat] Fetching WAR for {year}...")
    war_df = bwar_bat(year)
    war_df = normalize_name_column(war_df, f"bWAR batting {year}")

    # Try to detect year and WAR columns
    year_col = None
    for cand in ["year_ID", "year", "Year", "season", "Season"]:
        if cand in war_df.columns:
            year_col = cand
            break
    if year_col is None:
        raise KeyError(f"[bWAR Bat] No year column found for {year}. Columns: {list(war_df.columns)}")

    war_col = None
    for cand in ["WAR", "war"]:
        if cand in war_df.columns:
            war_col = cand
            break
    if war_col is None:
        raise KeyError(f"[bWAR Bat] No WAR column found for {year}. Columns: {list(war_df.columns)}")

    # Group by Name + year (in case of multiple stints) and sum WAR
    war_group = (
        war_df.groupby(["Name", year_col])[war_col]
        .sum()
        .reset_index()
    )
    war_group.rename(columns={year_col: "Season", war_col: "WAR"}, inplace=True)

    # cast Season to int if needed
    war_group["Season"] = war_group["Season"].astype(int)
    print(f"[bWAR Bat] {year} rows after grouping: {len(war_group)}")
    return war_group


def get_bwar_pitch_year(year: int) -> pd.DataFrame:
    """
    Get Baseball-Reference pitching WAR for a given year using bwar_pitch,
    and return a DataFrame with columns: ['Name', 'Season', 'WAR'].
    """
    print(f"[bWAR Pitch] Fetching WAR for {year}...")
    war_df = bwar_pitch(year)
    war_df = normalize_name_column(war_df, f"bWAR pitching {year}")

    year_col = None
    for cand in ["year_ID", "year", "Year", "season", "Season"]:
        if cand in war_df.columns:
            year_col = cand
            break
    if year_col is None:
        raise KeyError(f"[bWAR Pitch] No year column found for {year}. Columns: {list(war_df.columns)}")

    war_col = None
    for cand in ["WAR", "war"]:
        if cand in war_df.columns:
            war_col = cand
            break
    if war_col is None:
        raise KeyError(f"[bWAR Pitch] No WAR column found for {year}. Columns: {list(war_df.columns)}")

    war_group = (
        war_df.groupby(["Name", year_col])[war_col]
        .sum()
        .reset_index()
    )
    war_group.rename(columns={year_col: "Season", war_col: "WAR"}, inplace=True)

    war_group["Season"] = war_group["Season"].astype(int)
    print(f"[bWAR Pitch] {year} rows after grouping: {len(war_group)}")
    return war_group


# ---------------------------------------------------------------------
# Bat Tracking from Baseball Savant using your URL pattern
# ---------------------------------------------------------------------

def get_bat_tracking_data(year: int, min_swings: int = 5) -> pd.DataFrame:
    """
    Fetch bat-tracking leaderboard CSV from Baseball Savant.

    Uses the parameterization like:
    https://baseballsavant.mlb.com/leaderboard/bat-tracking
      ?gameType=Regular
      &minSwings=5
      &minGroupSwings=1
      &seasonStart=YYYY
      &seasonEnd=YYYY
      &type=batter
      &csv=1
    """
    base_url = "https://baseballsavant.mlb.com/leaderboard/bat-tracking"

    params = {
        "gameType": "Regular",
        "minSwings": min_swings,
        "minGroupSwings": 1,
        "seasonStart": year,
        "seasonEnd": year,
        "type": "batter",
        "csv": 1,
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/129.0.0.0 Safari/537.36"
        )
    }

    print(f"[BatTracking] Requesting CSV for {year}...")
    resp = requests.get(base_url, params=params, headers=headers)
    print(f"[BatTracking] {year} status: {resp.status_code}")
    print(f"[BatTracking] URL: {resp.url}")
    resp.raise_for_status()

    bt = pd.read_csv(StringIO(resp.text))
    print(f"[BatTracking] {year} shape: {bt.shape}")
    print(f"[BatTracking] {year} columns (first 15): {list(bt.columns)[:15]}")

    # Map 'id' to mlbID if present; normalize names
    if "id" in bt.columns:
        bt.rename(columns={"id": "mlbID"}, inplace=True)
        bt["mlbID"] = bt["mlbID"].astype(str)
    bt = normalize_statcast_ids(bt, f"Bat Tracking {year}")
    return bt


# ---------------------------------------------------------------------
# Hitters pipeline (ID-based merges + WAR)
# ---------------------------------------------------------------------

def get_hitters_data(years):
    hitters = []
    for year in years:
        print(f"\n=== Hitters {year} ===")

        # BRef batting (base)
        bref = batting_stats_bref(year)
        bref = normalize_bref_ids(bref, f"BRef batting {year}")

        # Statcast leaderboards
        statcast_ev = statcast_batter_exitvelo_barrels(year)
        statcast_ev = normalize_statcast_ids(statcast_ev, f"Statcast EV/Barrels {year}")

        xstats = statcast_batter_expected_stats(year)
        xstats = normalize_statcast_ids(xstats, f"Statcast xStats {year}")

        sprint = statcast_sprint_speed(year)
        sprint = normalize_statcast_ids(sprint, f"Statcast Sprint {year}")

        # Bat tracking
        bat_track = get_bat_tracking_data(year)
        # already normalized in helper

        # BRef WAR (batting)
        war_bat = get_bwar_bat_year(year)  # columns: ['Name', 'Season', 'WAR']

        # Merge everything on mlbID (left join from BRef)
        df = bref.merge(bat_track, on="mlbID", how="left", suffixes=("", "_bat"))
        df = df.merge(statcast_ev, on="mlbID", how="left", suffixes=("", "_ev"))
        df = df.merge(xstats, on="mlbID", how="left", suffixes=("", "_x"))
        df = df.merge(sprint, on="mlbID", how="left", suffixes=("", "_sprint"))

        # Season info
        df["Season"] = year

        # Merge WAR on Name + Season (bWAR tables are BRef-name based)
        df = df.merge(
            war_bat[["Name", "Season", "WAR"]],
            on=["Name", "Season"],
            how="left",
            suffixes=("", "_WAR"),
        )

        # Filter hitters with > 10 AB
        if "AB" in df.columns:
            before = len(df)
            df = df[df["AB"] > 10].copy()
            print(f"[Hitters {year}] Filter AB>10: {before} -> {len(df)}")

        hitters.append(df)

    return pd.concat(hitters, ignore_index=True)


# ---------------------------------------------------------------------
# Pitchers pipeline (ID-based merges + WAR)
# ---------------------------------------------------------------------

def get_pitchers_data(years):
    pitchers = []
    for year in years:
        print(f"\n=== Pitchers {year} ===")

        bref = pitching_stats_bref(year)
        bref = normalize_bref_ids(bref, f"BRef pitching {year}")

        statcast_ev = statcast_pitcher_exitvelo_barrels(year)
        statcast_ev = normalize_statcast_ids(statcast_ev, f"Statcast Pitch EV/Barrels {year}")

        xstats = statcast_pitcher_expected_stats(year)
        xstats = normalize_statcast_ids(xstats, f"Statcast Pitch xStats {year}")

        war_pitch = get_bwar_pitch_year(year)

        df = bref.merge(statcast_ev, on="mlbID", how="left", suffixes=("", "_ev"))
        df = df.merge(xstats, on="mlbID", how="left", suffixes=("", "_x"))

        df["Season"] = year
        df = df.merge(
            war_pitch[["Name", "Season", "WAR"]],
            on=["Name", "Season"],
            how="left",
            suffixes=("", "_WAR"),
        )

        pitchers.append(df)

    return pd.concat(pitchers, ignore_index=True)


# ---------------------------------------------------------------------
# Aggregation (per player)
# ---------------------------------------------------------------------

def compute_aggregate(df: pd.DataFrame, player_id_col: str = "mlbID", war_cols=("WAR",)):
    """
    Aggregate per-player across seasons and compute WAR/162 using BRef WAR.
    Uses mlbID as the group key; Name is kept for readability.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    agg = df.groupby(player_id_col)[numeric_cols].sum().reset_index()

    # For readability, keep a representative Name (first non-null)
    if "Name" in df.columns:
        name_map = (
            df.dropna(subset=["Name"])
              .groupby(player_id_col)["Name"]
              .first()
              .reset_index()
        )
        agg = agg.merge(name_map, on=player_id_col, how="left")

    # Games & WAR/162
    if "G" in df.columns:
        games = df.groupby(player_id_col)["G"].sum().reset_index(name="Games")
        agg = agg.merge(games, on=player_id_col, how="left")

        for war_col in war_cols:
            if war_col in agg.columns:
                games_nonzero = agg["Games"].replace(0, pd.NA)
                agg[f"{war_col}_per_162"] = agg[war_col] / games_nonzero * 162

    agg["Season"] = "TOTAL"
    return agg


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    hitter_years = [2023, 2024, 2025]
    current_year = datetime.now().year
    pitcher_years = list(range(2015, current_year + 1))

    print("Fetching hitter data...")
    hitters_by_year = get_hitters_data(hitter_years)
    hitters_total = compute_aggregate(hitters_by_year, player_id_col="mlbID", war_cols=("WAR",))

    print("\nFetching pitcher data...")
    pitchers_by_year = get_pitchers_data(pitcher_years)
    pitchers_total = compute_aggregate(pitchers_by_year, player_id_col="mlbID", war_cols=("WAR",))

    print("\nSaving CSVs...")
    hitters_by_year.to_csv("Hitters_2023-24_byYear.csv", index=False)
    hitters_total.to_csv("Hitters_2023-24_Total.csv", index=False)

    pitchers_by_year.to_csv(f"Pitchers_2015-{current_year}_byYear.csv", index=False)
    pitchers_total.to_csv(f"Pitchers_2015-{current_year}_Total.csv", index=False)

    print("Done.")


if __name__ == "__main__":
    main()
