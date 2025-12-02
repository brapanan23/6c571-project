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
      - map 'player_id' / 'id' / 'mlb_ID' / 'player_ID' to 'mlbID'
      - cast to string
      - ensure 'Name' exists when possible
    """
    df = df.copy()

    id_col = None
    for cand in ["player_id", "playerid", "id", "mlbID", "mlb_ID", "player_ID"]:
        if cand in df.columns:
            id_col = cand
            break

    if id_col is None:
        raise KeyError(f"[IDs] No id column found in {label}. Columns: {list(df.columns)}")

    if id_col != "mlbID":
        df.rename(columns={id_col: "mlbID"}, inplace=True)
        print(f"[IDs] Renamed '{id_col}' -> 'mlbID' for {label}")

    # Convert to string, handling float values (remove .0 if present)
    if df["mlbID"].dtype == 'float64':
        # Convert float to int first to remove .0, then to string
        df["mlbID"] = df["mlbID"].fillna(0).astype(int).astype(str)
        # Replace "0" with NaN for missing values
        df.loc[df["mlbID"] == "0", "mlbID"] = pd.NA
    else:
        df["mlbID"] = df["mlbID"].astype(str)
    df = normalize_name_column(df, label)
    return df


# ---------------------------------------------------------------------
# WAR helpers (bwar_bat / bwar_pitch)
# ---------------------------------------------------------------------

def get_bwar_bat_year(year: int) -> pd.DataFrame:
    """
    Get Baseball-Reference batting WAR for a given year using bwar_bat,
    and return a DataFrame with columns: ['Name', 'Season', 'WAR'] (and 'mlbID' if available).
    """
    print(f"[bWAR Bat] Fetching WAR for {year}...")
    war_df = bwar_bat(year)
    war_df = normalize_name_column(war_df, f"bWAR batting {year}")
    
    # Try to normalize mlbID if it exists
    if "mlbID" not in war_df.columns:
        war_df = normalize_statcast_ids(war_df, f"bWAR batting {year}")

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

    # Group by Name and year (and mlbID if available)
    group_cols = ["Name", year_col]
    if "mlbID" in war_df.columns:
        group_cols.append("mlbID")
    
    war_group = (
        war_df.groupby(group_cols)[war_col]
        .sum()
        .reset_index()
    )
    war_group.rename(columns={year_col: "Season", war_col: "WAR"}, inplace=True)

    war_group["Season"] = war_group["Season"].astype(int)
    print(f"[bWAR Bat] {year} rows after grouping: {len(war_group)}")
    if "mlbID" in war_group.columns:
        print(f"[bWAR Bat] mlbID available for merging")
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

def get_hitters_data(years, min_ab: int = 10, min_swings: int = 5):
    """
    Get hitters data for specified years.
    
    Parameters:
    -----------
    years : list
        List of years to fetch data for
    min_ab : int, default=10
        Minimum at-bats required to include a player (set to 0 to include all players)
    min_swings : int, default=5
        Minimum swings required for bat tracking data
    """
    hitters = []
    for year in years:
        print(f"\n=== Hitters {year} ===")

        bref = batting_stats_bref(year)
        bref = normalize_bref_ids(bref, f"BRef batting {year}")

        statcast_ev = statcast_batter_exitvelo_barrels(year)
        statcast_ev = normalize_statcast_ids(statcast_ev, f"Statcast EV/Barrels {year}")

        xstats = statcast_batter_expected_stats(year)
        xstats = normalize_statcast_ids(xstats, f"Statcast xStats {year}")

        sprint = statcast_sprint_speed(year)
        sprint = normalize_statcast_ids(sprint, f"Statcast Sprint {year}")

        bat_track = get_bat_tracking_data(year, min_swings=min_swings)

        war_bat = get_bwar_bat_year(year)  # columns: ['Name', 'Season', 'WAR']

        df = bref.merge(bat_track, on="mlbID", how="left", suffixes=("", "_bat"))
        df = df.merge(statcast_ev, on="mlbID", how="left", suffixes=("", "_ev"))
        df = df.merge(xstats, on="mlbID", how="left", suffixes=("", "_x"))
        df = df.merge(sprint, on="mlbID", how="left", suffixes=("", "_sprint"))

        df["Season"] = int(year)  # Ensure Season is int to match WAR data

        # Try to merge WAR on mlbID first (more reliable), fall back to Name if needed
        if "mlbID" in war_bat.columns:
            # Ensure mlbID types match (both should be strings)
            df["mlbID"] = df["mlbID"].astype(str)
            war_bat["mlbID"] = war_bat["mlbID"].astype(str)
            
            # Merge on mlbID and Season (more reliable than Name matching)
            df = df.merge(
                war_bat[["mlbID", "Season", "WAR"]],
                on=["mlbID", "Season"],
                how="left",
                suffixes=("", "_WAR"),
            )
            print(f"[Hitters {year}] WAR merged on mlbID: {df['WAR'].notna().sum()} / {len(df)} rows have WAR")
        else:
            # Fall back to Name merge if mlbID not available (may have encoding issues)
            df = df.merge(
                war_bat[["Name", "Season", "WAR"]],
                on=["Name", "Season"],
                how="left",
                suffixes=("", "_WAR"),
            )
            print(f"[Hitters {year}] WAR merged on Name (mlbID not available - may have encoding issues): {df['WAR'].notna().sum()} / {len(df)} rows have WAR")

        if "AB" in df.columns and min_ab > 0:
            before = len(df)
            df = df[df["AB"] > min_ab].copy()
            print(f"[Hitters {year}] Filter AB>{min_ab}: {before} -> {len(df)}")
        elif min_ab == 0:
            print(f"[Hitters {year}] No AB filter applied (min_ab=0)")

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
    
    Properly handles:
    - Summing counting stats (G, PA, AB, H, HR, etc.)
    - Recalculating rate stats from totals (BA, OBP, SLG, OPS)
    - Weighted averaging for average stats (avg_bat_speed, sprint_speed, etc.)
    - Max for maximum stats (max_hit_speed, max_distance)
    """
    # Columns to SUM (counting stats)
    sum_cols = [
        "G", "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "IBB", "SO", 
        "HBP", "SH", "SF", "GDP", "SB", "CS",
        "swings_competitive", "contact", "whiffs", "batted_ball_events", 
        "attempts", "ev50", "fbld", "gb", "ev95plus", "barrels",
        "batter_run_value", "competitive_runs", "bolts",
        "swords",  # cumulative
        "WAR",  # WAR should be summed across seasons
    ]
    
    # Columns to RECALCULATE from totals (rate stats)
    recalc_cols = [
        "BA", "OBP", "SLG", "OPS",
        "brl_percent", "brl_pa",
        "pa", "bip", "ba", "est_ba", "est_ba_minus_ba_diff",
        "slg", "est_slg", "est_slg_minus_slg_diff",
        "woba", "est_woba", "est_woba_minus_woba_diff",
    ]
    
    # Columns to AVERAGE (weighted where appropriate)
    avg_cols = [
        "avg_bat_speed", "hard_swing_rate", "squared_up_per_bat_contact",
        "squared_up_per_swing", "blast_per_bat_contact", "blast_per_swing",
        "swing_length", "whiff_per_swing", "batted_ball_event_per_swing",
        "avg_hit_angle", "anglesweetspotpercent", "avg_hit_speed",
        "avg_distance", "avg_hr_distance", "ev95percent",
        "sprint_speed", "hp_to_1b",
        "percent_swings_competitive",
    ]
    
    # Columns to take MAX
    max_cols = ["max_hit_speed", "max_distance"]
    
    # Columns to take FIRST or AVERAGE
    first_cols = ["Age", "age", "#days"]  # Take first or average
    
    # Filter to columns that actually exist in the dataframe
    available_sum = [c for c in sum_cols if c in df.columns]
    available_recalc = [c for c in recalc_cols if c in df.columns]
    available_avg = [c for c in avg_cols if c in df.columns]
    available_max = [c for c in max_cols if c in df.columns]
    available_first = [c for c in first_cols if c in df.columns]
    
    # Start with summing counting stats
    agg_dict = {}
    if available_sum:
        sum_agg = df.groupby(player_id_col)[available_sum].sum()
        for col in available_sum:
            agg_dict[col] = sum_agg[col]
    
    # Take max for maximum stats
    if available_max:
        max_agg = df.groupby(player_id_col)[available_max].max()
        for col in available_max:
            agg_dict[col] = max_agg[col]
    
    # Weighted averages for average stats
    if available_avg:
        for col in available_avg:
            if col == "percent_swings_competitive":
                # Will recalculate below from swings_competitive / PA
                continue
            elif col == "avg_bat_speed" and "contact" in df.columns:
                # Weight by contact
                def weighted_avg_contact(group):
                    mask = group[col].notna() & group["contact"].notna()
                    if mask.sum() == 0:
                        return pd.NA
                    numerator = (group.loc[mask, col] * group.loc[mask, "contact"]).sum()
                    denominator = group.loc[mask, "contact"].sum()
                    return numerator / denominator if denominator > 0 else pd.NA
                weighted = df.groupby(player_id_col).apply(weighted_avg_contact)
            elif col in ["hard_swing_rate", "squared_up_per_swing", "blast_per_swing", 
                        "whiff_per_swing", "batted_ball_event_per_swing"] and "swings_competitive" in df.columns:
                # Weight by swings_competitive
                def weighted_avg_swings(group):
                    mask = group[col].notna() & group["swings_competitive"].notna()
                    if mask.sum() == 0:
                        return pd.NA
                    numerator = (group.loc[mask, col] * group.loc[mask, "swings_competitive"]).sum()
                    denominator = group.loc[mask, "swings_competitive"].sum()
                    return numerator / denominator if denominator > 0 else pd.NA
                weighted = df.groupby(player_id_col).apply(weighted_avg_swings)
            elif col in ["squared_up_per_bat_contact", "blast_per_bat_contact"] and "contact" in df.columns:
                # Weight by contact
                def weighted_avg_contact(group):
                    mask = group[col].notna() & group["contact"].notna()
                    if mask.sum() == 0:
                        return pd.NA
                    numerator = (group.loc[mask, col] * group.loc[mask, "contact"]).sum()
                    denominator = group.loc[mask, "contact"].sum()
                    return numerator / denominator if denominator > 0 else pd.NA
                weighted = df.groupby(player_id_col).apply(weighted_avg_contact)
            elif col == "swing_length" and "swings_competitive" in df.columns:
                # Weight by swings_competitive
                def weighted_avg_swings(group):
                    mask = group[col].notna() & group["swings_competitive"].notna()
                    if mask.sum() == 0:
                        return pd.NA
                    numerator = (group.loc[mask, col] * group.loc[mask, "swings_competitive"]).sum()
                    denominator = group.loc[mask, "swings_competitive"].sum()
                    return numerator / denominator if denominator > 0 else pd.NA
                weighted = df.groupby(player_id_col).apply(weighted_avg_swings)
            elif col in ["avg_hit_angle", "anglesweetspotpercent", "avg_hit_speed", 
                        "avg_distance", "ev95percent"] and "batted_ball_events" in df.columns:
                # Weight by batted_ball_events
                def weighted_avg_bbe(group):
                    mask = group[col].notna() & group["batted_ball_events"].notna()
                    if mask.sum() == 0:
                        return pd.NA
                    numerator = (group.loc[mask, col] * group.loc[mask, "batted_ball_events"]).sum()
                    denominator = group.loc[mask, "batted_ball_events"].sum()
                    return numerator / denominator if denominator > 0 else pd.NA
                weighted = df.groupby(player_id_col).apply(weighted_avg_bbe)
            elif col == "avg_hr_distance" and "HR" in df.columns:
                # Weight by HR
                def weighted_avg_hr(group):
                    mask = group[col].notna() & group["HR"].notna()
                    if mask.sum() == 0:
                        return pd.NA
                    numerator = (group.loc[mask, col] * group.loc[mask, "HR"]).sum()
                    denominator = group.loc[mask, "HR"].sum()
                    return numerator / denominator if denominator > 0 else pd.NA
                weighted = df.groupby(player_id_col).apply(weighted_avg_hr)
            elif col in ["sprint_speed", "hp_to_1b"] and "attempts" in df.columns:
                # Weight by attempts
                def weighted_avg_attempts(group):
                    mask = group[col].notna() & group["attempts"].notna()
                    if mask.sum() == 0:
                        return pd.NA
                    numerator = (group.loc[mask, col] * group.loc[mask, "attempts"]).sum()
                    denominator = group.loc[mask, "attempts"].sum()
                    return numerator / denominator if denominator > 0 else pd.NA
                weighted = df.groupby(player_id_col).apply(weighted_avg_attempts)
            else:
                # Simple average (ignoring NaN)
                weighted = df.groupby(player_id_col)[col].mean()
            
            agg_dict[col] = weighted
    
    # Take first or average for age/days
    if available_first:
        for col in available_first:
            if col in ["Age", "age"]:
                # Take most recent (last) age
                agg_dict[col] = df.groupby(player_id_col)[col].last()
            else:
                # Average for #days
                agg_dict[col] = df.groupby(player_id_col)[col].mean()
    
    # Create aggregated dataframe
    agg = pd.DataFrame(agg_dict).reset_index()
    
    # Recalculate rate stats from totals
    if "H" in agg.columns and "AB" in agg.columns:
        agg["BA"] = agg["H"] / agg["AB"].replace(0, pd.NA)
    
    if "H" in agg.columns and "BB" in agg.columns and "HBP" in agg.columns and "AB" in agg.columns and "SF" in agg.columns:
        agg["OBP"] = (agg["H"] + agg["BB"] + agg["HBP"]) / (agg["AB"] + agg["BB"] + agg["HBP"] + agg["SF"]).replace(0, pd.NA)
    
    if "H" in agg.columns and "2B" in agg.columns and "3B" in agg.columns and "HR" in agg.columns and "AB" in agg.columns:
        # SLG = (1B + 2*2B + 3*3B + 4*HR) / AB
        singles = agg["H"] - agg["2B"] - agg["3B"] - agg["HR"]
        agg["SLG"] = (singles + 2*agg["2B"] + 3*agg["3B"] + 4*agg["HR"]) / agg["AB"].replace(0, pd.NA)
    
    if "OBP" in agg.columns and "SLG" in agg.columns:
        agg["OPS"] = agg["OBP"] + agg["SLG"]
    
    if "barrels" in agg.columns and "batted_ball_events" in agg.columns:
        agg["brl_percent"] = agg["barrels"] / agg["batted_ball_events"].replace(0, pd.NA)
    
    if "barrels" in agg.columns and "PA" in agg.columns:
        agg["brl_pa"] = agg["barrels"] / agg["PA"].replace(0, pd.NA)
    
    if "swings_competitive" in agg.columns and "PA" in agg.columns:
        agg["percent_swings_competitive"] = agg["swings_competitive"] / agg["PA"].replace(0, pd.NA)
    
    # Recalculate expected stats if we have the components
    if "bip" in df.columns:
        # Recalculate bip from batted_ball_events if available
        if "batted_ball_events" in agg.columns:
            agg["bip"] = agg["batted_ball_events"]
        elif "AB" in agg.columns:
            agg["bip"] = agg["AB"] - agg.get("SO", 0)  # Approximate
    
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
    if "G" in agg.columns:
        agg["Games"] = agg["G"]
        
        for war_col in war_cols:
            if war_col in agg.columns:
                games_nonzero = agg["Games"].replace(0, pd.NA)
                agg[f"{war_col}_per_162"] = agg[war_col] / games_nonzero * 162
    
    agg["Season"] = "TOTAL"
    return agg


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(min_ab: int = 10, min_swings: int = 5):
    """
    Main function to fetch and aggregate hitter and pitcher data.
    
    Parameters:
    -----------
    min_ab : int, default=10
        Minimum at-bats required to include a player (set to 0 to include all players)
    min_swings : int, default=5
        Minimum swings required for bat tracking data
    """
    hitter_years = [2023, 2024, 2025]
    current_year = datetime.now().year
    pitcher_years = list(range(2015, current_year + 1))

    print("Fetching hitter data...")
    print(f"Using thresholds: min_ab={min_ab}, min_swings={min_swings}")
    hitters_by_year = get_hitters_data(hitter_years, min_ab=min_ab, min_swings=min_swings)
    hitters_total = compute_aggregate(hitters_by_year, player_id_col="mlbID", war_cols=("WAR",))

    print("\nFetching pitcher data...")
    pitchers_by_year = get_pitchers_data(pitcher_years)
    pitchers_total = compute_aggregate(pitchers_by_year, player_id_col="mlbID", war_cols=("WAR",))

    print("\nSaving CSVs...")
    hitters_by_year.to_csv("data/Hitters_2023-24_byYear_retry.csv", index=False)
    hitters_total.to_csv("data/Hitters_2023-24_Total_retry.csv", index=False)

    pitchers_by_year.to_csv(f"data/Pitchers_2015-{current_year}_byYear_retry.csv", index=False)
    pitchers_total.to_csv(f"data/Pitchers_2015-{current_year}_Total_retry.csv", index=False)

    print("Done.")


if __name__ == "__main__":
    main()
