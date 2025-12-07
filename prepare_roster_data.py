#!/usr/bin/env python3
"""
Prepare player data for roster optimization.

This script:
1. Loads trained WAR prediction and salary models
2. Uses 2024 season data to predict 2025 xWAR for all players
3. Uses 2024 stats to predict expected salary/cost
4. Outputs players.csv for use with the Julia roster optimizer

Run this BEFORE running roster_optimizer.ipynb

Requirements:
- Run hitter_war_t_to_tp1_multiyear_backfill.ipynb first (saves models/hitter_war_model_newest.joblib)
- Run pitcher_war_t_to_tp1_multiyear_optimized.ipynb first (saves models/pitcher_war_model_newest.joblib)
- Run aav_regression.ipynb first (saves models/hitter_salary_model.joblib, models/pitcher_salary_model.joblib)
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_models():
    """Load all trained models from the models directory."""
    models = {}
    
    # Load hitter WAR model
    if os.path.exists('models/hitter_war_model_newest.joblib'):
        models['hitter_war'] = joblib.load('models/hitter_war_model_newest.joblib')
        # Check if it has feature_names (new format) or feature_cols (old format)
        if 'feature_names' in models['hitter_war']:
            num_features = len(models['hitter_war']['feature_names'])
        else:
            num_features = len(models['hitter_war'].get('feature_cols', []))
        print(f"✓ Loaded hitter WAR model ({num_features} features)")
    else:
        print("✗ Missing hitter WAR model - run hitter_war_t_to_tp1_multiyear_backfill.ipynb first")
    
    # Load pitcher WAR model
    if os.path.exists('models/pitcher_war_model_newest.joblib'):
        models['pitcher_war'] = joblib.load('models/pitcher_war_model_newest.joblib')
        num_features = len(models['pitcher_war'].get('feature_cols', models['pitcher_war'].get('feature_names', [])))
        print(f"✓ Loaded pitcher WAR model ({num_features} features)")
    else:
        print("✗ Missing pitcher WAR model - run pitcher_war_t_to_tp1_multiyear_optimized.ipynb first")
    
    # Load hitter salary model
    if os.path.exists('models/hitter_salary_model.joblib'):
        models['hitter_salary'] = joblib.load('models/hitter_salary_model.joblib')
        print(f"✓ Loaded hitter salary model (features: {models['hitter_salary']['features']})")
    else:
        print("✗ Missing hitter salary model - run aav_regression.ipynb first")
    
    # Load pitcher salary model
    if os.path.exists('models/pitcher_salary_model.joblib'):
        models['pitcher_salary'] = joblib.load('models/pitcher_salary_model.joblib')
        print(f"✓ Loaded pitcher salary model (features: {models['pitcher_salary']['features']})")
    else:
        print("✗ Missing pitcher salary model - run aav_regression.ipynb first")
    
    return models


def extract_hitter_features(row, war_model):
    """
    Extract and prepare features for hitter WAR prediction.
    Handles position one-hot encoding as done in the training notebook.
    """
    # Get feature names (includes both numeric and position one-hot features)
    all_feature_names = war_model.get('feature_names', war_model.get('feature_cols', []))
    
    # Separate numeric features (those that don't start with 'pos_')
    numeric_feature_names = [f for f in all_feature_names if not f.startswith('pos_')]
    
    # Extract numeric features from row
    row_dict = row.to_dict()
    X_numeric = np.array([[row_dict.get(col, np.nan) for col in numeric_feature_names]])
    
    # Handle position one-hot encoding
    if 'onehot_encoder' in war_model:
        # Get position value (try 'position' or 'primary_pos')
        position = row.get('position', row.get('primary_pos', 'DH'))
        if pd.isna(position):
            position = 'DH'
        
        # One-hot encode position
        position_encoded = war_model['onehot_encoder'].transform([[str(position)]])
        
        # Concatenate numeric and position features
        X = np.hstack([X_numeric, position_encoded])
    else:
        # No position encoding (old model format)
        X = X_numeric
    
    return X


def prepare_hitter_data(models):
    """Load 2024 hitter data and predict 2025 xWAR and expected cost."""
    print("\n--- Processing Hitters ---")
    
    # Load hitter data
    hitters_df = pd.read_csv('data/Hitters_2015-2025_byYear_retry.csv')
    hitters_2024 = hitters_df[hitters_df['Season'] == 2024].copy()
    print(f"Loaded {len(hitters_2024)} hitters from 2024 season")
    
    # Compute WAR_per_162 for current season
    hitters_2024['WAR_per_162'] = (hitters_2024['WAR'] / hitters_2024['G'].replace(0, np.nan)) * 162
    
    # Get position from the data (use 'position' column if available)
    if 'position' not in hitters_2024.columns and 'primary_pos' in hitters_2024.columns:
        hitters_2024['position'] = hitters_2024['primary_pos']
    elif 'position' not in hitters_2024.columns:
        hitters_2024['position'] = 'DH'  # Default
    
    # Get WAR model artifacts
    war_model = models.get('hitter_war')
    salary_model = models.get('hitter_salary')
    
    results = []
    
    for idx, row in hitters_2024.iterrows():
        player_data = {
            'player_id': row['mlbID'],
            'name': row['Name'],
            'position': row.get('position', row.get('primary_pos', 'OF')),
            'age': row['Age'],
            'team': row['Tm'],
            'war_2024': row['WAR'],
            'war_per_162_2024': row['WAR_per_162'],
            'g_2024': row['G'],
            'pa_2024': row['PA'],
        }
        
        # Predict xWAR for 2025 using WAR model
        if war_model:
            try:
                # Extract features (handles position one-hot encoding)
                X = extract_hitter_features(row, war_model)
                
                # Impute and scale
                X_imp = war_model['imputer'].transform(X)
                X_scaled = war_model['scaler'].transform(X_imp)
                
                # Predict
                xwar = war_model['model'].predict(X_scaled)[0]
                player_data['xwar'] = np.clip(xwar, -3, 10)
            except Exception as e:
                print(f"Warning: Error predicting WAR for {row['Name']}: {e}")
                # Fallback: use current WAR regressed to mean
                xwar = row['WAR_per_162'] * 0.7 if pd.notna(row['WAR_per_162']) else 0
                player_data['xwar'] = np.clip(xwar, -3, 10)
        else:
            # Fallback: use current WAR regressed to mean (only for players with meaningful PA)
            if row['PA'] >= 100:
                xwar = row['WAR_per_162'] * 0.7 if pd.notna(row['WAR_per_162']) else 0
                player_data['xwar'] = np.clip(xwar, -3, 10)
            else:
                # Low PA players get lower projections
                player_data['xwar'] = row['WAR'] * 0.5 if pd.notna(row['WAR']) else 0
        
        # Predict expected cost using salary model
        # Note: The model was trained on UNSCALED data, so don't scale here
        if salary_model:
            try:
                features = salary_model['features']
                X = pd.DataFrame([row])[features]
                log_salary = salary_model['model'].predict(X)[0]
                cost = np.exp(log_salary) / 1e6  # Convert to millions
                player_data['cost'] = cost
            except Exception as e:
                # Fallback: minimum salary
                player_data['cost'] = 0.75  # $750K
        else:
            # Fallback: minimum salary
            player_data['cost'] = 0.75
        
        results.append(player_data)
    
    hitters_result = pd.DataFrame(results)
    print(f"Processed {len(hitters_result)} hitters with xWAR predictions")
    print(f"  xWAR range: {hitters_result['xwar'].min():.2f} to {hitters_result['xwar'].max():.2f}")
    print(f"  Cost range: ${hitters_result['cost'].min():.2f}M to ${hitters_result['cost'].max():.2f}M")
    
    return hitters_result


def prepare_pitcher_data(models):
    """Load 2024 pitcher data and predict 2025 xWAR and expected cost."""
    print("\n--- Processing Pitchers ---")
    
    # Load pitcher data
    pitchers_df = pd.read_csv('data/Pitchers_2015-2025_byYear_retry.csv')
    pitchers_2024 = pitchers_df[pitchers_df['Season'] == 2024].copy()
    print(f"Loaded {len(pitchers_2024)} pitchers from 2024 season")
    
    # Compute WAR_per_162 for current season
    pitchers_2024['WAR_per_162'] = (pitchers_2024['WAR'] / pitchers_2024['G'].replace(0, np.nan)) * 162
    
    # Get model artifacts
    war_model = models.get('pitcher_war')
    salary_model = models.get('pitcher_salary')
    
    # Filter for meaningful workload (matching training criteria)
    min_ip = war_model.get('min_ip', 30.0) if war_model else 30.0
    
    results = []
    
    for idx, row in pitchers_2024.iterrows():
        # Determine position based on games started
        if pd.notna(row['GS']) and row['GS'] > 0:
            gs_ratio = row['GS'] / row['G'] if row['G'] > 0 else 0
            position = 'SP' if gs_ratio > 0.5 else 'RP'
        else:
            position = 'RP'
        
        player_data = {
            'player_id': row['mlbID'],
            'name': row['Name'],
            'position': position,
            'age': row['Age'],
            'team': row['Tm'],
            'war_2024': row['WAR'],
            'war_per_162_2024': row['WAR_per_162'],
            'g_2024': row['G'],
            'ip_2024': row['IP'],
        }
        
        # Predict xWAR for 2025 using WAR model
        if war_model and row['IP'] >= min_ip:
            try:
                # Get feature columns (pitcher model uses feature_cols, not feature_names)
                feature_cols = war_model.get('feature_cols', war_model.get('feature_names', []))
                
                # Add IP_next as a feature if required (use current IP as estimate)
                row_copy = row.copy()
                if 'IP_next' in feature_cols:
                    row_copy['IP_next'] = row['IP']
                
                # Extract features
                X = pd.DataFrame([row_copy])[feature_cols].values
                
                # Impute and scale
                X_imp = war_model['imputer'].transform(X)
                X_scaled = war_model['scaler'].transform(X_imp)
                
                # Predict
                xwar = war_model['model'].predict(X_scaled)[0]
                player_data['xwar'] = np.clip(xwar, -3, 10)
            except Exception as e:
                print(f"Warning: Error predicting WAR for {row['Name']}: {e}")
                # Fallback: use current WAR regressed to mean
                xwar = row['WAR_per_162'] * 0.6 if pd.notna(row['WAR_per_162']) else 0
                player_data['xwar'] = np.clip(xwar, -3, 10)
        else:
            # Fallback for low-IP pitchers: use actual WAR (not per-162) regressed
            if row['IP'] >= 20:
                xwar = row['WAR_per_162'] * 0.5 if pd.notna(row['WAR_per_162']) else 0
                player_data['xwar'] = np.clip(xwar, -2, 8)
            else:
                # Very low IP pitchers get minimal projection
                player_data['xwar'] = row['WAR'] * 0.3 if pd.notna(row['WAR']) else 0
        
        # Predict expected cost using salary model
        # Note: The model was trained on UNSCALED data, so don't scale here
        if salary_model:
            try:
                features = salary_model['features']
                X = pd.DataFrame([row])[features]
                log_salary = salary_model['model'].predict(X)[0]
                cost = np.exp(log_salary) / 1e6  # Convert to millions
                player_data['cost'] = cost
            except Exception as e:
                # Fallback: minimum salary
                player_data['cost'] = 0.75
        else:
            # Fallback: minimum salary
            player_data['cost'] = 0.75
        
        results.append(player_data)
    
    pitchers_result = pd.DataFrame(results)
    print(f"Processed {len(pitchers_result)} pitchers with xWAR predictions")
    print(f"  xWAR range: {pitchers_result['xwar'].min():.2f} to {pitchers_result['xwar'].max():.2f}")
    print(f"  Cost range: ${pitchers_result['cost'].min():.2f}M to ${pitchers_result['cost'].max():.2f}M")
    
    return pitchers_result


def assign_hitter_positions(hitters_df):
    """Assign positions to hitters based on available data."""
    # Try to load training data with positions
    try:
        training_data = pd.read_csv('data/Hitters_Training_Data.csv')
        if 'Position' in training_data.columns:
            pos_map = training_data.groupby('mlbID')['Position'].first().to_dict()
            hitters_df['position'] = hitters_df['player_id'].map(pos_map)
    except:
        pass
    
    # Fill missing positions with reasonable defaults
    hitters_df['position'] = hitters_df['position'].fillna('OF')
    
    # Map positions to standard categories
    pos_mapping = {
        'C': 'C', 'CA': 'C', '2': 'C',
        '1B': '1B', 'B1': '1B', '3': '1B',
        '2B': '2B', 'B2': '2B', '4': '2B',
        '3B': '3B', 'B3': '3B', '5': '3B',
        'SS': 'SS', '6': 'SS',
        'LF': 'LF', '7': 'LF',
        'CF': 'CF', '8': 'CF',
        'RF': 'RF', '9': 'RF',
        'DH': 'DH', 'D': 'DH', '10': 'DH',
        'OF': 'OF', 'O': 'OF',
    }
    
    hitters_df['position'] = hitters_df['position'].map(
        lambda x: pos_mapping.get(str(x).upper(), 'OF')
    )
    
    # Distribute OF players across LF, CF, RF
    of_players = hitters_df[hitters_df['position'] == 'OF'].index
    for i, idx in enumerate(of_players):
        positions = ['LF', 'CF', 'RF']
        hitters_df.loc[idx, 'position'] = positions[i % 3]
    
    return hitters_df


def load_actual_war_2025():
    """
    Load actual 2025 WAR/162 data for comparison with predictions.
    Returns a DataFrame with player_id and actual_war_2025.
    """
    print("\n--- Loading Actual 2025 WAR Data ---")
    
    actual_war_data = []
    
    # Load hitter 2025 data
    try:
        hitters_df = pd.read_csv('data/Hitters_2015-2025_byYear_retry.csv')
        hitters_2025 = hitters_df[hitters_df['Season'] == 2025].copy()
        if len(hitters_2025) > 0:
            # Calculate WAR_per_162 for 2025
            hitters_2025['WAR_per_162'] = (hitters_2025['WAR'] / hitters_2025['G'].replace(0, np.nan)) * 162
            hitter_war = hitters_2025[['mlbID', 'WAR', 'WAR_per_162', 'G']].copy()
            hitter_war = hitter_war.rename(columns={
                'mlbID': 'player_id',
                'WAR': 'war_2025',
                'WAR_per_162': 'actual_war_2025',
                'G': 'g_2025'
            })
            actual_war_data.append(hitter_war)
            print(f"  Loaded {len(hitter_war)} hitters with 2025 data")
    except Exception as e:
        print(f"  Warning: Could not load hitter 2025 data: {e}")
    
    # Load pitcher 2025 data
    try:
        pitchers_df = pd.read_csv('data/Pitchers_2015-2025_byYear_retry.csv')
        pitchers_2025 = pitchers_df[pitchers_df['Season'] == 2025].copy()
        if len(pitchers_2025) > 0:
            # Calculate WAR_per_162 for 2025
            pitchers_2025['WAR_per_162'] = (pitchers_2025['WAR'] / pitchers_2025['G'].replace(0, np.nan)) * 162
            pitcher_war = pitchers_2025[['mlbID', 'WAR', 'WAR_per_162', 'G']].copy()
            pitcher_war = pitcher_war.rename(columns={
                'mlbID': 'player_id',
                'WAR': 'war_2025',
                'WAR_per_162': 'actual_war_2025',
                'G': 'g_2025'
            })
            actual_war_data.append(pitcher_war)
            print(f"  Loaded {len(pitcher_war)} pitchers with 2025 data")
    except Exception as e:
        print(f"  Warning: Could not load pitcher 2025 data: {e}")
    
    if actual_war_data:
        all_actual_war = pd.concat(actual_war_data, ignore_index=True)
        # If a player appears in both (unlikely but possible), take the first
        all_actual_war = all_actual_war.drop_duplicates(subset=['player_id'], keep='first')
        print(f"  Total unique players with 2025 actual WAR: {len(all_actual_war)}")
        return all_actual_war[['player_id', 'actual_war_2025', 'war_2025', 'g_2025']]
    else:
        print("  No 2025 actual WAR data loaded")
        return pd.DataFrame(columns=['player_id', 'actual_war_2025', 'war_2025', 'g_2025'])


def get_team_abbrev_to_canonical():
    """
    Map team abbreviations (like NYM, NYY, CHC, CHW) to full canonical team names.
    """
    abbrev_mapping = {
        'ARI': 'Arizona Diamondbacks',
        'AZ': 'Arizona Diamondbacks',
        'ATL': 'Atlanta Braves',
        'BAL': 'Baltimore Orioles',
        'BOS': 'Boston Red Sox',
        'CHC': 'Chicago Cubs',
        'CHW': 'Chicago White Sox',
        'CWS': 'Chicago White Sox',
        'CIN': 'Cincinnati Reds',
        'CLE': 'Cleveland Guardians',
        'COL': 'Colorado Rockies',
        'DET': 'Detroit Tigers',
        'HOU': 'Houston Astros',
        'KCR': 'Kansas City Royals',
        'KC': 'Kansas City Royals',
        'LAA': 'Los Angeles Angels',
        'LAD': 'Los Angeles Dodgers',
        'MIA': 'Miami Marlins',
        'MIL': 'Milwaukee Brewers',
        'MIN': 'Minnesota Twins',
        'NYM': 'New York Mets',
        'NYY': 'New York Yankees',
        'OAK': 'Oakland Athletics',
        'PHI': 'Philadelphia Phillies',
        'PIT': 'Pittsburgh Pirates',
        'SDP': 'San Diego Padres',
        'SD': 'San Diego Padres',
        'SFG': 'San Francisco Giants',
        'SF': 'San Francisco Giants',
        'SEA': 'Seattle Mariners',
        'STL': 'St. Louis Cardinals',
        'TBR': 'Tampa Bay Rays',
        'TB': 'Tampa Bay Rays',
        'TEX': 'Texas Rangers',
        'TOR': 'Toronto Blue Jays',
        'WSN': 'Washington Nationals',
        'WSH': 'Washington Nationals',
        'WAS': 'Washington Nationals',
    }
    return abbrev_mapping


def get_team_name_to_canonical():
    """
    Map full team names (like "New York", "Chicago") to canonical names.
    For ambiguous cases, we'll resolve using the 'team' abbreviation from performance data.
    """
    name_mapping = {
        'Arizona': 'Arizona Diamondbacks',
        'Atlanta': 'Atlanta Braves',
        'Baltimore': 'Baltimore Orioles',
        'Boston': 'Boston Red Sox',
        'Cincinnati': 'Cincinnati Reds',
        'Cleveland': 'Cleveland Guardians',
        'Colorado': 'Colorado Rockies',
        'Detroit': 'Detroit Tigers',
        'Houston': 'Houston Astros',
        'Kansas City': 'Kansas City Royals',
        'Miami': 'Miami Marlins',
        'Milwaukee': 'Milwaukee Brewers',
        'Minnesota': 'Minnesota Twins',
        'Oakland': 'Oakland Athletics',
        'Philadelphia': 'Philadelphia Phillies',
        'Pittsburgh': 'Pittsburgh Pirates',
        'San Diego': 'San Diego Padres',
        'San Francisco': 'San Francisco Giants',
        'Seattle': 'Seattle Mariners',
        'St. Louis': 'St. Louis Cardinals',
        'Tampa Bay': 'Tampa Bay Rays',
        'Texas': 'Texas Rangers',
        'Toronto': 'Toronto Blue Jays',
        'Washington': 'Washington Nationals',
        'Athletics': 'Oakland Athletics',
    }
    return name_mapping


def normalize_team_name(team_str, team_abbrev_mapping=None, team_name_mapping=None, team_abbrev=None):
    """
    Convert team name(s) to full canonical team name.
    Handles multi-team strings (comma-separated) by taking the first team.
    For ambiguous cases like "New York" or "Chicago", uses team_abbrev if provided.
    
    Args:
        team_str: Team name string from contract data
        team_abbrev_mapping: Dict mapping abbreviations (NYM, NYY, etc.) to canonical names
        team_name_mapping: Dict mapping full names to canonical names
        team_abbrev: Team abbreviation from performance data (e.g., 'NYM', 'NYY') to resolve ambiguity
    """
    if pd.isna(team_str) or team_str == '':
        return None
    
    # Handle multi-team cases (take first team)
    if ',' in str(team_str):
        team_str = str(team_str).split(',')[0].strip()
    
    team_str_lower = str(team_str).lower()
    
    # First, try to use team abbreviation if provided (most reliable for ambiguous cases)
    if team_abbrev and team_abbrev_mapping:
        abbrev_upper = str(team_abbrev).upper().strip()
        if abbrev_upper in team_abbrev_mapping:
            return team_abbrev_mapping[abbrev_upper]
    
    # Check for explicit mentions in the string
    if 'mets' in team_str_lower or 'nym' in team_str_lower:
        return 'New York Mets'
    if 'yankees' in team_str_lower or 'nyy' in team_str_lower:
        return 'New York Yankees'
    if 'angels' in team_str_lower or 'laa' in team_str_lower:
        return 'Los Angeles Angels'
    if 'dodgers' in team_str_lower or 'lad' in team_str_lower:
        return 'Los Angeles Dodgers'
    if 'white sox' in team_str_lower or 'chw' in team_str_lower or 'cws' in team_str_lower:
        return 'Chicago White Sox'
    if 'cubs' in team_str_lower or 'chc' in team_str_lower:
        return 'Chicago Cubs'
    
    # Try direct mapping from name_mapping
    if team_name_mapping and team_str in team_name_mapping:
        return team_name_mapping[team_str]
    
    # Try case-insensitive match
    if team_name_mapping:
        team_str_upper = str(team_str).upper()
        for name, canonical in team_name_mapping.items():
            if name.upper() == team_str_upper:
                return canonical
    
    # For ambiguous "New York" or "Chicago" without abbreviation, return as-is
    # (will be handled by using performance data's team column)
    if team_str == 'New York' or team_str == 'Chicago' or team_str == 'Los Angeles':
        # Return as-is - will be resolved using team abbreviation from performance data
        return team_str
    
    # Return original if no match found (might already be canonical)
    return str(team_str)


def determine_free_agents_and_teams():
    """
    Determine which players are free agents and assign team codes.
    Returns a DataFrame with player_id, team_or_fa, and locked status.
    
    Logic:
    1. Players with 2025 contracts:
       - If 2025 contract is a NEW signing (team changed AND AAV changed) → FA after 2024
       - If only team changed (AAV same) → trade (not FA, use 2025 team)
       - If only AAV changed (team same) → extension (not FA, use 2025 team)
       - If neither changed → same contract (not FA, use 2025 team)
    2. Players without 2025 contracts but with 2024 data → FA, locked = false
    
    Goal: Identify players who were free agents AFTER the 2024 season (before 2025 signings).
    A player is a free agent only if BOTH their team AND AAV changed from 2024 to 2025.
    """
    print("\n--- Determining Free Agents and Team Assignments ---")
    
    team_abbrev_mapping = get_team_abbrev_to_canonical()
    team_name_mapping = get_team_name_to_canonical()
    player_status = []
    
    # Get all players from 2024 performance data (need 'team' column for abbreviations)
    hitters_2024 = pd.read_csv('data/Hitters_2015-2025_byYear_retry.csv')
    hitters_2024 = hitters_2024[hitters_2024['Season'] == 2024]
    pitchers_2024 = pd.read_csv('data/Pitchers_2015-2025_byYear_retry.csv')
    pitchers_2024 = pitchers_2024[pitchers_2024['Season'] == 2024]
    
    # Create lookup for player_id -> team abbreviation from performance data
    hitter_team_abbrev = {}
    if 'team' in hitters_2024.columns:
        for _, row in hitters_2024.iterrows():
            player_id = row['mlbID']
            if pd.notna(row.get('team')):
                hitter_team_abbrev[player_id] = str(row['team']).upper().strip()
    
    pitcher_team_abbrev = {}
    if 'team' in pitchers_2024.columns:
        for _, row in pitchers_2024.iterrows():
            player_id = row['mlbID']
            if pd.notna(row.get('team')):
                pitcher_team_abbrev[player_id] = str(row['team']).upper().strip()
    
    # Get 2024 and 2025 contracts
    hitter_contracts = pd.read_csv('data/Hitters_with_Contracts.csv')
    hitter_contracts_2024 = hitter_contracts[hitter_contracts['Season'] == 2024]
    hitter_contracts_2025 = hitter_contracts[hitter_contracts['Season'] == 2025]
    
    pitcher_contracts = pd.read_csv('data/Pitchers_with_Contracts.csv')
    pitcher_contracts_2024 = pitcher_contracts[pitcher_contracts['Season'] == 2024]
    pitcher_contracts_2025 = pitcher_contracts[pitcher_contracts['Season'] == 2025]
    
    # Create lookup for 2024 contracts (player_id -> (team, aav))
    hitter_contracts_2024_dict = {}
    if 'mlbID' in hitter_contracts_2024.columns and 'Tm' in hitter_contracts_2024.columns:
        for _, row in hitter_contracts_2024.iterrows():
            player_id = row['mlbID']
            if player_id not in hitter_contracts_2024_dict:  # Take first if multiple
                # Use team abbreviation from performance data to resolve ambiguous cases
                team_abbrev = hitter_team_abbrev.get(player_id)
                team = normalize_team_name(
                    row['Tm'], 
                    team_abbrev_mapping=team_abbrev_mapping,
                    team_name_mapping=team_name_mapping,
                    team_abbrev=team_abbrev
                )
                aav = row.get('AnnualValue', row.get('Salary', 0))
                hitter_contracts_2024_dict[player_id] = (team, aav)
    
    pitcher_contracts_2024_dict = {}
    if 'mlbID' in pitcher_contracts_2024.columns and 'Tm' in pitcher_contracts_2024.columns:
        for _, row in pitcher_contracts_2024.iterrows():
            player_id = row['mlbID']
            if player_id not in pitcher_contracts_2024_dict:  # Take first if multiple
                # Use team abbreviation from performance data to resolve ambiguous cases
                team_abbrev = pitcher_team_abbrev.get(player_id)
                team = normalize_team_name(
                    row['Tm'],
                    team_abbrev_mapping=team_abbrev_mapping,
                    team_name_mapping=team_name_mapping,
                    team_abbrev=team_abbrev
                )
                aav = row.get('AnnualValue', row.get('Salary', 0))
                pitcher_contracts_2024_dict[player_id] = (team, aav)
    
    # Create lookup for 2025 contracts (player_id -> (team, aav))
    # For 2025, try to get team abbrev from 2025 performance data if available
    hitters_2025 = pd.read_csv('data/Hitters_2015-2025_byYear_retry.csv')
    hitters_2025 = hitters_2025[hitters_2025['Season'] == 2025]
    hitter_team_abbrev_2025 = {}
    if 'team' in hitters_2025.columns:
        for _, row in hitters_2025.iterrows():
            player_id = row['mlbID']
            if pd.notna(row.get('team')):
                hitter_team_abbrev_2025[player_id] = str(row['team']).upper().strip()
    
    hitter_contracts_2025_dict = {}
    if 'mlbID' in hitter_contracts_2025.columns and 'Tm' in hitter_contracts_2025.columns:
        for _, row in hitter_contracts_2025.iterrows():
            player_id = row['mlbID']
            if player_id not in hitter_contracts_2025_dict:  # Take first if multiple
                # Use 2025 team abbreviation if available, otherwise try 2024, otherwise None
                team_abbrev = hitter_team_abbrev_2025.get(player_id) or hitter_team_abbrev.get(player_id)
                team = normalize_team_name(
                    row['Tm'],
                    team_abbrev_mapping=team_abbrev_mapping,
                    team_name_mapping=team_name_mapping,
                    team_abbrev=team_abbrev
                )
                aav = row.get('AnnualValue', row.get('Salary', 0))
                hitter_contracts_2025_dict[player_id] = (team, aav)
    
    # For 2025, try to get team abbrev from 2025 performance data if available
    pitchers_2025 = pd.read_csv('data/Pitchers_2015-2025_byYear_retry.csv')
    pitchers_2025 = pitchers_2025[pitchers_2025['Season'] == 2025]
    pitcher_team_abbrev_2025 = {}
    if 'team' in pitchers_2025.columns:
        for _, row in pitchers_2025.iterrows():
            player_id = row['mlbID']
            if pd.notna(row.get('team')):
                pitcher_team_abbrev_2025[player_id] = str(row['team']).upper().strip()
    
    pitcher_contracts_2025_dict = {}
    if 'mlbID' in pitcher_contracts_2025.columns and 'Tm' in pitcher_contracts_2025.columns:
        for _, row in pitcher_contracts_2025.iterrows():
            player_id = row['mlbID']
            if player_id not in pitcher_contracts_2025_dict:  # Take first if multiple
                # Use 2025 team abbreviation if available, otherwise try 2024, otherwise None
                team_abbrev = pitcher_team_abbrev_2025.get(player_id) or pitcher_team_abbrev.get(player_id)
                team = normalize_team_name(
                    row['Tm'],
                    team_abbrev_mapping=team_abbrev_mapping,
                    team_name_mapping=team_name_mapping,
                    team_abbrev=team_abbrev
                )
                aav = row.get('AnnualValue', row.get('Salary', 0))
                pitcher_contracts_2025_dict[player_id] = (team, aav)
    
    # Process hitters
    for _, row in hitters_2024.iterrows():
        player_id = row['mlbID']
        
        if player_id in hitter_contracts_2025_dict:
            # Has 2025 contract - check if it's a new signing or extension
            team_2025, aav_2025 = hitter_contracts_2025_dict[player_id]
            
            if player_id in hitter_contracts_2024_dict:
                # Had 2024 contract - check if 2025 is extension or new signing
                team_2024, aav_2024 = hitter_contracts_2024_dict[player_id]
                
                # Check if team changed
                team_changed = (team_2024 != team_2025)
                
                # Check if AAV changed (any change, not just significant)
                if aav_2024 > 0:
                    aav_changed = abs(aav_2025 - aav_2024) > 0.01  # Small threshold to account for rounding
                else:
                    aav_changed = True  # If no 2024 AAV, assume AAV changed
                
                # Free agent detection: BOTH team changed AND AAV changed must be true
                # If only team changed → trade (not FA, use 2025 team)
                # If only AAV changed → extension (not FA, use 2025 team)
                # If both changed → free agent signing (was FA after 2024)
                if team_changed and aav_changed:
                    # Was a free agent after 2024, signed new contract for 2025
                    team_or_fa = 'FA'
                    locked = False
                else:
                    # Either extension (team same, AAV changed) or trade (team changed, AAV same)
                    # In both cases, not a FA - use 2025 team
                    team_or_fa = team_2025
                    locked = True
            else:
                # No 2024 contract but has 2025 contract - likely a call-up, not a FA
                # Use 2025 team but mark as not locked (or could be FA, depends on context)
                team_or_fa = team_2025
                locked = True
        else:
            # No 2025 contract → free agent after 2024
            team_or_fa = 'FA'
            locked = False
        
        player_status.append({
            'player_id': player_id,
            'team_or_fa': team_or_fa,
            'locked': locked
        })
    
    # Process pitchers (same logic)
    for _, row in pitchers_2024.iterrows():
        player_id = row['mlbID']
        
        if player_id in pitcher_contracts_2025_dict:
            # Has 2025 contract - check if it's a new signing or extension
            team_2025, aav_2025 = pitcher_contracts_2025_dict[player_id]
            
            if player_id in pitcher_contracts_2024_dict:
                # Had 2024 contract - check if 2025 is extension or new signing
                team_2024, aav_2024 = pitcher_contracts_2024_dict[player_id]
                
                # Check if team changed
                team_changed = (team_2024 != team_2025)
                
                # Check if AAV changed (any change, not just significant)
                if aav_2024 > 0:
                    aav_changed = abs(aav_2025 - aav_2024) > 0.01  # Small threshold to account for rounding
                else:
                    aav_changed = True  # If no 2024 AAV, assume AAV changed
                
                # Free agent detection: BOTH team changed AND AAV changed must be true
                # If only team changed → trade (not FA, use 2025 team)
                # If only AAV changed → extension (not FA, use 2025 team)
                # If both changed → free agent signing (was FA after 2024)
                if team_changed and aav_changed:
                    # Was a free agent after 2024, signed new contract for 2025
                    team_or_fa = 'FA'
                    locked = False
                else:
                    # Either extension (team same, AAV changed) or trade (team changed, AAV same)
                    # In both cases, not a FA - use 2025 team
                    team_or_fa = team_2025
                    locked = True
            else:
                # No 2024 contract but has 2025 contract - likely a call-up, not a FA
                team_or_fa = team_2025
                locked = True
        else:
            # No 2025 contract → free agent after 2024
            team_or_fa = 'FA'
            locked = False
        
        player_status.append({
            'player_id': player_id,
            'team_or_fa': team_or_fa,
            'locked': locked
        })
    
    status_df = pd.DataFrame(player_status)
    status_df = status_df.drop_duplicates(subset=['player_id'], keep='first')
    
    fa_count = (status_df['team_or_fa'] == 'FA').sum()
    signed_count = (status_df['team_or_fa'] != 'FA').sum()
    
    print(f"  Total players processed: {len(status_df)}")
    print(f"  Free agents: {fa_count}")
    print(f"  Players with teams: {signed_count}")
    print(f"  Locked players (non-FA): {status_df['locked'].sum()}")
    
    return status_df


def load_market_costs(most_recent_season=2025):
    """
    Load market costs (AnnualValue) from contract CSV files for the most recent season.
    Returns a DataFrame with player_id and market_cost.
    """
    print(f"\n--- Loading Market Costs for Season {most_recent_season} ---")
    
    market_costs = []
    
    # Load hitter contracts
    try:
        hitter_contracts = pd.read_csv('data/Hitters_with_Contracts.csv')
        if 'mlbID' in hitter_contracts.columns and 'AnnualValue' in hitter_contracts.columns:
            hitter_costs = hitter_contracts[
                hitter_contracts['Season'] == most_recent_season
            ][['mlbID', 'AnnualValue']].copy()
            hitter_costs = hitter_costs.rename(columns={'mlbID': 'player_id'})
            hitter_costs['market_cost'] = hitter_costs['AnnualValue'] / 1e6  # Convert to millions
            hitter_costs = hitter_costs[['player_id', 'market_cost']].dropna()
            market_costs.append(hitter_costs)
            print(f"  Loaded {len(hitter_costs)} hitter contracts")
    except Exception as e:
        print(f"  Warning: Could not load hitter contracts: {e}")
    
    # Load pitcher contracts
    try:
        pitcher_contracts = pd.read_csv('data/Pitchers_with_Contracts.csv')
        if 'mlbID' in pitcher_contracts.columns and 'AnnualValue' in pitcher_contracts.columns:
            pitcher_costs = pitcher_contracts[
                pitcher_contracts['Season'] == most_recent_season
            ][['mlbID', 'AnnualValue']].copy()
            pitcher_costs = pitcher_costs.rename(columns={'mlbID': 'player_id'})
            pitcher_costs['market_cost'] = pitcher_costs['AnnualValue'] / 1e6  # Convert to millions
            pitcher_costs = pitcher_costs[['player_id', 'market_cost']].dropna()
            market_costs.append(pitcher_costs)
            print(f"  Loaded {len(pitcher_costs)} pitcher contracts")
    except Exception as e:
        print(f"  Warning: Could not load pitcher contracts: {e}")
    
    if market_costs:
        all_market_costs = pd.concat(market_costs, ignore_index=True)
        # If a player has multiple contracts, take the maximum (or could take mean)
        all_market_costs = all_market_costs.groupby('player_id')['market_cost'].max().reset_index()
        print(f"  Total unique players with market costs: {len(all_market_costs)}")
        return all_market_costs
    else:
        print("  No market costs loaded")
        return pd.DataFrame(columns=['player_id', 'market_cost'])


def main():
    print("=" * 60)
    print("ROSTER OPTIMIZATION DATA PREPARATION")
    print("=" * 60)
    
    # Load models
    print("\n[1/4] Loading trained models...")
    models = load_models()
    
    if not models:
        print("\n⚠ No models found! Please run the training notebooks first:")
        print("  1. hitter_war_t_to_tp1_multiyear_backfill.ipynb")
        print("  2. pitcher_war_t_to_tp1_multiyear_optimized.ipynb")
        print("  3. aav_regression.ipynb")
        print("\nUsing fallback predictions (WAR regressed to mean)...")
    
    # Prepare hitter data
    print("\n[2/4] Preparing hitter data...")
    hitters = prepare_hitter_data(models)
    hitters = assign_hitter_positions(hitters)
    
    # Prepare pitcher data
    print("\n[3/4] Preparing pitcher data...")
    pitchers = prepare_pitcher_data(models)
    
    # Combine all players
    print("\n[4/6] Combining player data...")
    
    # Select columns for output
    output_cols = ['player_id', 'name', 'position', 'xwar', 'cost']
    
    all_players = pd.concat([
        hitters[output_cols],
        pitchers[output_cols]
    ], ignore_index=True)
    
    # Remove players with no xWAR prediction or invalid data
    all_players = all_players.dropna(subset=['xwar', 'cost'])
    all_players = all_players[all_players['cost'] > 0]
    
    # Cap costs at reasonable max (no player > $50M for optimization)
    all_players['cost'] = all_players['cost'].clip(upper=50.0)
    
    # Round values for cleaner output
    all_players['xwar'] = all_players['xwar'].round(3)
    all_players['cost'] = all_players['cost'].round(3)
    
    # Load and merge actual 2025 WAR data for comparison
    print("\n[5/6] Loading and merging actual 2025 WAR data...")
    actual_war_2025 = load_actual_war_2025()
    
    if len(actual_war_2025) > 0:
        # Merge actual WAR on player_id
        all_players = all_players.merge(
            actual_war_2025[['player_id', 'actual_war_2025', 'war_2025', 'g_2025']],
            on='player_id',
            how='left'
        )
        # Round actual_war_2025
        all_players['actual_war_2025'] = all_players['actual_war_2025'].round(3)
        print(f"  Merged actual 2025 WAR for {all_players['actual_war_2025'].notna().sum()} players")
    else:
        # Add empty columns if no 2025 data found
        all_players['actual_war_2025'] = np.nan
        all_players['war_2025'] = np.nan
        all_players['g_2025'] = np.nan
        print("  No 2025 actual WAR data available to merge")
    
    # Load and merge market costs
    print("\n[6/7] Loading and merging market costs...")
    market_costs = load_market_costs(most_recent_season=2025)
    
    if len(market_costs) > 0:
        # Merge market costs on player_id
        all_players = all_players.merge(
            market_costs,
            on='player_id',
            how='left'
        )
        # Round market_cost
        all_players['market_cost'] = all_players['market_cost'].round(3)
        print(f"  Merged market costs for {all_players['market_cost'].notna().sum()} players")
    else:
        # Add empty market_cost column if no contracts found
        all_players['market_cost'] = np.nan
        print("  No market costs available to merge")
    
    # Determine free agents and team assignments
    print("\n[7/7] Determining free agents and team assignments...")
    player_status = determine_free_agents_and_teams()
    
    # Merge team_or_fa and locked columns
    all_players = all_players.merge(
        player_status[['player_id', 'team_or_fa', 'locked']],
        on='player_id',
        how='left'
    )
    
    # Fill missing values (players not in 2024 data)
    all_players['team_or_fa'] = all_players['team_or_fa'].fillna('FA')
    all_players['locked'] = all_players['locked'].fillna(False)
    
    print(f"  Merged team/FA status for {len(all_players)} players")
    print(f"  Free agents in dataset: {(all_players['team_or_fa'] == 'FA').sum()}")
    print(f"  Players with teams: {(all_players['team_or_fa'] != 'FA').sum()}")
    
    # Save to CSV
    all_players.to_csv('players.csv', index=False)
    
    print(f"\n✓ Saved {len(all_players)} players to 'players.csv'")
    print(f"\nPosition breakdown:")
    print(all_players['position'].value_counts())
    print(f"\nTop 10 players by xWAR:")
    display_cols = ['name', 'position', 'xwar', 'cost']
    if 'market_cost' in all_players.columns:
        display_cols.append('market_cost')
    if 'actual_war_2025' in all_players.columns:
        display_cols.append('actual_war_2025')
    print(all_players.nlargest(10, 'xwar')[display_cols].to_string(index=False))
    
    if 'market_cost' in all_players.columns and all_players['market_cost'].notna().sum() > 0:
        print(f"\nMarket cost statistics:")
        print(f"  Players with market cost: {all_players['market_cost'].notna().sum()}")
        print(f"  Average market cost: ${all_players['market_cost'].mean():.2f}M")
        print(f"  Median market cost: ${all_players['market_cost'].median():.2f}M")
        print(f"  Max market cost: ${all_players['market_cost'].max():.2f}M")
    
    if 'actual_war_2025' in all_players.columns and all_players['actual_war_2025'].notna().sum() > 0:
        print(f"\nActual vs Predicted WAR (2025) statistics:")
        players_with_actual = all_players[all_players['actual_war_2025'].notna()]
        print(f"  Players with actual 2025 WAR: {len(players_with_actual)}")
        print(f"  Average predicted xWAR: {players_with_actual['xwar'].mean():.2f}")
        print(f"  Average actual WAR/162: {players_with_actual['actual_war_2025'].mean():.2f}")
        print(f"  Mean absolute error: {np.abs(players_with_actual['xwar'] - players_with_actual['actual_war_2025']).mean():.2f}")
    
    print("\n" + "=" * 60)
    print("READY FOR OPTIMIZATION")
    print("=" * 60)
    print("\nNow run roster_optimizer.ipynb to build the optimal 2025 team!")


if __name__ == '__main__':
    main()
