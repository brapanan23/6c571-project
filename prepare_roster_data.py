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
- Run hitter_war_t_to_tp1_multiyear_backfill.ipynb first (saves models/hitter_war_model.joblib)
- Run pitcher_war_t_to_tp1_multiyear_optimized.ipynb first (saves models/pitcher_war_model.joblib)
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
    if os.path.exists('models/hitter_war_model.joblib'):
        models['hitter_war'] = joblib.load('models/hitter_war_model.joblib')
        print(f"✓ Loaded hitter WAR model ({len(models['hitter_war']['feature_cols'])} features)")
    else:
        print("✗ Missing hitter WAR model - run hitter_war_t_to_tp1_multiyear_backfill.ipynb first")
    
    # Load pitcher WAR model
    if os.path.exists('models/pitcher_war_model.joblib'):
        models['pitcher_war'] = joblib.load('models/pitcher_war_model.joblib')
        print(f"✓ Loaded pitcher WAR model ({len(models['pitcher_war']['feature_cols'])} features)")
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


def prepare_hitter_data(models):
    """Load 2024 hitter data and predict 2025 xWAR and expected cost."""
    print("\n--- Processing Hitters ---")
    
    # Load hitter data
    hitters_df = pd.read_csv('data/Hitters_2015-2025_byYear_retry.csv')
    hitters_2024 = hitters_df[hitters_df['Season'] == 2024].copy()
    print(f"Loaded {len(hitters_2024)} hitters from 2024 season")
    
    # Compute WAR_per_162 for current season
    hitters_2024['WAR_per_162'] = (hitters_2024['WAR'] / hitters_2024['G'].replace(0, np.nan)) * 162
    
    # Get WAR model artifacts
    war_model = models.get('hitter_war')
    salary_model = models.get('hitter_salary')
    
    results = []
    
    for idx, row in hitters_2024.iterrows():
        player_data = {
            'player_id': row['mlbID'],
            'name': row['Name'],
            'position': 'OF',  # Default - will get from training data if available
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
                feature_cols = war_model['feature_cols']
                X = pd.DataFrame([row])[feature_cols].values
                X_imp = war_model['imputer'].transform(X)
                X_scaled = war_model['scaler'].transform(X_imp)
                xwar = war_model['model'].predict(X_scaled)[0]
                player_data['xwar'] = xwar
            except Exception as e:
                # Fallback: use current WAR regressed to mean
                xwar = row['WAR_per_162'] * 0.7 if pd.notna(row['WAR_per_162']) else 0
                # Cap at reasonable range
                player_data['xwar'] = np.clip(xwar, -3, 10)
        else:
            # Fallback: use current WAR regressed to mean (only for players with meaningful PA)
            if row['PA'] >= 100:
                xwar = row['WAR_per_162'] * 0.7 if pd.notna(row['WAR_per_162']) else 0
                # Cap at reasonable range
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
                feature_cols = war_model['feature_cols']
                # Add IP_next as a feature if required (use current IP as estimate)
                row_copy = row.copy()
                if 'IP_next' in feature_cols:
                    row_copy['IP_next'] = row['IP']
                X = pd.DataFrame([row_copy])[feature_cols].values
                X_imp = war_model['imputer'].transform(X)
                X_scaled = war_model['scaler'].transform(X_imp)
                xwar = war_model['model'].predict(X_scaled)[0]
                player_data['xwar'] = np.clip(xwar, -3, 10)
            except Exception as e:
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
    print("\n[4/4] Combining and saving player data...")
    
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
    
    # Save to CSV
    all_players.to_csv('players.csv', index=False)
    
    print(f"\n✓ Saved {len(all_players)} players to 'players.csv'")
    print(f"\nPosition breakdown:")
    print(all_players['position'].value_counts())
    print(f"\nTop 10 players by xWAR:")
    print(all_players.nlargest(10, 'xwar')[['name', 'position', 'xwar', 'cost']].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("READY FOR OPTIMIZATION")
    print("=" * 60)
    print("\nNow run roster_optimizer.ipynb to build the optimal 2025 team!")


if __name__ == '__main__':
    main()

