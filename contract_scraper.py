"""
MLB Contract Data Scraper

Scrapes salary/contract data from Spotrac team payroll pages.
Calculates annual value (price per year) for each player.

- Hitters: 2023, 2024, 2025
- Pitchers: 2015-2025

Output format includes:
- Player name
- Team
- Position
- Base Salary (annual value)
- Season
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
from io import StringIO
import os
from datetime import datetime
from tqdm import tqdm

# Headers to mimic browser request
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# MLB Teams for Spotrac URLs
MLB_TEAMS = [
    "arizona-diamondbacks",
    "atlanta-braves",
    "baltimore-orioles",
    "boston-red-sox",
    "chicago-cubs",
    "chicago-white-sox",
    "cincinnati-reds",
    "cleveland-guardians",
    "colorado-rockies",
    "detroit-tigers",
    "houston-astros",
    "kansas-city-royals",
    "los-angeles-angels",
    "los-angeles-dodgers",
    "miami-marlins",
    "milwaukee-brewers",
    "minnesota-twins",
    "new-york-mets",
    "new-york-yankees",
    "oakland-athletics",
    "philadelphia-phillies",
    "pittsburgh-pirates",
    "san-diego-padres",
    "san-francisco-giants",
    "seattle-mariners",
    "st-louis-cardinals",
    "tampa-bay-rays",
    "texas-rangers",
    "toronto-blue-jays",
    "washington-nationals",
]

# Team name mappings for matching with existing data
TEAM_ABBREV = {
    "arizona-diamondbacks": "AZ",
    "atlanta-braves": "ATL",
    "baltimore-orioles": "BAL",
    "boston-red-sox": "BOS",
    "chicago-cubs": "CHC",
    "chicago-white-sox": "CWS",
    "cincinnati-reds": "CIN",
    "cleveland-guardians": "CLE",
    "colorado-rockies": "COL",
    "detroit-tigers": "DET",
    "houston-astros": "HOU",
    "kansas-city-royals": "KC",
    "los-angeles-angels": "LAA",
    "los-angeles-dodgers": "LAD",
    "miami-marlins": "MIA",
    "milwaukee-brewers": "MIL",
    "minnesota-twins": "MIN",
    "new-york-mets": "NYM",
    "new-york-yankees": "NYY",
    "oakland-athletics": "OAK",
    "philadelphia-phillies": "PHI",
    "pittsburgh-pirates": "PIT",
    "san-diego-padres": "SD",
    "san-francisco-giants": "SF",
    "seattle-mariners": "SEA",
    "st-louis-cardinals": "STL",
    "tampa-bay-rays": "TB",
    "texas-rangers": "TEX",
    "toronto-blue-jays": "TOR",
    "washington-nationals": "WSH",
}

# Pitcher positions
PITCHER_POSITIONS = ['SP', 'RP', 'CL', 'P']


def parse_money(value: str) -> float:
    """Parse money string like '$10,000,000' to float."""
    if pd.isna(value) or value == '-' or value == '' or value is None:
        return 0.0
    
    value = str(value).strip().replace('$', '').replace(',', '')
    
    # Handle millions (M)
    if 'M' in value.upper():
        value = value.upper().replace('M', '')
        try:
            return float(value) * 1_000_000
        except ValueError:
            return 0.0
    
    # Handle thousands (K)
    if 'K' in value.upper():
        value = value.upper().replace('K', '')
        try:
            return float(value) * 1_000
        except ValueError:
            return 0.0
    
    try:
        return float(value)
    except ValueError:
        return 0.0


def clean_player_name(name: str) -> str:
    """
    Clean and normalize player name from Spotrac format.
    Spotrac often has format like "Judge Aaron Judge" - we extract the full name.
    """
    if pd.isna(name):
        return ""
    
    name = str(name).strip()
    
    # Spotrac format: "LastName FirstName LastName" - extract "FirstName LastName"
    # Try to detect this pattern
    words = name.split()
    if len(words) >= 3:
        # Check if first word is repeated at end
        if words[0].lower() == words[-1].lower():
            # Pattern: "Judge Aaron Judge" -> "Aaron Judge"
            name = ' '.join(words[1:])
        elif len(words) == 4 and words[0].lower() == words[2].lower():
            # Pattern: "Last First Middle Last" -> "First Middle Last"
            name = ' '.join(words[1:])
    
    # Remove common suffixes
    for suffix in [' Jr.', ' Jr', ' Sr.', ' Sr', ' III', ' II', ' IV']:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    
    # Remove asterisks and other markers
    name = re.sub(r'[\*\#\+]', '', name)
    
    return name.strip()


def scrape_team_payroll(team: str, year: int) -> pd.DataFrame:
    """
    Scrape payroll data for a specific team and year from Spotrac.
    
    Parameters:
    -----------
    team : str
        Team slug (e.g., 'new-york-yankees')
    year : int
        Season year
    
    Returns:
    --------
    DataFrame with player salary data
    """
    url = f"https://www.spotrac.com/mlb/{team}/payroll/{year}/"
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return pd.DataFrame()
    except requests.RequestException as e:
        print(f"[{team}] Error: {e}")
        return pd.DataFrame()
    
    soup = BeautifulSoup(resp.text, 'html.parser')
    tables = soup.find_all('table')
    
    if not tables:
        return pd.DataFrame()
    
    all_players = []
    
    for table in tables:
        try:
            df = pd.read_html(StringIO(str(table)))[0]
            
            # Look for player column (usually first column with format "Player (N)")
            player_col = None
            for col in df.columns:
                if 'player' in str(col).lower():
                    player_col = col
                    break
            
            if player_col is None:
                continue
            
            # Look for salary columns
            salary_col = None
            base_salary_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if 'base salary' in col_lower or 'base_salary' in col_lower:
                    base_salary_col = col
                elif 'salary' in col_lower and 'signing' not in col_lower:
                    salary_col = col
            
            # Use base salary if available, otherwise any salary column
            use_salary_col = base_salary_col or salary_col
            
            if use_salary_col is None:
                continue
            
            # Look for position column
            pos_col = None
            for col in df.columns:
                if str(col).lower() == 'pos':
                    pos_col = col
                    break
            
            # Extract data
            for _, row in df.iterrows():
                player_name = clean_player_name(str(row[player_col]))
                salary = parse_money(str(row[use_salary_col]))
                position = str(row[pos_col]) if pos_col and pd.notna(row[pos_col]) else ''
                
                if player_name and salary > 0:
                    all_players.append({
                        'Name': player_name,
                        'Team': TEAM_ABBREV.get(team, team.upper()[:3]),
                        'TeamFull': team,
                        'Position': position,
                        'Salary': salary,
                        'AnnualValue': salary,  # Already annual value
                        'Season': year,
                    })
        
        except Exception as e:
            continue
    
    return pd.DataFrame(all_players)


def scrape_all_teams_payroll(year: int, delay: float = 0.5) -> pd.DataFrame:
    """
    Scrape payroll data for all MLB teams for a given year.
    
    Parameters:
    -----------
    year : int
        Season year
    delay : float
        Delay between requests (be nice to servers)
    
    Returns:
    --------
    DataFrame with all players' salary data
    """
    all_data = []
    
    for team in tqdm(MLB_TEAMS, desc=f"Scraping {year}", unit="team"):
        df = scrape_team_payroll(team, year)
        if not df.empty:
            all_data.append(df)
        time.sleep(delay)  # Rate limiting
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        print(f"[{year}] Total: {len(result)} player records")
        return result
    
    return pd.DataFrame()


def get_hitter_contracts(years: list) -> pd.DataFrame:
    """
    Get salary data for hitters for specified years.
    Filters out pitchers based on position.
    
    Parameters:
    -----------
    years : list
        List of years to fetch (e.g., [2023, 2024, 2025])
    
    Returns:
    --------
    DataFrame with hitter salary data
    """
    all_data = []
    
    for year in tqdm(years, desc="Hitter years", unit="year"):
        df = scrape_all_teams_payroll(year)
        
        if not df.empty:
            # Filter to non-pitchers
            df_hitters = df[~df['Position'].isin(PITCHER_POSITIONS)].copy()
            df_hitters['PlayerType'] = 'Hitter'
            all_data.append(df_hitters)
            tqdm.write(f"[{year}] Hitters: {len(df_hitters)} (filtered from {len(df)} total)")
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        # Remove duplicates (player might be on multiple tables)
        result = result.sort_values('Salary', ascending=False).drop_duplicates(
            subset=['Name', 'Season'], keep='first'
        )
        return result
    
    return pd.DataFrame()


def get_pitcher_contracts(years: list) -> pd.DataFrame:
    """
    Get salary data for pitchers for specified years.
    Filters to only include pitchers based on position.
    
    Parameters:
    -----------
    years : list
        List of years to fetch
    
    Returns:
    --------
    DataFrame with pitcher salary data
    """
    all_data = []
    
    for year in tqdm(years, desc="Pitcher years", unit="year"):
        df = scrape_all_teams_payroll(year)
        
        if not df.empty:
            # Filter to pitchers only
            df_pitchers = df[df['Position'].isin(PITCHER_POSITIONS)].copy()
            df_pitchers['PlayerType'] = 'Pitcher'
            all_data.append(df_pitchers)
            tqdm.write(f"[{year}] Pitchers: {len(df_pitchers)} (filtered from {len(df)} total)")
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values('Salary', ascending=False).drop_duplicates(
            subset=['Name', 'Season'], keep='first'
        )
        return result
    
    return pd.DataFrame()


def fix_escaped_unicode(name: str) -> str:
    """
    Decode escaped UTF-8 bytes like \\xc3\\xb1 back to proper unicode (ñ).
    This handles CSVs with improperly encoded unicode characters.
    """
    import codecs
    if pd.isna(name):
        return ''
    name = str(name)
    if '\\x' in name:
        try:
            name = codecs.decode(name, 'unicode_escape')
            name = name.encode('latin-1').decode('utf-8')
        except:
            pass
    return name


def extract_spotrac_name(name: str) -> str:
    """
    Fix Spotrac name format: "LastName Suffix FirstName LastName" -> "FirstName LastName"
    
    Examples:
    - "Witt Jr. Bobby Witt" -> "Bobby Witt"
    - "Harris II Michael Harris" -> "Michael Harris"  
    - "De La Cruz Elly De La Cruz" -> "Elly De La Cruz"
    """
    if pd.isna(name):
        return ''
    name = str(name).strip()
    
    # Remove leading numbers (like "2 CJ Abrams")
    name = re.sub(r'^\d+\s+', '', name)
    
    # Handle suffix patterns (Jr., II, etc.) - take what comes after
    suffix_patterns = [
        r'\s+Jr\.?\s+',
        r'\s+Sr\.?\s+',
        r'\s+II\s+',
        r'\s+III\s+',
        r'\s+IV\s+',
    ]
    
    for pattern in suffix_patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            after = name[match.end():].strip()
            if after:
                return after
    
    # Handle repeated name pattern (no suffix): "LastName FirstName LastName"
    words = name.split()
    if len(words) >= 3:
        for i in range(1, len(words) // 2 + 1):
            start_part = ' '.join(words[:i]).lower()
            end_part = ' '.join(words[-i:]).lower()
            if start_part == end_part:
                return ' '.join(words[i:])
    
    return name


def normalize_name_for_matching(name: str, is_contract: bool = False) -> str:
    """
    Normalize name for matching between datasets.
    Handles unicode, accents, Spotrac format quirks, and common variations.
    
    Parameters:
    -----------
    name : str
        Player name to normalize
    is_contract : bool
        If True, apply Spotrac-specific name extraction first
    """
    import unicodedata
    
    if pd.isna(name):
        return ""
    
    # Fix escaped unicode sequences
    name = fix_escaped_unicode(name)
    
    # Fix Spotrac format if this is contract data
    if is_contract:
        name = extract_spotrac_name(name)
    
    # Normalize unicode (remove accents)
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    name = name.lower()
    
    # Remove middle initials (like "H." in "Josh H. Smith")
    name = re.sub(r'\b\w\.\s*', '', name)
    name = name.replace('.', '')
    
    # Remove remaining punctuation
    name = re.sub(r'[^\w\s]', '', name)
    
    # Remove suffixes at end
    for suffix in [' jr', ' sr', ' iii', ' ii', ' iv']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    
    return ' '.join(name.split()).strip()


def match_contracts_with_data(contracts_df: pd.DataFrame, 
                               existing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match contract data with existing player data.
    
    Parameters:
    -----------
    contracts_df : DataFrame
        Contract/salary data from scraping
    existing_df : DataFrame
        Existing player data (from data_pull.py output)
    
    Returns:
    --------
    Existing DataFrame with salary columns added
    """
    contracts_df = contracts_df.copy()
    existing_df = existing_df.copy()
    
    # Normalize names for matching (use is_contract=True for Spotrac data format)
    contracts_df['Name_match'] = contracts_df['Name'].apply(
        lambda x: normalize_name_for_matching(x, is_contract=True)
    )
    existing_df['Name_match'] = existing_df['Name'].apply(
        lambda x: normalize_name_for_matching(x, is_contract=False)
    )
    
    # Select columns to merge
    merge_cols = ['Name_match', 'Season', 'Salary', 'AnnualValue', 'Position']
    
    # Ensure Season is int
    contracts_df['Season'] = contracts_df['Season'].astype(int)
    existing_df['Season'] = existing_df['Season'].astype(int)
    
    # Merge on normalized name and season
    merged = existing_df.merge(
        contracts_df[merge_cols],
        on=['Name_match', 'Season'],
        how='left',
        suffixes=('', '_contract')
    )
    
    # Drop the match column
    merged = merged.drop(columns=['Name_match'])
    
    # Report match rate
    matched = merged['Salary'].notna().sum()
    total = len(merged)
    print(f"[Match] {matched}/{total} players matched with salary data ({matched/total*100:.1f}%)")
    
    return merged


def main():
    """
    Main function to scrape and process contract data.
    """
    print("="*60)
    print("MLB Contract/Salary Data Scraper")
    print("Data source: Spotrac team payroll pages")
    print("="*60)
    
    # Define years
    hitter_years = [2023, 2024, 2025]
    current_year = datetime.now().year
    pitcher_years = list(range(2015, min(current_year + 1, 2026)))
    
    print(f"\nHitter years: {hitter_years}")
    print(f"Pitcher years: {pitcher_years}")
    
    # Scrape hitter salaries
    print("\n" + "="*60)
    print("SCRAPING HITTER SALARIES")
    print("="*60)
    hitter_contracts = get_hitter_contracts(hitter_years)
    
    if not hitter_contracts.empty:
        print(f"\n[Summary] Retrieved {len(hitter_contracts)} hitter salary records")
        print(f"[Sample] Top 5 hitter salaries:")
        print(hitter_contracts.nlargest(5, 'Salary')[['Name', 'Team', 'Salary', 'Season']])
    
    # Scrape pitcher salaries
    print("\n" + "="*60)
    print("SCRAPING PITCHER SALARIES")
    print("="*60)
    pitcher_contracts = get_pitcher_contracts(pitcher_years)
    
    if not pitcher_contracts.empty:
        print(f"\n[Summary] Retrieved {len(pitcher_contracts)} pitcher salary records")
        print(f"[Sample] Top 5 pitcher salaries:")
        print(pitcher_contracts.nlargest(5, 'Salary')[['Name', 'Team', 'Salary', 'Season']])
    
    # Save to CSV
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    if not hitter_contracts.empty:
        hitter_output = os.path.join(output_dir, "Hitter_Contracts_2023-25.csv")
        hitter_contracts.to_csv(hitter_output, index=False)
        print(f"\n[Saved] {hitter_output}")
    
    if not pitcher_contracts.empty:
        pitcher_output = os.path.join(output_dir, f"Pitcher_Contracts_2015-{current_year}.csv")
        pitcher_contracts.to_csv(pitcher_output, index=False)
        print(f"[Saved] {pitcher_output}")
    
    # Try to merge with existing data if available
    print("\n" + "="*60)
    print("MERGING WITH EXISTING DATA")
    print("="*60)
    
    try:
        existing_hitters = pd.read_csv("data/Hitters_2023-24_byYear_retry.csv")
        if not hitter_contracts.empty:
            merged_hitters = match_contracts_with_data(hitter_contracts, existing_hitters)
            merged_output = os.path.join(output_dir, "Hitters_with_Contracts.csv")
            merged_hitters.to_csv(merged_output, index=False)
            print(f"[Saved] {merged_output}")
    except FileNotFoundError:
        print("[Info] Existing hitter data not found - skipping merge")
    except Exception as e:
        print(f"[Error] Hitter merge failed: {e}")
    
    try:
        existing_pitchers = pd.read_csv(f"data/Pitchers_2015-{current_year}_byYear_retry.csv")
        if not pitcher_contracts.empty:
            merged_pitchers = match_contracts_with_data(pitcher_contracts, existing_pitchers)
            merged_output = os.path.join(output_dir, "Pitchers_with_Contracts.csv")
            merged_pitchers.to_csv(merged_output, index=False)
            print(f"[Saved] {merged_output}")
    except FileNotFoundError:
        print("[Info] Existing pitcher data not found - skipping merge")
    except Exception as e:
        print(f"[Error] Pitcher merge failed: {e}")
    
    print("\n" + "="*60)
    print("Contract scraping complete!")
    print("="*60)
    
    return hitter_contracts, pitcher_contracts


if __name__ == "__main__":
    hitter_contracts, pitcher_contracts = main()
