import pandas as pd
import numpy as np

def load_insurance_data(filepath, separator='|'):
    """Loads insurance dataset safely with handling for large string/object limits."""
    try:
        df = pd.read_csv(filepath, sep=separator, low_memory=False)
        print(f"✅ Data successfully loaded. Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None

def compute_portfolio_loss_ratio(df, premium_col='TotalPremium', claims_col='TotalClaims'):
    """Calculates overall financial portfolio Loss Ratio percentage."""
    total_premiums = df[premium_col].sum()
    total_claims = df[claims_col].sum()
    if total_premiums == 0:
        return 0.0
    return (total_claims / total_premiums) * 100

def check_missing_data_quality(df):
    """Returns a dataframe mapping column names to missing value percentages."""
    missing_counts = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    quality_df = pd.DataFrame({'Missing Counts': missing_counts, 'Percentage (%)': missing_pct})
    return quality_df[quality_df['Missing Counts'] > 0].sort_values(by='Percentage (%)', ascending=False)