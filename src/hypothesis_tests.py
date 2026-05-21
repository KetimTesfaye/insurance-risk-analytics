import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

def test_numerical_kpi(group_a, group_b):
    """
    Runs an Independent T-Test for numerical KPIs (Claim Severity, Margin).
    Assumes unequal variances (Welch's t-test).
    
    Returns: t_statistic, p_value
    """
    # Drop empty values to prevent math errors
    group_a = group_a.dropna()
    group_b = group_b.dropna()
    
    stat, p_value = ttest_ind(group_a, group_b, equal_var=False)
    return stat, p_value

def test_categorical_kpi(series_a, series_b):
    """
    Runs a Chi-Squared test for categorical KPIs (Claim Frequency / Occurrence).
    Expects two pandas Series containing binary data (1 = claim, 0 = no claim).
    
    Returns: chi2_statistic, p_value
    """
    # Calculate how many had claims vs how many didn't for both groups
    claims_a = series_a.sum()
    no_claims_a = len(series_a) - claims_a
    
    claims_b = series_b.sum()
    no_claims_b = len(series_b) - claims_b
    
    # Build the contingency table
    contingency_table = [
        [claims_a, no_claims_a],
        [claims_b, no_claims_b]
    ]
    
    stat, p_value, dof, expected = chi2_contingency(contingency_table)
    return stat, p_value

def interpret_p_value(p_value, alpha=0.05):
    """Prints whether to reject or fail to reject the null hypothesis."""
    if p_value < alpha:
        return f"p-value = {p_value:.5f} (< {alpha}): Reject H0. The difference IS statistically significant."
    else:
        return f"p-value = {p_value:.5f} (>= {alpha}): Fail to reject H0. The difference is NOT statistically significant."