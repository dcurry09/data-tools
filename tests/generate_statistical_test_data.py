import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import sys
import os

def generate_ab_test_data(n_control: int = 1000, n_treatment: int = 1000,
                         control_rate: float = 0.1, treatment_rate: float = 0.12,
                         random_state: int = 42) -> pd.DataFrame:
    """
    Generate A/B test data with known conversion rates
    
    Parameters:
    -----------
    n_control : int
        Number of control group observations
    n_treatment : int
        Number of treatment group observations  
    control_rate : float
        True conversion rate for control group
    treatment_rate : float
        True conversion rate for treatment group
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with group assignments and conversion outcomes
    """
    np.random.seed(random_state)
    
    # Generate control group
    control_data = pd.DataFrame({
        'group': ['control'] * n_control,
        'converted': np.random.binomial(1, control_rate, n_control),
        'user_id': range(n_control)
    })
    
    # Generate treatment group
    treatment_data = pd.DataFrame({
        'group': ['treatment'] * n_treatment,
        'converted': np.random.binomial(1, treatment_rate, n_treatment),
        'user_id': range(n_control, n_control + n_treatment)
    })
    
    # Combine data
    ab_data = pd.concat([control_data, treatment_data], ignore_index=True)
    
    # Add additional realistic features
    ab_data['days_since_signup'] = np.random.exponential(30, len(ab_data))
    ab_data['page_views'] = np.random.poisson(5, len(ab_data))
    ab_data['time_on_site'] = np.random.gamma(2, 3, len(ab_data))
    
    # Add some correlation between features and conversion
    ab_data['revenue'] = (
        ab_data['converted'] * np.random.gamma(2, 25) +  # Converted users spend more
        np.random.gamma(1, 5)  # Base revenue for all users
    )
    
    return ab_data

def generate_correlation_data(n_samples: int = 500, 
                            correlations: Dict[str, float] = None,
                            random_state: int = 42) -> pd.DataFrame:
    """
    Generate data with specified correlation structure
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    correlations : Dict[str, float]
        Dictionary mapping feature names to target correlations
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with features having specified correlations with target
    """
    if correlations is None:
        correlations = {
            'strong_positive': 0.8,
            'moderate_positive': 0.5,
            'weak_positive': 0.2,
            'no_correlation': 0.0,
            'weak_negative': -0.2,
            'moderate_negative': -0.5,
            'strong_negative': -0.8
        }
    
    np.random.seed(random_state)
    
    # Generate target variable
    target = np.random.normal(0, 1, n_samples)
    
    data = {'target': target}
    
    # Generate features with specified correlations
    for feature_name, correlation in correlations.items():
        if correlation == 0:
            # No correlation - independent random variable
            feature = np.random.normal(0, 1, n_samples)
        else:
            # Create correlated variable using linear combination
            noise_variance = 1 - correlation**2
            noise = np.random.normal(0, np.sqrt(noise_variance), n_samples)
            feature = correlation * target + noise
        
        data[feature_name] = feature
    
    return pd.DataFrame(data)

def generate_group_effect_data(groups: Dict[str, Dict] = None,
                              n_per_group: int = 100,
                              random_state: int = 42) -> pd.DataFrame:
    """
    Generate data with known group effects
    
    Parameters:
    -----------
    groups : Dict[str, Dict]
        Dictionary mapping group names to their parameters
        Each group dict should have 'mean' and 'std' keys
    n_per_group : int
        Number of observations per group
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with group variable and outcome with known effects
    """
    if groups is None:
        groups = {
            'control': {'mean': 50, 'std': 10},
            'treatment_a': {'mean': 55, 'std': 10},  # Medium effect (Cohen's d = 0.5)
            'treatment_b': {'mean': 60, 'std': 10},  # Large effect (Cohen's d = 1.0)
            'treatment_c': {'mean': 52, 'std': 10}   # Small effect (Cohen's d = 0.2)
        }
    
    np.random.seed(random_state)
    
    data = []
    for group_name, params in groups.items():
        group_data = {
            'group': [group_name] * n_per_group,
            'outcome': np.random.normal(params['mean'], params['std'], n_per_group),
            'binary_outcome': np.random.binomial(1, min(0.9, params['mean']/60), n_per_group)
        }
        
        group_df = pd.DataFrame(group_data)
        data.append(group_df)
    
    combined_data = pd.concat(data, ignore_index=True)
    
    # Add additional variables
    combined_data['participant_id'] = range(len(combined_data))
    combined_data['baseline_score'] = np.random.normal(45, 8, len(combined_data))
    
    return combined_data

def generate_power_analysis_scenarios() -> Dict[str, pd.DataFrame]:
    """
    Generate datasets for different power analysis scenarios
    
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary mapping scenario names to datasets
    """
    scenarios = {}
    
    # Small effect size scenario (Cohen's d = 0.2)
    scenarios['small_effect'] = generate_group_effect_data(
        groups={
            'control': {'mean': 50, 'std': 10},
            'treatment': {'mean': 52, 'std': 10}
        },
        n_per_group=200,
        random_state=1
    )
    
    # Medium effect size scenario (Cohen's d = 0.5) 
    scenarios['medium_effect'] = generate_group_effect_data(
        groups={
            'control': {'mean': 50, 'std': 10},
            'treatment': {'mean': 55, 'std': 10}
        },
        n_per_group=100,
        random_state=2
    )
    
    # Large effect size scenario (Cohen's d = 0.8)
    scenarios['large_effect'] = generate_group_effect_data(
        groups={
            'control': {'mean': 50, 'std': 10},
            'treatment': {'mean': 58, 'std': 10}
        },
        n_per_group=50,
        random_state=3
    )
    
    # Underpowered scenario (small effect, small sample)
    scenarios['underpowered'] = generate_group_effect_data(
        groups={
            'control': {'mean': 50, 'std': 10},
            'treatment': {'mean': 52, 'std': 10}
        },
        n_per_group=20,
        random_state=4
    )
    
    return scenarios

def generate_multiple_testing_data(n_features: int = 20, n_samples: int = 100,
                                 n_true_signals: int = 3,
                                 signal_strength: float = 0.6,
                                 random_state: int = 42) -> pd.DataFrame:
    """
    Generate data for testing multiple comparison corrections
    
    Parameters:
    -----------
    n_features : int
        Total number of features to generate
    n_samples : int
        Number of samples
    n_true_signals : int
        Number of features with true signal
    signal_strength : float
        Correlation strength for true signals
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with target and many features, only some correlated
    """
    np.random.seed(random_state)
    
    # Generate target
    target = np.random.normal(0, 1, n_samples)
    data = {'target': target}
    
    # Generate features with true signals
    for i in range(n_true_signals):
        noise_variance = 1 - signal_strength**2
        noise = np.random.normal(0, np.sqrt(noise_variance), n_samples)
        feature = signal_strength * target + noise
        data[f'true_signal_{i+1}'] = feature
    
    # Generate noise features (no true signal)
    for i in range(n_features - n_true_signals):
        feature = np.random.normal(0, 1, n_samples)
        data[f'noise_feature_{i+1}'] = feature
    
    return pd.DataFrame(data)

def generate_categorical_association_data(association_strength: str = 'medium',
                                        n_samples: int = 200,
                                        random_state: int = 42) -> pd.DataFrame:
    """
    Generate categorical data with known association strength
    
    Parameters:
    -----------
    association_strength : str
        Strength of association ('weak', 'medium', 'strong')
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with categorical variables having known association
    """
    np.random.seed(random_state)
    
    # Define probability matrices for different association strengths
    if association_strength == 'weak':
        # Weak association (Cramer's V ≈ 0.1)
        prob_matrix = np.array([
            [0.3, 0.2],
            [0.2, 0.3]
        ])
    elif association_strength == 'medium':
        # Medium association (Cramer's V ≈ 0.3)
        prob_matrix = np.array([
            [0.4, 0.1],
            [0.1, 0.4]
        ])
    elif association_strength == 'strong':
        # Strong association (Cramer's V ≈ 0.6)
        prob_matrix = np.array([
            [0.45, 0.05],
            [0.05, 0.45]
        ])
    else:
        raise ValueError("association_strength must be 'weak', 'medium', or 'strong'")
    
    # Normalize to probabilities
    prob_matrix = prob_matrix / prob_matrix.sum()
    
    # Generate data based on probability matrix
    categories_a = []
    categories_b = []
    
    for _ in range(n_samples):
        # Sample from joint distribution
        flat_probs = prob_matrix.flatten()
        choice = np.random.choice(4, p=flat_probs)
        
        # Convert back to 2D indices
        i, j = divmod(choice, 2)
        categories_a.append(f'A{i+1}')
        categories_b.append(f'B{j+1}')
    
    data = pd.DataFrame({
        'category_a': categories_a,
        'category_b': categories_b
    })
    
    # Add additional variables
    data['id'] = range(n_samples)
    data['random_var'] = np.random.normal(0, 1, n_samples)
    
    return data

def generate_time_series_ab_test(n_days: int = 30, 
                                control_base_rate: float = 0.1,
                                treatment_effect: float = 0.02,
                                trend_effect: float = 0.001,
                                random_state: int = 42) -> pd.DataFrame:
    """
    Generate time series A/B test data with trend
    
    Parameters:
    -----------
    n_days : int
        Number of days of data
    control_base_rate : float
        Base conversion rate for control group
    treatment_effect : float
        Additional conversion rate for treatment
    trend_effect : float
        Daily trend in conversion rate
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with daily A/B test results
    """
    np.random.seed(random_state)
    
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    
    data = []
    for i, date in enumerate(dates):
        # Base rate increases over time (trend)
        day_base_rate = control_base_rate + (trend_effect * i)
        
        # Control group
        control_rate = day_base_rate
        control_visitors = np.random.poisson(1000)  # ~1000 visitors per day
        control_conversions = np.random.binomial(control_visitors, control_rate)
        
        # Treatment group  
        treatment_rate = day_base_rate + treatment_effect
        treatment_visitors = np.random.poisson(1000)
        treatment_conversions = np.random.binomial(treatment_visitors, treatment_rate)
        
        data.extend([
            {
                'date': date,
                'group': 'control',
                'visitors': control_visitors,
                'conversions': control_conversions,
                'conversion_rate': control_conversions / control_visitors if control_visitors > 0 else 0
            },
            {
                'date': date,
                'group': 'treatment', 
                'visitors': treatment_visitors,
                'conversions': treatment_conversions,
                'conversion_rate': treatment_conversions / treatment_visitors if treatment_visitors > 0 else 0
            }
        ])
    
    return pd.DataFrame(data)

def validate_statistical_properties(df: pd.DataFrame, 
                                  expected_properties: Dict) -> Dict[str, bool]:
    """
    Validate that generated data has expected statistical properties
    
    Parameters:
    -----------
    df : pd.DataFrame
        Generated dataset to validate
    expected_properties : Dict
        Dictionary of expected properties to validate
        
    Returns:
    --------
    Dict[str, bool]
        Dictionary indicating whether each property is satisfied
    """
    validation_results = {}
    
    # Check correlations
    if 'correlations' in expected_properties:
        for var1, var2, expected_corr in expected_properties['correlations']:
            if var1 in df.columns and var2 in df.columns:
                actual_corr = df[var1].corr(df[var2])
                tolerance = 0.1  # Allow 10% tolerance
                validation_results[f'correlation_{var1}_{var2}'] = (
                    abs(actual_corr - expected_corr) < tolerance
                )
    
    # Check group means
    if 'group_means' in expected_properties:
        for group_col, outcome_col, expected_means in expected_properties['group_means']:
            if group_col in df.columns and outcome_col in df.columns:
                for group, expected_mean in expected_means.items():
                    group_data = df[df[group_col] == group][outcome_col]
                    if len(group_data) > 0:
                        actual_mean = group_data.mean()
                        tolerance = expected_mean * 0.2  # 20% tolerance
                        validation_results[f'mean_{group}'] = (
                            abs(actual_mean - expected_mean) < tolerance
                        )
    
    # Check sample sizes
    if 'sample_sizes' in expected_properties:
        for group_col, expected_sizes in expected_properties['sample_sizes']:
            if group_col in df.columns:
                actual_sizes = df[group_col].value_counts()
                for group, expected_size in expected_sizes.items():
                    if group in actual_sizes.index:
                        actual_size = actual_sizes[group]
                        tolerance = max(10, expected_size * 0.1)  # 10% tolerance or at least 10
                        validation_results[f'size_{group}'] = (
                            abs(actual_size - expected_size) < tolerance
                        )
    
    return validation_results

if __name__ == "__main__":
    """Generate example datasets and validate their properties"""
    
    print("Generating statistical test datasets...")
    
    # Generate A/B test data
    ab_data = generate_ab_test_data(
        n_control=1000, n_treatment=1000,
        control_rate=0.1, treatment_rate=0.12
    )
    print(f"\nA/B Test Data: {len(ab_data)} rows")
    print(f"Control conversion rate: {ab_data[ab_data['group']=='control']['converted'].mean():.3f}")
    print(f"Treatment conversion rate: {ab_data[ab_data['group']=='treatment']['converted'].mean():.3f}")
    
    # Generate correlation data
    corr_data = generate_correlation_data(n_samples=500)
    print(f"\nCorrelation Data: {len(corr_data)} rows, {len(corr_data.columns)} columns")
    print("Actual correlations with target:")
    for col in corr_data.columns:
        if col != 'target':
            corr = corr_data['target'].corr(corr_data[col])
            print(f"  {col}: {corr:.3f}")
    
    # Generate group effect data
    group_data = generate_group_effect_data(n_per_group=100)
    print(f"\nGroup Effect Data: {len(group_data)} rows")
    print("Group means:")
    for group in group_data['group'].unique():
        mean_outcome = group_data[group_data['group']==group]['outcome'].mean()
        print(f"  {group}: {mean_outcome:.1f}")
    
    # Generate multiple testing data
    mult_test_data = generate_multiple_testing_data(
        n_features=20, n_samples=100, n_true_signals=3
    )
    print(f"\nMultiple Testing Data: {len(mult_test_data)} rows, {len(mult_test_data.columns)} columns")
    print("Correlations with target (first 10 features):")
    for col in list(mult_test_data.columns)[1:11]:  # Skip target, show first 10
        corr = mult_test_data['target'].corr(mult_test_data[col])
        print(f"  {col}: {corr:.3f}")
    
    print("\nDataset generation completed successfully!")