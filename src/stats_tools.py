import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, kruskal, f_oneway
from sklearn.utils import resample
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

# ==================== HYPOTHESIS TESTING ====================

def chi_square_test(df: pd.DataFrame, col1: str, col2: str, 
                   alpha: float = 0.05) -> Dict[str, Any]:
    """
    Perform chi-square test of independence between two categorical variables
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    col1, col2 : str
        Column names for the two categorical variables
    alpha : float
        Significance level (default 0.05)
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with test results including chi2 statistic, p-value, 
        effect size (Cramer's V), and interpretation
    """
    # Create contingency table
    contingency_table = pd.crosstab(df[col1], df[col2])
    
    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Calculate Cramer's V (effect size)
    n = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
    
    # Interpret effect size
    if cramers_v < 0.1:
        effect_interpretation = "negligible"
    elif cramers_v < 0.3:
        effect_interpretation = "small"
    elif cramers_v < 0.5:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    # Check assumptions
    min_expected = expected.min()
    cells_below_5 = (expected < 5).sum()
    total_cells = expected.size
    
    assumptions_met = min_expected >= 1 and (cells_below_5 / total_cells) <= 0.2
    
    return {
        'chi2_statistic': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'cramers_v': cramers_v,
        'effect_size_interpretation': effect_interpretation,
        'significant': bool(p_value < alpha),
        'alpha': alpha,
        'contingency_table': contingency_table,
        'expected_frequencies': expected,
        'assumptions_met': assumptions_met,
        'min_expected_frequency': min_expected,
        'cells_below_5_pct': cells_below_5 / total_cells * 100
    }

def compare_two_groups(group1: np.ndarray, group2: np.ndarray, 
                      test_type: str = 'auto', alpha: float = 0.05,
                      equal_var: bool = True) -> Dict[str, Any]:
    """
    Compare two groups using appropriate statistical test
    
    Parameters:
    -----------
    group1, group2 : np.ndarray
        Data for the two groups to compare
    test_type : str
        Type of test ('auto', 'ttest', 'mannwhitney')
        If 'auto', will choose based on normality tests
    alpha : float
        Significance level
    equal_var : bool
        Whether to assume equal variances for t-test
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with test results and effect sizes
    """
    # Remove missing values
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]
    
    # Check normality if test_type is auto
    if test_type == 'auto':
        if len(group1) >= 8 and len(group2) >= 8:
            _, p1 = stats.shapiro(group1)
            _, p2 = stats.shapiro(group2)
            # Use parametric test if both groups appear normal
            test_type = 'ttest' if (p1 > 0.05 and p2 > 0.05) else 'mannwhitney'
        else:
            # Use non-parametric for small samples
            test_type = 'mannwhitney'
    
    # Perform appropriate test
    if test_type == 'ttest':
        statistic, p_value = ttest_ind(group1, group2, equal_var=equal_var)
        test_name = "Independent samples t-test"
        
        # Calculate Cohen's d (effect size)
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        effect_size = abs(cohens_d)
        effect_size_name = "Cohen's d"
        
    else:  # mannwhitney
        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        test_name = "Mann-Whitney U test"
        
        # Calculate rank-biserial correlation (effect size for Mann-Whitney)
        n1, n2 = len(group1), len(group2)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        effect_size = abs(effect_size)
        effect_size_name = "Rank-biserial correlation"
        cohens_d = None
    
    # Interpret effect size
    if effect_size < 0.1:
        effect_interpretation = "negligible"
    elif effect_size < 0.3:
        effect_interpretation = "small"
    elif effect_size < 0.5:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    return {
        'test_name': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha,
        'alpha': alpha,
        'effect_size': effect_size,
        'effect_size_name': effect_size_name,
        'effect_size_interpretation': effect_interpretation,
        'cohens_d': cohens_d,
        'group1_mean': np.mean(group1),
        'group2_mean': np.mean(group2),
        'group1_median': np.median(group1),
        'group2_median': np.median(group2),
        'group1_n': len(group1),
        'group2_n': len(group2)
    }

def compare_multiple_groups(groups: List[np.ndarray], 
                           test_type: str = 'auto',
                           alpha: float = 0.05,
                           post_hoc: bool = True) -> Dict[str, Any]:
    """
    Compare multiple groups using ANOVA or Kruskal-Wallis test
    
    Parameters:
    -----------
    groups : List[np.ndarray]
        List of arrays, each containing data for one group
    test_type : str
        Type of test ('auto', 'anova', 'kruskal')
    alpha : float
        Significance level
    post_hoc : bool
        Whether to perform post-hoc pairwise comparisons
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with test results and post-hoc comparisons
    """
    # Remove missing values from each group
    groups = [group[~np.isnan(group)] for group in groups]
    
    # Check normality if test_type is auto
    if test_type == 'auto':
        normal_groups = []
        for group in groups:
            if len(group) >= 8:
                _, p_val = stats.shapiro(group)
                normal_groups.append(p_val > 0.05)
            else:
                normal_groups.append(False)
        
        # Use ANOVA if all groups appear normal
        test_type = 'anova' if all(normal_groups) else 'kruskal'
    
    # Perform appropriate test
    if test_type == 'anova':
        statistic, p_value = f_oneway(*groups)
        test_name = "One-way ANOVA"
        
        # Calculate eta-squared (effect size)
        k = len(groups)  # number of groups
        n_total = sum(len(group) for group in groups)
        ss_between = sum(len(group) * (np.mean(group) - np.mean(np.concatenate(groups)))**2 
                        for group in groups)
        ss_total = sum((value - np.mean(np.concatenate(groups)))**2 
                      for group in groups for value in group)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        effect_size_name = "Eta-squared"
        
    else:  # kruskal
        statistic, p_value = kruskal(*groups)
        test_name = "Kruskal-Wallis test"
        
        # Calculate epsilon-squared (effect size for Kruskal-Wallis)
        n_total = sum(len(group) for group in groups)
        eta_squared = (statistic - len(groups) + 1) / (n_total - len(groups))
        effect_size_name = "Epsilon-squared"
    
    # Interpret effect size
    if eta_squared < 0.01:
        effect_interpretation = "negligible"
    elif eta_squared < 0.06:
        effect_interpretation = "small"
    elif eta_squared < 0.14:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    results = {
        'test_name': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha,
        'alpha': alpha,
        'effect_size': eta_squared,
        'effect_size_name': effect_size_name,
        'effect_size_interpretation': effect_interpretation,
        'group_means': [np.mean(group) for group in groups],
        'group_medians': [np.median(group) for group in groups],
        'group_sizes': [len(group) for group in groups]
    }
    
    # Post-hoc pairwise comparisons
    if post_hoc and p_value < alpha:
        pairwise_results = []
        for i, j in combinations(range(len(groups)), 2):
            pair_result = compare_two_groups(
                groups[i], groups[j], 
                test_type='ttest' if test_type == 'anova' else 'mannwhitney',
                alpha=alpha
            )
            pair_result['group_indices'] = (i, j)
            pairwise_results.append(pair_result)
        
        # Apply multiple comparison correction
        p_values = [result['p_value'] for result in pairwise_results]
        corrected_p_values = multiple_comparison_correction(p_values, method='bonferroni')
        
        for i, result in enumerate(pairwise_results):
            result['p_value_corrected'] = corrected_p_values[i]
            result['significant_corrected'] = corrected_p_values[i] < alpha
        
        results['post_hoc_comparisons'] = pairwise_results
    
    return results

# ==================== CONFIDENCE INTERVALS ====================

def bootstrap_confidence_interval(data: np.ndarray, 
                                statistic_func: callable = np.mean,
                                confidence_level: float = 0.95,
                                n_bootstrap: int = 1000,
                                random_state: int = 42) -> Dict[str, float]:
    """
    Calculate bootstrap confidence interval for a statistic
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    statistic_func : callable
        Function to calculate the statistic (default: np.mean)
    confidence_level : float
        Confidence level (default: 0.95)
    n_bootstrap : int
        Number of bootstrap samples
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dict[str, float]
        Dictionary with statistic value and confidence interval
    """
    np.random.seed(random_state)
    
    # Remove missing values
    data = data[~np.isnan(data)]
    
    if len(data) == 0:
        return {
            'statistic': np.nan,
            'lower_ci': np.nan,
            'upper_ci': np.nan,
            'confidence_level': confidence_level
        }
    
    # Original statistic
    original_stat = statistic_func(data)
    
    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        bootstrap_sample = resample(data, random_state=None)
        bootstrap_stats.append(statistic_func(bootstrap_sample))
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_ci = np.percentile(bootstrap_stats, lower_percentile)
    upper_ci = np.percentile(bootstrap_stats, upper_percentile)
    
    return {
        'statistic': original_stat,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'confidence_level': confidence_level,
        'bootstrap_samples': n_bootstrap
    }

def proportion_confidence_interval(successes: int, trials: int,
                                 confidence_level: float = 0.95,
                                 method: str = 'wilson') -> Dict[str, float]:
    """
    Calculate confidence interval for a proportion
    
    Parameters:
    -----------
    successes : int
        Number of successes
    trials : int
        Total number of trials
    confidence_level : float
        Confidence level
    method : str
        Method to use ('wilson', 'wald', 'exact')
        
    Returns:
    --------
    Dict[str, float]
        Dictionary with proportion and confidence interval
    """
    if trials == 0:
        return {
            'proportion': np.nan,
            'lower_ci': np.nan,
            'upper_ci': np.nan,
            'confidence_level': confidence_level,
            'method': method
        }
    
    p = successes / trials
    alpha = 1 - confidence_level
    z = stats.norm.ppf(1 - alpha / 2)
    
    if method == 'wilson':
        # Wilson score interval (recommended)
        n = trials
        x = successes
        denominator = 1 + z**2 / n
        centre = (x + z**2 / 2) / n / denominator
        delta = z / denominator * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        lower_ci = centre - delta
        upper_ci = centre + delta
        
    elif method == 'wald':
        # Wald interval (simple but less accurate for extreme proportions)
        margin_error = z * np.sqrt(p * (1 - p) / trials)
        lower_ci = p - margin_error
        upper_ci = p + margin_error
        
    elif method == 'exact':
        # Exact binomial interval (Clopper-Pearson)
        lower_ci = stats.beta.ppf(alpha / 2, successes, trials - successes + 1)
        upper_ci = stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
        
        # Handle edge cases
        if successes == 0:
            lower_ci = 0
        if successes == trials:
            upper_ci = 1
    
    # Ensure bounds are within [0, 1]
    lower_ci = max(0, lower_ci)
    upper_ci = min(1, upper_ci)
    
    return {
        'proportion': p,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'confidence_level': confidence_level,
        'method': method,
        'successes': successes,
        'trials': trials
    }

# ==================== POWER ANALYSIS ====================

def power_analysis_two_groups(effect_size: float, 
                             alpha: float = 0.05,
                             power: float = 0.8,
                             ratio: float = 1.0,
                             alternative: str = 'two-sided') -> Dict[str, Any]:
    """
    Calculate sample size needed for two-group comparison
    
    Parameters:
    -----------
    effect_size : float
        Expected effect size (Cohen's d)
    alpha : float
        Type I error rate
    power : float
        Desired statistical power (1 - Type II error rate)
    ratio : float
        Ratio of sample sizes (n2/n1)
    alternative : str
        Type of test ('two-sided', 'one-sided')
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with required sample sizes and analysis details
    """
    try:
        from statsmodels.stats.power import ttest_power
        from statsmodels.stats.power import tt_solve_power
        
        # Calculate required sample size
        # Note: statsmodels uses 'nobs' for sample size calculation
        n1 = tt_solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            alternative=alternative
        )
        
        n2 = int(np.ceil(n1 * ratio))
        total_n = int(np.ceil(n1)) + n2
        
        # Calculate achieved power with recommended sample sizes
        achieved_power = ttest_power(
            effect_size=effect_size,
            nobs=n1,
            alpha=alpha,
            alternative=alternative
        )
        
        return {
            'n1_required': int(np.ceil(n1)),
            'n2_required': int(np.ceil(n2)),
            'total_n_required': int(np.ceil(total_n)),
            'effect_size': effect_size,
            'alpha': alpha,
            'power_requested': power,
            'power_achieved': achieved_power,
            'ratio': ratio,
            'alternative': alternative
        }
        
    except ImportError:
        # Fallback calculation without statsmodels
        print("Warning: statsmodels not available. Using approximate calculation.")
        
        # Approximate calculation
        if alternative == 'two-sided':
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        z_beta = stats.norm.ppf(power)
        
        # Sample size calculation
        n1 = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        n2 = n1 * ratio
        total_n = n1 + n2
        
        return {
            'n1_required': int(np.ceil(n1)),
            'n2_required': int(np.ceil(n2)),
            'total_n_required': int(np.ceil(total_n)),
            'effect_size': effect_size,
            'alpha': alpha,
            'power_requested': power,
            'power_achieved': None,  # Cannot calculate without statsmodels
            'ratio': ratio,
            'alternative': alternative,
            'note': 'Approximate calculation - install statsmodels for exact results'
        }

def minimum_detectable_effect(n1: int, n2: int = None,
                            alpha: float = 0.05,
                            power: float = 0.8,
                            alternative: str = 'two-sided') -> Dict[str, Any]:
    """
    Calculate minimum detectable effect size given sample sizes
    
    Parameters:
    -----------
    n1, n2 : int
        Sample sizes for groups 1 and 2 (if n2 is None, assumes equal sizes)
    alpha : float
        Type I error rate
    power : float
        Statistical power
    alternative : str
        Type of test ('two-sided', 'one-sided')
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with minimum detectable effect size
    """
    if n2 is None:
        n2 = n1
    
    try:
        from statsmodels.stats.power import tt_solve_power
        
        effect_size = tt_solve_power(
            nobs=n1,
            power=power,
            alpha=alpha,
            alternative=alternative
        )
        
        return {
            'minimum_detectable_effect': effect_size,
            'n1': n1,
            'n2': n2,
            'alpha': alpha,
            'power': power,
            'alternative': alternative
        }
        
    except ImportError:
        print("Warning: statsmodels not available. Using approximate calculation.")
        
        # Approximate calculation
        if alternative == 'two-sided':
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        z_beta = stats.norm.ppf(power)
        
        # Effect size calculation
        effect_size = (z_alpha + z_beta) * np.sqrt(2 / ((n1 * n2) / (n1 + n2)))
        
        return {
            'minimum_detectable_effect': effect_size,
            'n1': n1,
            'n2': n2,
            'alpha': alpha,
            'power': power,
            'alternative': alternative,
            'note': 'Approximate calculation - install statsmodels for exact results'
        }

# ==================== MULTIPLE COMPARISONS ====================

def multiple_comparison_correction(p_values: List[float], 
                                 method: str = 'bonferroni',
                                 alpha: float = 0.05) -> np.ndarray:
    """
    Apply multiple comparison correction to p-values
    
    Parameters:
    -----------
    p_values : List[float]
        List of uncorrected p-values
    method : str
        Correction method ('bonferroni', 'holm', 'fdr_bh', 'fdr_by')
    alpha : float
        Family-wise error rate
        
    Returns:
    --------
    np.ndarray
        Array of corrected p-values
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    if method == 'bonferroni':
        # Bonferroni correction
        corrected = p_values * n
        corrected = np.minimum(corrected, 1.0)
        
    elif method == 'holm':
        # Holm-Bonferroni correction
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        corrected_sorted = np.zeros_like(sorted_p)
        for i in range(n):
            corrected_sorted[i] = sorted_p[i] * (n - i)
        
        # Ensure monotonicity
        for i in range(1, n):
            corrected_sorted[i] = max(corrected_sorted[i], corrected_sorted[i-1])
        
        # Reorder back to original order
        corrected = np.zeros_like(p_values)
        corrected[sorted_indices] = corrected_sorted
        corrected = np.minimum(corrected, 1.0)
        
    elif method == 'fdr_bh':
        # Benjamini-Hochberg FDR correction
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        corrected_sorted = np.zeros_like(sorted_p)
        for i in range(n-1, -1, -1):
            if i == n-1:
                corrected_sorted[i] = sorted_p[i]
            else:
                corrected_sorted[i] = min(corrected_sorted[i+1], 
                                        sorted_p[i] * n / (i + 1))
        
        # Reorder back to original order
        corrected = np.zeros_like(p_values)
        corrected[sorted_indices] = corrected_sorted
        corrected = np.minimum(corrected, 1.0)
        
    elif method == 'fdr_by':
        # Benjamini-Yekutieli FDR correction
        c_n = np.sum(1.0 / np.arange(1, n + 1))
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        corrected_sorted = np.zeros_like(sorted_p)
        for i in range(n-1, -1, -1):
            if i == n-1:
                corrected_sorted[i] = sorted_p[i] * c_n
            else:
                corrected_sorted[i] = min(corrected_sorted[i+1], 
                                        sorted_p[i] * n * c_n / (i + 1))
        
        # Reorder back to original order
        corrected = np.zeros_like(p_values)
        corrected[sorted_indices] = corrected_sorted
        corrected = np.minimum(corrected, 1.0)
        
    else:
        raise ValueError(f"Unknown correction method: {method}")
    
    return corrected

# ==================== EFFECT SIZE CALCULATIONS ====================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size
    
    Parameters:
    -----------
    group1, group2 : np.ndarray
        Data for the two groups
        
    Returns:
    --------
    float
        Cohen's d effect size
    """
    # Remove missing values
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]
    
    if len(group1) == 0 or len(group2) == 0:
        return np.nan
    
    # Calculate means
    mean1, mean2 = np.mean(group1), np.mean(group2)
    
    # Calculate pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                         (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    if pooled_std == 0:
        return 0
    
    return (mean1 - mean2) / pooled_std

def cramers_v(confusion_matrix: np.ndarray) -> float:
    """
    Calculate Cramer's V effect size for categorical association
    
    Parameters:
    -----------
    confusion_matrix : np.ndarray
        Contingency table
        
    Returns:
    --------
    float
        Cramer's V effect size
    """
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    min_dim = min(confusion_matrix.shape) - 1
    
    if n == 0 or min_dim == 0:
        return 0
    
    return np.sqrt(chi2 / (n * min_dim))

# ==================== CORRELATION ANALYSIS ====================

def correlation_with_significance(x: np.ndarray, y: np.ndarray,
                                method: str = 'pearson',
                                alpha: float = 0.05) -> Dict[str, Any]:
    """
    Calculate correlation with significance test
    
    Parameters:
    -----------
    x, y : np.ndarray
        Input variables
    method : str
        Correlation method ('pearson', 'spearman', 'kendall')
    alpha : float
        Significance level
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with correlation and significance test results
    """
    # Remove missing values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 3:
        return {
            'correlation': np.nan,
            'p_value': np.nan,
            'significant': False,
            'n': len(x_clean),
            'method': method
        }
    
    # Calculate correlation and p-value
    if method == 'pearson':
        correlation, p_value = stats.pearsonr(x_clean, y_clean)
    elif method == 'spearman':
        correlation, p_value = stats.spearmanr(x_clean, y_clean)
    elif method == 'kendall':
        correlation, p_value = stats.kendalltau(x_clean, y_clean)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    # Calculate confidence interval (for Pearson only)
    confidence_interval = None
    if method == 'pearson' and len(x_clean) > 3:
        # Fisher z-transformation for confidence interval
        z = np.arctanh(correlation)
        se = 1 / np.sqrt(len(x_clean) - 3)
        z_critical = stats.norm.ppf(1 - alpha / 2)
        
        z_lower = z - z_critical * se
        z_upper = z + z_critical * se
        
        confidence_interval = (np.tanh(z_lower), np.tanh(z_upper))
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'significant': p_value < alpha,
        'alpha': alpha,
        'confidence_interval': confidence_interval,
        'n': len(x_clean),
        'method': method
    }

# ==================== UTILITY FUNCTIONS ====================

def interpret_p_value(p_value: float, alpha: float = 0.05) -> str:
    """
    Provide interpretation of p-value
    
    Parameters:
    -----------
    p_value : float
        P-value to interpret
    alpha : float
        Significance threshold
        
    Returns:
    --------
    str
        Interpretation of the p-value
    """
    if p_value < 0.001:
        return "very strong evidence against null hypothesis (p < 0.001)"
    elif p_value < 0.01:
        return "strong evidence against null hypothesis (p < 0.01)"
    elif p_value < alpha:
        return f"moderate evidence against null hypothesis (p < {alpha})"
    elif p_value < 0.1:
        return "weak evidence against null hypothesis (p < 0.1)"
    else:
        return "insufficient evidence against null hypothesis"

def format_p_value(p_value: float) -> str:
    """
    Format p-value for reporting
    
    Parameters:
    -----------
    p_value : float
        P-value to format
        
    Returns:
    --------
    str
        Formatted p-value string
    """
    if p_value < 0.001:
        return "p < 0.001"
    else:
        return f"p = {p_value:.3f}"

if __name__ == "__main__":
    print("Statistical Tools Module loaded successfully!")
    print("\nAvailable functions:")
    print("Hypothesis Testing:")
    print("- chi_square_test(df, col1, col2)")
    print("- compare_two_groups(group1, group2)")
    print("- compare_multiple_groups(groups)")
    print("\nConfidence Intervals:")
    print("- bootstrap_confidence_interval(data)")
    print("- proportion_confidence_interval(successes, trials)")
    print("\nPower Analysis:")
    print("- power_analysis_two_groups(effect_size)")
    print("- minimum_detectable_effect(n1, n2)")
    print("\nMultiple Comparisons:")
    print("- multiple_comparison_correction(p_values)")
    print("\nEffect Sizes:")
    print("- cohens_d(group1, group2)")
    print("- cramers_v(confusion_matrix)")
    print("\nCorrelation:")
    print("- correlation_with_significance(x, y)")