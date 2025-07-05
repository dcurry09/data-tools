#!/usr/bin/env python3
"""
Demo of Statistical Rigor Enhancements to Data Tools Toolkit

This script demonstrates the enhanced statistical capabilities added to the data tools toolkit.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
from data_tools import group_analyze, find_signal_strength
from generate_test_data import generate_test_data
import stats_tools

def main():
    print("="*70)
    print("DATA TOOLS TOOLKIT - STATISTICAL RIGOR ENHANCEMENT DEMO")
    print("="*70)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # ==================== DEMO 1: A/B Test Analysis ====================
    print("\n" + "="*50)
    print("DEMO 1: A/B TEST ANALYSIS WITH STATISTICAL RIGOR")
    print("="*50)
    
    # Generate A/B test data
    n_control = 1000
    n_treatment = 1000
    
    ab_test_data = pd.DataFrame({
        'group': ['control'] * n_control + ['treatment'] * n_treatment,
        'converted': np.concatenate([
            np.random.binomial(1, 0.10, n_control),    # Control: 10% conversion
            np.random.binomial(1, 0.12, n_treatment)   # Treatment: 12% conversion (20% lift)
        ]),
        'revenue': np.concatenate([
            np.random.gamma(2, 25, n_control),         # Control group revenue
            np.random.gamma(2, 30, n_treatment)        # Treatment group revenue  
        ])
    })
    
    print(f"Generated A/B test data: {len(ab_test_data)} observations")
    print(f"Control group size: {n_control}, Treatment group size: {n_treatment}")
    
    # Analyze conversion rates with statistical tests
    print("\nAnalyzing conversion rates with statistical significance testing...")
    conversion_result = group_analyze(
        ab_test_data, 
        'group', 
        'converted',
        statistical_tests=True,
        alpha=0.05,
        confidence_level=0.95
    )
    
    print(f"\nConversion rate analysis completed!")
    if conversion_result['test_results']:
        if 'chi_square' in conversion_result['test_results']:
            chi2 = conversion_result['test_results']['chi_square']
            print(f"Statistical significance: p = {chi2['p_value']:.4f}")
            print(f"Effect size (Cramer's V): {chi2['cramers_v']:.3f}")
            print(f"Result: {'Significant' if chi2['significant'] else 'Not significant'} difference")
    
    # ==================== DEMO 2: Feature Signal Strength Analysis ====================
    print("\n" + "="*50)
    print("DEMO 2: FEATURE SIGNAL STRENGTH WITH STATISTICAL TESTING")
    print("="*50)
    
    # Generate dataset with known signal strengths
    n_samples = 500
    signal_data = pd.DataFrame({
        'target': np.random.binomial(1, 0.3, n_samples)
    })
    
    # Add features with different signal strengths
    signal_data['strong_predictor'] = (
        signal_data['target'] * 2.5 + np.random.normal(0, 0.8, n_samples)
    )
    
    signal_data['moderate_predictor'] = (
        signal_data['target'] * 1.2 + np.random.normal(0, 1.2, n_samples)
    )
    
    signal_data['weak_predictor'] = (
        signal_data['target'] * 0.4 + np.random.normal(0, 1.8, n_samples)
    )
    
    signal_data['noise_feature'] = np.random.normal(0, 1, n_samples)
    
    print(f"Generated feature dataset: {len(signal_data)} observations, {len(signal_data.columns)} features")
    
    # Analyze signal strength with statistical significance
    print("\nAnalyzing feature signal strength with statistical significance testing...")
    signal_results = find_signal_strength(
        signal_data,
        'target',
        statistical_tests=True,
        alpha=0.05
    )
    
    print(f"\nFeature analysis completed! Top 3 features by signal strength:")
    for i, (_, row) in enumerate(signal_results.head(3).iterrows()):
        feature = row['column']
        correlation = row['correlation']
        significant = row.get('correlation_significant', 'N/A')
        p_value = row.get('correlation_p_value', 'N/A')
        
        print(f"{i+1}. {feature}:")
        print(f"   Correlation: {correlation:.3f}")
        print(f"   P-value: {p_value:.2e}" if isinstance(p_value, float) else f"   P-value: {p_value}")
        print(f"   Significant: {significant}")
    
    # ==================== DEMO 3: Group Comparison with Multiple Testing ====================
    print("\n" + "="*50)
    print("DEMO 3: MULTIPLE GROUP COMPARISON WITH POST-HOC TESTING")
    print("="*50)
    
    # Generate multi-group experiment data
    n_per_group = 150
    multi_group_data = pd.DataFrame({
        'treatment': ['control'] * n_per_group + 
                    ['treatment_a'] * n_per_group + 
                    ['treatment_b'] * n_per_group + 
                    ['treatment_c'] * n_per_group,
        'outcome': np.concatenate([
            np.random.normal(50, 10, n_per_group),    # Control: mean 50
            np.random.normal(55, 10, n_per_group),    # Treatment A: mean 55 (medium effect)
            np.random.normal(60, 10, n_per_group),    # Treatment B: mean 60 (large effect)
            np.random.normal(52, 10, n_per_group)     # Treatment C: mean 52 (small effect)
        ])
    })
    
    print(f"Generated multi-group experiment: {len(multi_group_data)} observations, 4 groups")
    
    # Analyze with multiple comparison correction
    print("\nAnalyzing multiple groups with post-hoc testing and Bonferroni correction...")
    multi_group_result = group_analyze(
        multi_group_data,
        'treatment',
        'outcome',
        statistical_tests=True,
        alpha=0.05
    )
    
    print(f"\nMultiple group analysis completed!")
    if multi_group_result['test_results']:
        if 'multiple_group_comparison' in multi_group_result['test_results']:
            comp = multi_group_result['test_results']['multiple_group_comparison']
            print(f"Overall test: {comp['test_name']}")
            print(f"Overall p-value: {comp['p_value']:.4f}")
            print(f"Overall effect size: {comp['effect_size']:.3f}")
            
            if 'post_hoc_comparisons' in comp:
                print(f"\nPost-hoc pairwise comparisons (Bonferroni corrected):")
                group_names = list(multi_group_result['grouped_results'].index)
                for pair_comp in comp['post_hoc_comparisons']:
                    g1, g2 = pair_comp['group_indices']
                    p_corrected = pair_comp['p_value_corrected']
                    significant = pair_comp['significant_corrected']
                    print(f"  {group_names[g1]} vs {group_names[g2]}: p = {p_corrected:.4f} ({'significant' if significant else 'not significant'})")
    
    # ==================== DEMO 4: Power Analysis ====================
    print("\n" + "="*50)
    print("DEMO 4: POWER ANALYSIS FOR EXPERIMENT PLANNING")
    print("="*50)
    
    print("Calculating required sample sizes for different effect sizes...")
    
    effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large effects
    for effect_size in effect_sizes:
        power_result = stats_tools.power_analysis_two_groups(
            effect_size=effect_size,
            alpha=0.05,
            power=0.8
        )
        
        print(f"\nFor Cohen's d = {effect_size} ({'small' if effect_size == 0.2 else 'medium' if effect_size == 0.5 else 'large'} effect):")
        print(f"  Required sample size per group: {power_result['n1_required']}")
        print(f"  Total required sample size: {power_result['total_n_required']}")
    
    # Calculate minimum detectable effect for a fixed sample size
    print(f"\nFor a fixed sample size of 100 per group:")
    mde_result = stats_tools.minimum_detectable_effect(n1=100, n2=100, power=0.8, alpha=0.05)
    print(f"  Minimum detectable effect size: {mde_result['minimum_detectable_effect']:.3f}")
    
    # ==================== DEMO 5: Confidence Intervals ====================
    print("\n" + "="*50)
    print("DEMO 5: CONFIDENCE INTERVALS FOR ROBUST ESTIMATION")
    print("="*50)
    
    # Generate sample data
    sample_data = np.random.normal(100, 15, 200)
    
    print(f"Sample data: n = {len(sample_data)}, mean = {np.mean(sample_data):.2f}")
    
    # Bootstrap confidence interval for mean
    ci_result = stats_tools.bootstrap_confidence_interval(
        sample_data,
        statistic_func=np.mean,
        confidence_level=0.95,
        n_bootstrap=1000
    )
    
    print(f"\n95% Bootstrap Confidence Interval for mean:")
    print(f"  Mean: {ci_result['statistic']:.2f}")
    print(f"  95% CI: [{ci_result['lower_ci']:.2f}, {ci_result['upper_ci']:.2f}]")
    
    # Proportion confidence interval
    successes = 85
    trials = 200
    
    prop_ci = stats_tools.proportion_confidence_interval(
        successes, trials, confidence_level=0.95, method='wilson'
    )
    
    print(f"\n95% Confidence Interval for proportion:")
    print(f"  Proportion: {prop_ci['proportion']:.3f} ({successes}/{trials})")
    print(f"  95% CI: [{prop_ci['lower_ci']:.3f}, {prop_ci['upper_ci']:.3f}]")
    
    print("\n" + "="*70)
    print("STATISTICAL RIGOR ENHANCEMENT DEMO COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print(f"\nKey enhancements added to the toolkit:")
    print(f"✓ Hypothesis testing (t-tests, ANOVA, chi-square)")
    print(f"✓ Effect size calculations (Cohen's d, Cramer's V)")
    print(f"✓ Confidence intervals (bootstrap, proportion)")
    print(f"✓ Power analysis and sample size calculations")
    print(f"✓ Multiple comparison corrections")
    print(f"✓ Statistical significance testing for correlations")
    print(f"✓ Proper interpretation and reporting of results")
    
    print(f"\nAll functions now provide:")
    print(f"• Statistical significance tests")
    print(f"• Effect size estimates with interpretation")
    print(f"• Confidence intervals for key metrics")
    print(f"• Proper handling of multiple comparisons")
    print(f"• Clear interpretation of statistical results")

if __name__ == "__main__":
    main()