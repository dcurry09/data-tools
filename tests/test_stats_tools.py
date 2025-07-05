import unittest
import numpy as np
import pandas as pd
from scipy import stats
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stats_tools import (
    chi_square_test, compare_two_groups, compare_multiple_groups,
    bootstrap_confidence_interval, proportion_confidence_interval,
    power_analysis_two_groups, minimum_detectable_effect,
    multiple_comparison_correction, cohens_d, cramers_v,
    correlation_with_significance, interpret_p_value, format_p_value
)

class TestHypothesisTesting(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create test data for chi-square test
        self.df_categorical = pd.DataFrame({
            'group': ['A', 'B', 'A', 'B', 'A', 'B'] * 50,
            'outcome': ([0, 1, 0, 0, 1, 1] * 30) + ([1, 1, 0, 1, 1, 0] * 20)
        })
        
        # Create test data for group comparisons
        self.group1 = np.random.normal(10, 2, 50)
        self.group2 = np.random.normal(12, 2, 50)  # Different mean
        self.group3 = np.random.normal(11, 2, 50)
        
        # Data with known effect size (Cohen's d ≈ 1.0)
        self.large_effect_group1 = np.random.normal(0, 1, 30)
        self.large_effect_group2 = np.random.normal(1, 1, 30)
    
    def test_chi_square_test(self):
        """Test chi-square test of independence"""
        result = chi_square_test(self.df_categorical, 'group', 'outcome')
        
        # Check that all expected keys are present
        expected_keys = [
            'chi2_statistic', 'p_value', 'degrees_of_freedom', 'cramers_v',
            'effect_size_interpretation', 'significant', 'alpha'
        ]
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check data types
        self.assertIsInstance(result['chi2_statistic'], (int, float))
        self.assertIsInstance(result['p_value'], (int, float))
        self.assertIsInstance(result['degrees_of_freedom'], int)
        self.assertIsInstance(result['significant'], bool)
        
        # Check that p-value is between 0 and 1
        self.assertGreaterEqual(result['p_value'], 0)
        self.assertLessEqual(result['p_value'], 1)
        
        # Check that Cramer's V is between 0 and 1
        self.assertGreaterEqual(result['cramers_v'], 0)
        self.assertLessEqual(result['cramers_v'], 1)
    
    def test_compare_two_groups_ttest(self):
        """Test two-group comparison with t-test"""
        result = compare_two_groups(self.group1, self.group2, test_type='ttest')
        
        # Check that all expected keys are present
        expected_keys = [
            'test_name', 'statistic', 'p_value', 'significant', 'effect_size',
            'cohens_d', 'group1_mean', 'group2_mean'
        ]
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check that we get a t-test
        self.assertEqual(result['test_name'], "Independent samples t-test")
        
        # Check that Cohen's d is calculated
        self.assertIsNotNone(result['cohens_d'])
        
        # Check that effect size is reasonable
        self.assertGreater(result['effect_size'], 0)
    
    def test_compare_two_groups_mannwhitney(self):
        """Test two-group comparison with Mann-Whitney U test"""
        result = compare_two_groups(self.group1, self.group2, test_type='mannwhitney')
        
        # Check that we get Mann-Whitney test
        self.assertEqual(result['test_name'], "Mann-Whitney U test")
        
        # Check that Cohen's d is None for non-parametric test
        self.assertIsNone(result['cohens_d'])
        
        # Check that effect size is rank-biserial correlation
        self.assertEqual(result['effect_size_name'], "Rank-biserial correlation")
    
    def test_compare_two_groups_auto(self):
        """Test automatic test selection"""
        result = compare_two_groups(self.group1, self.group2, test_type='auto')
        
        # Should choose an appropriate test
        self.assertIn(result['test_name'], 
                     ["Independent samples t-test", "Mann-Whitney U test"])
    
    def test_compare_multiple_groups(self):
        """Test multiple group comparison"""
        groups = [self.group1, self.group2, self.group3]
        result = compare_multiple_groups(groups, post_hoc=True)
        
        # Check that all expected keys are present
        expected_keys = [
            'test_name', 'statistic', 'p_value', 'significant', 'effect_size'
        ]
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check that post-hoc comparisons are included
        if result['significant']:
            self.assertIn('post_hoc_comparisons', result)
            self.assertIsInstance(result['post_hoc_comparisons'], list)
    
    def test_known_effect_size(self):
        """Test with data that has known effect size"""
        result = compare_two_groups(
            self.large_effect_group1, self.large_effect_group2, 
            test_type='ttest'
        )
        
        # Cohen's d should be approximately 1.0 (large effect)
        self.assertGreater(abs(result['cohens_d']), 0.7)  # Allow some variation
        self.assertEqual(result['effect_size_interpretation'], 'large')

class TestConfidenceIntervals(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.normal_data = np.random.normal(10, 2, 100)
        self.success_data = [1] * 80 + [0] * 20  # 80% success rate
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval"""
        result = bootstrap_confidence_interval(
            self.normal_data, 
            statistic_func=np.mean,
            confidence_level=0.95,
            n_bootstrap=1000
        )
        
        # Check that all expected keys are present
        expected_keys = ['statistic', 'lower_ci', 'upper_ci', 'confidence_level']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check that confidence interval makes sense
        self.assertLess(result['lower_ci'], result['statistic'])
        self.assertGreater(result['upper_ci'], result['statistic'])
        
        # Check that statistic is close to expected mean
        self.assertAlmostEqual(result['statistic'], np.mean(self.normal_data), places=3)
    
    def test_proportion_confidence_interval(self):
        """Test proportion confidence interval"""
        successes = 80
        trials = 100
        
        # Test Wilson method (recommended)
        result = proportion_confidence_interval(
            successes, trials, confidence_level=0.95, method='wilson'
        )
        
        # Check that all expected keys are present
        expected_keys = ['proportion', 'lower_ci', 'upper_ci', 'confidence_level', 'method']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check that proportion is correct
        self.assertAlmostEqual(result['proportion'], 0.8, places=3)
        
        # Check that confidence interval makes sense
        self.assertLess(result['lower_ci'], result['proportion'])
        self.assertGreater(result['upper_ci'], result['proportion'])
        
        # Check bounds are within [0, 1]
        self.assertGreaterEqual(result['lower_ci'], 0)
        self.assertLessEqual(result['upper_ci'], 1)
    
    def test_proportion_confidence_interval_edge_cases(self):
        """Test proportion confidence interval edge cases"""
        # Test with 0 successes
        result_zero = proportion_confidence_interval(0, 100, method='exact')
        self.assertEqual(result_zero['proportion'], 0)
        self.assertEqual(result_zero['lower_ci'], 0)
        
        # Test with all successes
        result_all = proportion_confidence_interval(100, 100, method='exact')
        self.assertEqual(result_all['proportion'], 1)
        self.assertEqual(result_all['upper_ci'], 1)

class TestPowerAnalysis(unittest.TestCase):
    
    def test_power_analysis_two_groups(self):
        """Test power analysis for two groups"""
        result = power_analysis_two_groups(
            effect_size=0.5,  # Medium effect size
            alpha=0.05,
            power=0.8
        )
        
        # Check that all expected keys are present
        expected_keys = [
            'n1_required', 'n2_required', 'total_n_required', 
            'effect_size', 'alpha', 'power_requested'
        ]
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check that sample sizes are reasonable for medium effect
        self.assertGreater(result['n1_required'], 30)  # Should need reasonable sample
        self.assertLess(result['n1_required'], 200)    # But not too large
        
        # Check that total sample size is sum of groups
        self.assertEqual(result['total_n_required'], 
                        result['n1_required'] + result['n2_required'])
    
    def test_minimum_detectable_effect(self):
        """Test minimum detectable effect calculation"""
        result = minimum_detectable_effect(
            n1=50, n2=50,
            alpha=0.05,
            power=0.8
        )
        
        # Check that all expected keys are present
        expected_keys = ['minimum_detectable_effect', 'n1', 'n2', 'alpha', 'power']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check that effect size is reasonable
        self.assertGreater(result['minimum_detectable_effect'], 0.3)
        self.assertLess(result['minimum_detectable_effect'], 1.0)

class TestMultipleComparisons(unittest.TestCase):
    
    def setUp(self):
        """Set up test p-values"""
        self.p_values = [0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
    
    def test_bonferroni_correction(self):
        """Test Bonferroni correction"""
        corrected = multiple_comparison_correction(
            self.p_values, method='bonferroni'
        )
        
        # Check that corrected p-values are larger (more conservative)
        for i, p in enumerate(self.p_values):
            self.assertGreaterEqual(corrected[i], p)
        
        # Check that no corrected p-value exceeds 1
        self.assertTrue(all(p <= 1.0 for p in corrected))
    
    def test_holm_correction(self):
        """Test Holm-Bonferroni correction"""
        corrected = multiple_comparison_correction(
            self.p_values, method='holm'
        )
        
        # Check that corrected p-values are larger
        for i, p in enumerate(self.p_values):
            self.assertGreaterEqual(corrected[i], p)
        
        # Holm should be less conservative than Bonferroni
        bonferroni = multiple_comparison_correction(
            self.p_values, method='bonferroni'
        )
        for i in range(len(self.p_values)):
            self.assertLessEqual(corrected[i], bonferroni[i])
    
    def test_fdr_correction(self):
        """Test FDR (Benjamini-Hochberg) correction"""
        corrected = multiple_comparison_correction(
            self.p_values, method='fdr_bh'
        )
        
        # FDR should be less conservative than Bonferroni
        bonferroni = multiple_comparison_correction(
            self.p_values, method='bonferroni'
        )
        for i in range(len(self.p_values)):
            self.assertLessEqual(corrected[i], bonferroni[i])

class TestEffectSizes(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Data with known Cohen's d ≈ 1.0
        self.group1_large_effect = np.random.normal(0, 1, 50)
        self.group2_large_effect = np.random.normal(1, 1, 50)
        
        # Data with no effect
        self.group1_no_effect = np.random.normal(0, 1, 50)
        self.group2_no_effect = np.random.normal(0, 1, 50)
    
    def test_cohens_d(self):
        """Test Cohen's d calculation"""
        # Test large effect
        d_large = cohens_d(self.group1_large_effect, self.group2_large_effect)
        self.assertGreater(abs(d_large), 0.7)  # Should be close to 1.0
        
        # Test no effect
        d_none = cohens_d(self.group1_no_effect, self.group2_no_effect)
        self.assertLess(abs(d_none), 0.3)  # Should be close to 0
    
    def test_cramers_v(self):
        """Test Cramer's V calculation"""
        # Create contingency table with known association
        strong_association = np.array([
            [40, 10],
            [10, 40]
        ])
        
        weak_association = np.array([
            [25, 25],
            [25, 25]
        ])
        
        v_strong = cramers_v(strong_association)
        v_weak = cramers_v(weak_association)
        
        # Strong association should have higher Cramer's V
        self.assertGreater(v_strong, v_weak)
        
        # Both should be between 0 and 1
        self.assertGreaterEqual(v_strong, 0)
        self.assertLessEqual(v_strong, 1)
        self.assertGreaterEqual(v_weak, 0)
        self.assertLessEqual(v_weak, 1)

class TestCorrelationAnalysis(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create data with known correlation
        self.x = np.random.normal(0, 1, 100)
        self.y_strong = 0.8 * self.x + 0.6 * np.random.normal(0, 1, 100)  # r ≈ 0.8
        self.y_none = np.random.normal(0, 1, 100)  # r ≈ 0
    
    def test_correlation_with_significance(self):
        """Test correlation with significance testing"""
        # Test strong correlation
        result_strong = correlation_with_significance(
            self.x, self.y_strong, method='pearson'
        )
        
        # Check that all expected keys are present
        expected_keys = ['correlation', 'p_value', 'significant', 'confidence_interval', 'n']
        for key in expected_keys:
            self.assertIn(key, result_strong)
        
        # Strong correlation should be significant
        self.assertTrue(result_strong['significant'])
        self.assertGreater(abs(result_strong['correlation']), 0.6)
        
        # Test no correlation
        result_none = correlation_with_significance(
            self.x, self.y_none, method='pearson'
        )
        
        # No correlation should not be significant
        self.assertFalse(result_none['significant'])
        self.assertLess(abs(result_none['correlation']), 0.3)

class TestUtilityFunctions(unittest.TestCase):
    
    def test_interpret_p_value(self):
        """Test p-value interpretation"""
        # Very small p-value
        interp_small = interpret_p_value(0.0001)
        self.assertIn("very strong evidence", interp_small)
        
        # Large p-value
        interp_large = interpret_p_value(0.8)
        self.assertIn("insufficient evidence", interp_large)
    
    def test_format_p_value(self):
        """Test p-value formatting"""
        # Very small p-value
        formatted_small = format_p_value(0.0001)
        self.assertEqual(formatted_small, "p < 0.001")
        
        # Regular p-value
        formatted_regular = format_p_value(0.034)
        self.assertEqual(formatted_regular, "p = 0.034")

class TestIntegrationWithRealData(unittest.TestCase):
    
    def setUp(self):
        """Create realistic test dataset"""
        np.random.seed(42)
        n = 200
        
        # Create realistic dataset for A/B test
        self.ab_test_data = pd.DataFrame({
            'group': ['A'] * (n//2) + ['B'] * (n//2),
            'converted': np.concatenate([
                np.random.binomial(1, 0.1, n//2),   # Group A: 10% conversion
                np.random.binomial(1, 0.15, n//2)   # Group B: 15% conversion
            ]),
            'revenue': np.concatenate([
                np.random.gamma(2, 10, n//2),      # Group A: lower revenue
                np.random.gamma(2, 12, n//2)       # Group B: higher revenue
            ])
        })
    
    def test_ab_test_analysis(self):
        """Test complete A/B test analysis"""
        # Test conversion rate difference
        conversion_result = chi_square_test(
            self.ab_test_data, 'group', 'converted'
        )
        
        # Should find some difference (may or may not be significant due to random data)
        self.assertIsInstance(conversion_result['p_value'], (int, float))
        self.assertGreater(conversion_result['cramers_v'], 0)
        
        # Test revenue difference
        group_a_revenue = self.ab_test_data[
            self.ab_test_data['group'] == 'A'
        ]['revenue'].values
        group_b_revenue = self.ab_test_data[
            self.ab_test_data['group'] == 'B'
        ]['revenue'].values
        
        revenue_result = compare_two_groups(group_a_revenue, group_b_revenue)
        
        # Should calculate effect size
        self.assertIsInstance(revenue_result['effect_size'], (int, float))
        self.assertIn(revenue_result['effect_size_interpretation'], 
                     ['negligible', 'small', 'medium', 'large'])

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)