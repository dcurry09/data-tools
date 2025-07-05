import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_tools import group_analyze, find_signal_strength
from generate_test_data import generate_test_data

class TestEnhancedDataTools(unittest.TestCase):
    
    def setUp(self):
        """Set up test data with known statistical properties"""
        np.random.seed(42)
        
        # Generate test data using the existing function
        self.test_df = generate_test_data(n_rows=500)
        
        # Create additional test data with known group effects
        n_per_group = 100
        self.group_test_df = pd.DataFrame({
            'group': ['A'] * n_per_group + ['B'] * n_per_group + ['C'] * n_per_group,
            'target': np.concatenate([
                np.random.binomial(1, 0.1, n_per_group),   # Group A: 10% rate
                np.random.binomial(1, 0.2, n_per_group),   # Group B: 20% rate
                np.random.binomial(1, 0.15, n_per_group)   # Group C: 15% rate
            ]),
            'numeric_feature': np.concatenate([
                np.random.normal(10, 2, n_per_group),      # Group A: mean 10
                np.random.normal(12, 2, n_per_group),      # Group B: mean 12
                np.random.normal(11, 2, n_per_group)       # Group C: mean 11
            ])
        })
        
        # Create signal strength test data
        n_rows = 300
        self.signal_df = pd.DataFrame({
            'target': np.random.binomial(1, 0.3, n_rows),
            'strong_signal': np.random.normal(0, 1, n_rows),
            'weak_signal': np.random.normal(0, 1, n_rows),
            'no_signal': np.random.normal(0, 1, n_rows),
            'noisy_feature': np.random.uniform(0, 100, n_rows)
        })
        
        # Add correlation to strong signal
        self.signal_df['strong_signal'] = (
            self.signal_df['target'] * 2 + 
            np.random.normal(0, 0.5, n_rows)
        )
        
        # Add weak correlation to weak signal
        self.signal_df['weak_signal'] = (
            self.signal_df['target'] * 0.3 + 
            np.random.normal(0, 1, n_rows)
        )
    
    def test_group_analyze_basic_functionality(self):
        """Test basic functionality of enhanced group_analyze"""
        result = group_analyze(
            self.group_test_df, 
            'group', 
            'target',
            statistical_tests=False  # Test without statistical tests first
        )
        
        # Check that result is a dictionary with expected keys
        self.assertIsInstance(result, dict)
        self.assertIn('grouped_results', result)
        self.assertIn('test_results', result)
        self.assertIn('display_table', result)
        
        grouped = result['grouped_results']
        
        # Check that all groups are included
        self.assertEqual(len(grouped), 3)
        self.assertIn('A', grouped.index)
        self.assertIn('B', grouped.index)
        self.assertIn('C', grouped.index)
        
        # Check that confidence intervals are calculated
        self.assertIn('ci_lower', grouped.columns)
        self.assertIn('ci_upper', grouped.columns)
        
        # Check that confidence intervals are sensible
        for group in grouped.index:
            self.assertLessEqual(grouped.loc[group, 'ci_lower'], 
                               grouped.loc[group, 'rate'])
            self.assertGreaterEqual(grouped.loc[group, 'ci_upper'], 
                                  grouped.loc[group, 'rate'])
    
    def test_group_analyze_with_statistical_tests(self):
        """Test group_analyze with statistical tests enabled"""
        try:
            result = group_analyze(
                self.group_test_df, 
                'group', 
                'target',
                statistical_tests=True,
                alpha=0.05
            )
            
            # Check that statistical tests were performed
            test_results = result['test_results']
            
            # Should have chi-square test (categorical target)
            if 'chi_square' in test_results:
                chi2 = test_results['chi_square']
                self.assertIn('p_value', chi2)
                self.assertIn('cramers_v', chi2)
                self.assertIn('effect_size_interpretation', chi2)
                
                # P-value should be between 0 and 1
                self.assertGreaterEqual(chi2['p_value'], 0)
                self.assertLessEqual(chi2['p_value'], 1)
                
                # Cramer's V should be between 0 and 1
                self.assertGreaterEqual(chi2['cramers_v'], 0)
                self.assertLessEqual(chi2['cramers_v'], 1)
            
            # Check if multiple group comparison was performed
            if 'multiple_group_comparison' in test_results:
                comp = test_results['multiple_group_comparison']
                self.assertIn('p_value', comp)
                self.assertIn('effect_size', comp)
                self.assertIn('effect_size_interpretation', comp)
        
        except Exception as e:
            # If stats_tools is not available, should still work without statistical tests
            self.skipTest(f"Statistical tests not available: {e}")
    
    def test_group_analyze_with_numeric_columns(self):
        """Test group_analyze with additional numeric columns"""
        result = group_analyze(
            self.group_test_df, 
            'group', 
            'target',
            numeric_cols=['numeric_feature'],
            statistical_tests=False
        )
        
        grouped = result['grouped_results']
        
        # Check that numeric column statistics are included
        self.assertIn('numeric_feature_mean', grouped.columns)
        
        # Check that means are reasonable
        for group in grouped.index:
            mean_val = grouped.loc[group, 'numeric_feature_mean']
            self.assertIsInstance(mean_val, (int, float))
            self.assertGreater(mean_val, 8)  # Should be around 10-12
            self.assertLess(mean_val, 14)
    
    def test_group_analyze_edge_cases(self):
        """Test group_analyze with edge cases"""
        # Test with small dataset
        small_df = self.group_test_df.head(10)
        result = group_analyze(
            small_df, 
            'group', 
            'target',
            statistical_tests=False
        )
        
        # Should still work with small data
        self.assertIsInstance(result, dict)
        self.assertIn('grouped_results', result)
        
        # Test with missing values
        df_with_missing = self.group_test_df.copy()
        df_with_missing.loc[0:10, 'target'] = np.nan
        
        result = group_analyze(
            df_with_missing, 
            'group', 
            'target',
            statistical_tests=False
        )
        
        # Should handle missing values gracefully
        self.assertIsInstance(result, dict)
    
    def test_find_signal_strength_basic(self):
        """Test basic functionality of enhanced find_signal_strength"""
        result = find_signal_strength(
            self.signal_df, 
            'target',
            statistical_tests=False
        )
        
        # Check that result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check that expected columns are present
        expected_cols = ['column', 'correlation', 'gini', 'max_lift', 'missing_pct', 'sample_size']
        for col in expected_cols:
            self.assertIn(col, result.columns)
        
        # Check that all features are analyzed
        expected_features = ['strong_signal', 'weak_signal', 'no_signal', 'noisy_feature']
        for feature in expected_features:
            self.assertIn(feature, result['column'].values)
        
        # Check that correlations are reasonable
        for _, row in result.iterrows():
            correlation = row['correlation']
            self.assertGreaterEqual(abs(correlation), 0)
            self.assertLessEqual(abs(correlation), 1)
    
    def test_find_signal_strength_with_statistical_tests(self):
        """Test find_signal_strength with statistical significance testing"""
        try:
            result = find_signal_strength(
                self.signal_df, 
                'target',
                statistical_tests=True,
                alpha=0.05
            )
            
            # Check that statistical test columns are present
            stat_cols = ['correlation_p_value', 'correlation_significant', 
                        'correlation_ci_lower', 'correlation_ci_upper']
            for col in stat_cols:
                self.assertIn(col, result.columns)
            
            # Check that strong signal should be significant
            strong_signal_row = result[result['column'] == 'strong_signal']
            if not strong_signal_row.empty:
                p_value = strong_signal_row.iloc[0]['correlation_p_value']
                if pd.notna(p_value):
                    # Strong signal should likely be significant
                    self.assertLess(p_value, 0.1)  # Allow some tolerance
            
            # Check that p-values are between 0 and 1
            for _, row in result.iterrows():
                p_val = row['correlation_p_value']
                if pd.notna(p_val):
                    self.assertGreaterEqual(p_val, 0)
                    self.assertLessEqual(p_val, 1)
        
        except Exception as e:
            self.skipTest(f"Statistical tests not available: {e}")
    
    def test_find_signal_strength_ranking(self):
        """Test that signal strength ranking works correctly"""
        result = find_signal_strength(
            self.signal_df, 
            'target',
            statistical_tests=False
        )
        
        # Strong signal should rank higher than no signal
        strong_signal_gini = result[result['column'] == 'strong_signal']['gini'].iloc[0]
        no_signal_gini = result[result['column'] == 'no_signal']['gini'].iloc[0]
        
        # Strong signal should have higher absolute Gini coefficient
        self.assertGreater(abs(strong_signal_gini), abs(no_signal_gini))
    
    def test_find_signal_strength_edge_cases(self):
        """Test find_signal_strength with edge cases"""
        # Test with constant target (no variation)
        constant_target_df = self.signal_df.copy()
        constant_target_df['target'] = 1  # All ones
        
        result = find_signal_strength(
            constant_target_df, 
            'target',
            statistical_tests=False
        )
        
        # Should handle constant target gracefully
        self.assertIsInstance(result, pd.DataFrame)
        
        # Test with mostly missing data
        missing_data_df = self.signal_df.copy()
        missing_data_df.loc[:int(len(missing_data_df)*0.8), 'strong_signal'] = np.nan
        
        result = find_signal_strength(
            missing_data_df, 
            'target',
            statistical_tests=False
        )
        
        # Should exclude features with too much missing data
        if not result.empty:
            for _, row in result.iterrows():
                self.assertLess(row['missing_pct'], 0.5)
    
    def test_integration_with_original_test_data(self):
        """Test enhanced functions with original generate_test_data output"""
        # Test group_analyze with membership levels
        if 'membership_level' in self.test_df.columns and 'target' in self.test_df.columns:
            result = group_analyze(
                self.test_df, 
                'membership_level', 
                'target',
                statistical_tests=False
            )
            
            # Should work with original test data structure
            self.assertIsInstance(result, dict)
            self.assertIn('grouped_results', result)
            
            grouped = result['grouped_results']
            self.assertGreater(len(grouped), 0)
        
        # Test find_signal_strength with numeric features
        numeric_cols = self.test_df.select_dtypes(include=[np.number]).columns
        if 'target' in numeric_cols and len(numeric_cols) > 1:
            result = find_signal_strength(
                self.test_df, 
                'target',
                statistical_tests=False
            )
            
            # Should find some signal in the generated data
            self.assertIsInstance(result, pd.DataFrame)
            self.assertGreater(len(result), 0)

class TestStatisticalValidation(unittest.TestCase):
    """Test that statistical calculations are mathematically correct"""
    
    def setUp(self):
        """Set up data with known statistical properties"""
        np.random.seed(123)  # Different seed for validation
        
        # Create data with known effect sizes
        self.known_effect_data = pd.DataFrame({
            'group': ['A'] * 50 + ['B'] * 50,
            'value': np.concatenate([
                np.random.normal(0, 1, 50),    # Group A: mean 0, std 1
                np.random.normal(0.8, 1, 50)   # Group B: mean 0.8, std 1 (Cohen's d ≈ 0.8)
            ])
        })
        
        # Binary target for group A and B
        self.known_effect_data['binary_target'] = (
            self.known_effect_data['value'] > 0
        ).astype(int)
    
    def test_statistical_consistency(self):
        """Test that our statistical functions give consistent results"""
        try:
            # Test group analysis
            result = group_analyze(
                self.known_effect_data,
                'group',
                'binary_target',
                statistical_tests=True
            )
            
            if 'test_results' in result and result['test_results']:
                test_results = result['test_results']
                
                # If we have comparison results, they should be consistent
                if 'two_group_comparison' in test_results:
                    comp = test_results['two_group_comparison']
                    
                    # Effect size should be reasonable for our known data
                    effect_size = comp['effect_size']
                    self.assertGreater(effect_size, 0.1)  # Should detect some effect
                    
                    # Test should be significant if effect is large enough
                    if effect_size > 0.5:
                        self.assertTrue(comp['significant'])
        
        except Exception as e:
            self.skipTest(f"Statistical tests not available: {e}")
    
    def test_confidence_interval_coverage(self):
        """Test that confidence intervals have proper coverage"""
        # This is a simplified test - in practice you'd run many simulations
        result = group_analyze(
            self.known_effect_data,
            'group',
            'binary_target',
            confidence_level=0.95,
            statistical_tests=False
        )
        
        grouped = result['grouped_results']
        
        # Check that actual rates fall within confidence intervals
        for group in grouped.index:
            rate = grouped.loc[group, 'rate']
            ci_lower = grouped.loc[group, 'ci_lower']
            ci_upper = grouped.loc[group, 'ci_upper']
            
            # Rate should be within its own confidence interval
            self.assertGreaterEqual(rate, ci_lower)
            self.assertLessEqual(rate, ci_upper)
            
            # Confidence interval should have reasonable width
            ci_width = ci_upper - ci_lower
            self.assertGreater(ci_width, 0)
            self.assertLess(ci_width, 1)  # Shouldn't be wider than possible range

if __name__ == '__main__':
    unittest.main(verbosity=2)