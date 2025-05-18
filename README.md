# Data Science Toolkit

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/data-science-toolkit.git
cd data-science-toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter notebook
jupyter notebook
```

## 📁 Project Structure

```
data-science-toolkit/
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── src/
│   └── ds_toolkit.py       # Main toolkit functions
├── notebooks/
│   └── demo.ipynb          # Example usage notebook
├── data/
│   ├── generate_test_data.py  # Test data generator
│   └── test_data.csv          # Sample dataset
└── tests/
    └── test_toolkit.py        # Unit tests (optional)
```

## 🛠️ Features

### Data Discovery & Cleaning
- `quick_look()` - Dataset overview with shape, dtypes, missing values
- `profile_columns()` - Detailed column profiling
- `handle_missing_values()` - Multiple imputation strategies
- `remove_outliers()` - IQR and Z-score methods
- `clean_column_names()` - Standardize naming conventions

### Data Analysis & Exploration
- `explore_relationships()` - Correlation analysis with visualizations
- `analyze_categorical()` - Categorical variable analysis
- `plot_distributions()` - Distribution plots for numeric features

### Grouping & Segmentation
- `group_analyze()` - Target rate analysis by groups
- `group_compare()` - Cross-tabulation analysis
- `multi_group_analyze()` - Multi-dimensional grouping
- `segment_profiler()` - Deep segment profiling

### Feature Engineering
- `create_quantile_bins()` - Deciles, quartiles, custom quantiles
- `analyze_by_quantiles()` - Target analysis by quantiles
- `find_signal_strength()` - Feature importance ranking
- `prepare_features()` - Complete feature preparation pipeline

### Visualization
- `create_dashboard()` - Comprehensive visual dashboard
- `plot_quantile_analysis()` - Quantile performance visualization
- `plot_learning_curves()` - Model evaluation plots

## 📊 Example Usage

```python
import pandas as pd
from ds_toolkit import *

# Load data
df = pd.read_csv('data/test_data.csv')

# Quick overview
quick_look(df)

# Analyze target rates by group
group_analyze(df, 'membership_level', 'target')

# Find feature importance
signal_strength = find_signal_strength(df, 'target')

# Full EDA pipeline
df_clean = perform_eda(df, target_col='target')

# Prepare for modeling
X_train, X_test, y_train, y_test, scaler = prepare_features(
    df_clean, 
    target_col='target',
    test_size=0.2
)
```

## 🧪 Testing

The toolkit includes a test data generator that creates realistic datasets with:
- 500 rows with multiple data types
- Edge cases (missing values, outliers, high cardinality)
- Binary target variable with ~20% positive rate
- Various distributions and correlations

Generate test data:
```python
python data/generate_test_data.py
```

## 📋 Requirements

- Python 3.8+
- Core dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn
- Additional: tabulate for formatted table output
- See `requirements.txt` for complete list