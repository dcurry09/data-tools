"""
Data Science Toolkit for Interview
Author: [Your Name]
Date: May 2025
Description: Comprehensive toolkit for data discovery, cleaning, exploration, modeling prep, and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# ==================== DATA DISCOVERY FUNCTIONS ====================

def quick_look(df, n_samples=5):
    """
    Quick overview of dataset including shape, dtypes, missing values, and sample data
    """
    print("="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    print("\nData Types:")
    print(df.dtypes.value_counts())
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage': missing_pct})
    print(missing_df[missing_df['Missing Count'] > 0].sort_values('Percentage', ascending=False))
    print("\nFirst few rows:")
    display(df.head(n_samples))
    print("\nRandom sample:")
    display(df.sample(n_samples))
    print("\nDescriptive statistics:")
    display(df.describe())

def profile_columns(df):
    """
    Detailed profiling of each column including unique values, data types, and distributions
    """
    profile = []
    
    for col in df.columns:
        col_info = {
            'Column': col,
            'Type': df[col].dtype,
            'Non-Null Count': df[col].count(),
            'Null Count': df[col].isnull().sum(),
            'Null %': 100 * df[col].isnull().sum() / len(df),
            'Unique Values': df[col].nunique(),
            'Unique %': 100 * df[col].nunique() / df[col].count() if df[col].count() > 0 else 0
        }
        
        if df[col].dtype in ['int64', 'float64']:
            col_info.update({
                'Mean': df[col].mean(),
                'Std': df[col].std(),
                'Min': df[col].min(),
                'Max': df[col].max()
            })
        
        if df[col].dtype == 'object':
            col_info['Top Value'] = df[col].value_counts().index[0] if len(df[col].value_counts()) > 0 else None
            col_info['Top Value Freq'] = df[col].value_counts().values[0] if len(df[col].value_counts()) > 0 else None
        
        profile.append(col_info)
    
    return pd.DataFrame(profile)

# ==================== DATA CLEANING FUNCTIONS ====================

def handle_missing_values(df, strategy='auto', threshold=0.5):
    """
    Handle missing values with various strategies
    
    Parameters:
    - strategy: 'auto', 'drop', 'mean', 'median', 'mode', 'forward_fill'
    - threshold: if missing > threshold, drop column (only for 'auto')
    """
    df_clean = df.copy()
    
    if strategy == 'auto':
        # Drop columns with too many missing values
        missing_pct = df_clean.isnull().sum() / len(df_clean)
        cols_to_drop = missing_pct[missing_pct > threshold].index
        df_clean = df_clean.drop(columns=cols_to_drop)
        print(f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing values")
        
        # Fill remaining missing values based on data type
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['int64', 'float64']:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else:
                    df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown', inplace=True)
    
    elif strategy == 'drop':
        df_clean = df_clean.dropna()
    
    elif strategy in ['mean', 'median']:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy=strategy)
        df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    
    elif strategy == 'mode':
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                df_clean[col].fillna(mode_val, inplace=True)
    
    elif strategy == 'forward_fill':
        df_clean = df_clean.fillna(method='ffill')
    
    return df_clean

def remove_outliers(df, columns=None, method='iqr', threshold=3):
    """
    Remove outliers using IQR or Z-score method
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    outliers_removed = 0
    
    for col in columns:
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            mask = z_scores < threshold
            mask = pd.Series(mask, index=df_clean[col].dropna().index)
            mask = mask.reindex(df_clean.index, fill_value=True)
        
        outliers_removed += (~mask).sum()
        df_clean = df_clean[mask]
    
    print(f"Removed {outliers_removed} outliers using {method} method")
    return df_clean

def clean_column_names(df):
    """
    Clean and standardize column names
    """
    df_clean = df.copy()
    df_clean.columns = (df_clean.columns
                       .str.lower()
                       .str.replace(' ', '_')
                       .str.replace('[^a-zA-Z0-9_]', '', regex=True))
    return df_clean

# ==================== DATA EXPLORATION FUNCTIONS ====================

def explore_relationships(df, target_col=None, plot=True):
    """
    Explore relationships between features and optionally with a target variable
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    if plot:
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    # If target column specified, show correlations with target
    if target_col and target_col in numeric_cols:
        target_corr = corr_matrix[target_col].sort_values(ascending=False)
        print(f"\nCorrelations with {target_col}:")
        print(target_corr)
        
        if plot:
            plt.figure(figsize=(10, 6))
            target_corr[1:].plot(kind='barh')
            plt.title(f'Feature Correlations with {target_col}')
            plt.xlabel('Correlation Coefficient')
            plt.tight_layout()
            plt.show()
    
    return corr_matrix

def plot_distributions(df, columns=None, n_cols=3):
    """
    Plot distributions of numeric columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    n_plots = len(columns)
    n_rows = (n_plots - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        df[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        
        # Add mean and median lines
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        ax.legend()
    
    # Hide empty subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def analyze_categorical(df, columns=None, top_n=10, plot=True):
    """
    Analyze categorical variables
    """
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns
    
    for col in columns:
        print(f"\n{'='*50}")
        print(f"Analysis of {col}")
        print(f"{'='*50}")
        
        value_counts = df[col].value_counts()
        print(f"Unique values: {df[col].nunique()}")
        print(f"\nTop {top_n} values:")
        print(value_counts.head(top_n))
        
        if plot and len(value_counts) <= 20:
            plt.figure(figsize=(10, 6))
            value_counts.head(top_n).plot(kind='bar')
            plt.title(f'Top {top_n} values in {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

# ==================== DATA TRANSFORMATION FUNCTIONS ====================

def create_quantile_bins(df, column, n_quantiles=10, labels=None):
    """
    Create quantile bins for a numeric column
    
    Parameters:
    - column: Column to bin
    - n_quantiles: Number of quantiles (10 for deciles, 4 for quartiles, etc.)
    - labels: Optional labels for bins
    """
    df_copy = df.copy()
    
    if labels is None:
        labels = [f'Q{i+1}' for i in range(n_quantiles)]
    
    # Create quantile bins
    df_copy[f'{column}_quantile'] = pd.qcut(
        df_copy[column], 
        q=n_quantiles, 
        labels=labels, 
        duplicates='drop'
    )
    
    # Also create numeric quantile for easier aggregation
    df_copy[f'{column}_quantile_num'] = pd.qcut(
        df_copy[column], 
        q=n_quantiles, 
        labels=range(1, n_quantiles + 1), 
        duplicates='drop'
    )
    
    return df_copy

def create_custom_bins(df, column, bins, labels=None):
    """
    Create custom bins for a numeric column
    
    Parameters:
    - column: Column to bin
    - bins: List of bin edges or number of equal-width bins
    - labels: Optional labels for bins
    """
    df_copy = df.copy()
    
    if labels is None and isinstance(bins, list):
        labels = [f'Bin_{i+1}' for i in range(len(bins) - 1)]
    
    df_copy[f'{column}_bin'] = pd.cut(
        df_copy[column], 
        bins=bins, 
        labels=labels,
        include_lowest=True
    )
    
    return df_copy

def analyze_by_quantiles(df, value_column, target_column, n_quantiles=10):
    """
    Analyze target rates by quantiles of a value column
    
    Returns a DataFrame with quantile statistics
    """
    df_quantiled = create_quantile_bins(df, value_column, n_quantiles)
    
    # Calculate statistics by quantile
    analysis = df_quantiled.groupby(f'{value_column}_quantile').agg({
        target_column: ['count', 'sum', 'mean'],
        value_column: ['min', 'max', 'mean']
    }).round(4)
    
    # Flatten column names
    analysis.columns = ['_'.join(col).strip() for col in analysis.columns.values]
    
    # Add percentage of total
    analysis['pct_of_total'] = 100 * analysis[f'{target_column}_count'] / len(df)
    
    # Rename columns for clarity
    analysis.rename(columns={
        f'{target_column}_count': 'count',
        f'{target_column}_sum': 'positive_cases',
        f'{target_column}_mean': 'positive_rate',
        f'{value_column}_min': 'min_value',
        f'{value_column}_max': 'max_value',
        f'{value_column}_mean': 'mean_value'
    }, inplace=True)
    
    return analysis

def plot_quantile_analysis(df, value_column, target_column, n_quantiles=10):
    """
    Plot target rates by quantiles with a dual-axis chart
    """
    analysis = analyze_by_quantiles(df, value_column, target_column, n_quantiles)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Bar plot for counts
    x = range(len(analysis))
    ax1.bar(x, analysis['count'], alpha=0.7, color='lightblue', label='Count')
    ax1.set_xlabel(f'{value_column} Quantiles')
    ax1.set_ylabel('Count', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(analysis.index, rotation=45)
    
    # Line plot for positive rate
    ax2 = ax1.twinx()
    ax2.plot(x, analysis['positive_rate'], color='red', marker='o', linewidth=2, markersize=8, label='Positive Rate')
    ax2.set_ylabel('Positive Rate', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, max(analysis['positive_rate']) * 1.1)
    
    # Add horizontal line for overall rate
    overall_rate = df[target_column].mean()
    ax2.axhline(y=overall_rate, color='green', linestyle='--', linewidth=2, label=f'Overall Rate ({overall_rate:.3f})')
    
    plt.title(f'{target_column} Rate by {value_column} Quantiles')
    fig.tight_layout()
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.show()
    
    return analysis

def find_signal_strength(df, target_column, exclude_columns=None):
    """
    Calculate signal strength (predictive power) for all numeric columns
    """
    if exclude_columns is None:
        exclude_columns = []
    
    exclude_columns.append(target_column)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in exclude_columns]
    
    signal_strength = []
    
    for col in numeric_cols:
        try:
            # Skip columns with too many missing values
            if df[col].isnull().sum() / len(df) > 0.5:
                continue
                
            # Calculate correlation
            correlation = df[col].corr(df[target_column])
            
            # Calculate information value (simplified)
            analysis = analyze_by_quantiles(df.dropna(subset=[col]), col, target_column, n_quantiles=10)
            
            # Calculate max lift
            max_lift = (analysis['positive_rate'].max() / df[target_column].mean()) - 1
            
            # Calculate Gini coefficient
            sorted_df = df.sort_values(by=col)
            cumsum_target = sorted_df[target_column].cumsum() / sorted_df[target_column].sum()
            cumsum_pop = np.arange(1, len(sorted_df) + 1) / len(sorted_df)
            gini = 2 * (cumsum_pop * cumsum_target).sum() / len(sorted_df) - 1
            
            signal_strength.append({
                'column': col,
                'correlation': correlation,
                'max_lift': max_lift,
                'gini': gini,
                'missing_pct': df[col].isnull().sum() / len(df)
            })
            
        except Exception as e:
            print(f"Error processing {col}: {e}")
            continue
    
    signal_df = pd.DataFrame(signal_strength).sort_values('gini', ascending=False, key=abs)
    return signal_df

# ==================== GROUPING AND AGGREGATION FUNCTIONS ====================

def group_analyze(df, group_col, target_col, numeric_cols=None, show_lift=True, top_n=None):
    """
    Group data and analyze target rates with additional metrics
    
    Parameters:
    - group_col: Column to group by
    - target_col: Binary target column
    - numeric_cols: List of numeric columns to include stats for
    - show_lift: Whether to show lift vs overall rate
    - top_n: Show only top N groups by size
    """
    # Calculate overall rate for lift calculation
    overall_rate = df[target_col].mean()
    
    # Basic aggregation
    agg_dict = {
        target_col: ['count', 'sum', 'mean'],
    }
    
    # Add numeric columns if specified
    if numeric_cols:
        for col in numeric_cols:
            if col in df.columns and col != target_col:
                agg_dict[col] = ['mean', 'std']
    
    # Perform groupby
    grouped = df.groupby(group_col).agg(agg_dict)
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    
    # Rename columns for clarity
    grouped.rename(columns={
        f'{target_col}_count': 'count',
        f'{target_col}_sum': 'positives',
        f'{target_col}_mean': 'rate'
    }, inplace=True)
    
    # Add additional metrics
    grouped['pct_of_total'] = 100 * grouped['count'] / len(df)
    grouped['pct_of_positives'] = 100 * grouped['positives'] / df[target_col].sum()
    
    if show_lift:
        grouped['lift'] = grouped['rate'] / overall_rate
        grouped['lift_pct'] = 100 * (grouped['lift'] - 1)
    
    # Sort by count descending
    grouped = grouped.sort_values('count', ascending=False)
    
    # Limit to top N if specified
    if top_n:
        grouped = grouped.head(top_n)
    
    # Format for display
    display_df = grouped.copy()
    
    # Format percentage columns
    pct_cols = ['rate', 'pct_of_total', 'pct_of_positives']
    for col in pct_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f'{x:.2%}')
    
    if show_lift:
        display_df['lift'] = display_df['lift'].apply(lambda x: f'{x:.3f}')
        display_df['lift_pct'] = display_df['lift_pct'].apply(lambda x: f'{x:+.1f}%')
    
    # Round numeric columns
    for col in display_df.columns:
        if display_df[col].dtype in ['float64', 'float32']:
            display_df[col] = display_df[col].round(2)
    
    # Create table
    table = tabulate(display_df, headers='keys', tablefmt='grid', floatfmt='.3f')
    
    print(f"\nAnalysis of {target_col} by {group_col}")
    print(f"Overall {target_col} rate: {overall_rate:.2%}")
    print(table)
    
    return grouped

def group_compare(df, group_col1, group_col2, target_col, min_size=30):
    """
    Compare target rates across two categorical variables
    
    Creates a pivot table showing rates for each combination
    """
    # Create pivot table
    pivot = pd.crosstab(
        df[group_col1], 
        df[group_col2], 
        values=df[target_col], 
        aggfunc=['count', 'mean']
    )
    
    # Get count and rate tables
    count_table = pivot['count'].fillna(0).astype(int)
    rate_table = pivot['mean'].fillna(0)
    
    # Filter out small groups
    mask = count_table >= min_size
    rate_table = rate_table.where(mask, '')
    
    # Format rates as percentages
    rate_display = rate_table.applymap(lambda x: f'{x:.2%}' if x != '' else '')
    
    # Add row and column totals
    row_totals = df.groupby(group_col1)[target_col].agg(['count', 'mean'])
    col_totals = df.groupby(group_col2)[target_col].agg(['count', 'mean'])
    
    # Display count table
    print(f"\nCounts: {group_col1} vs {group_col2}")
    count_display = count_table.copy()
    count_display['Total'] = count_table.sum(axis=1)
    count_display.loc['Total'] = count_table.sum(axis=0)
    count_display.loc['Total', 'Total'] = count_table.sum().sum()
    print(tabulate(count_display, headers='keys', tablefmt='grid'))
    
    # Display rate table
    print(f"\n{target_col} Rates: {group_col1} vs {group_col2}")
    rate_display['Overall'] = row_totals['mean'].apply(lambda x: f'{x:.2%}')
    
    # Add overall column rates
    overall_row = pd.Series(index=rate_display.columns)
    for col in rate_display.columns[:-1]:  # Exclude 'Overall' column
        if col in col_totals.index:
            overall_row[col] = f"{col_totals.loc[col, 'mean']:.2%}"
    overall_row['Overall'] = f"{df[target_col].mean():.2%}"
    
    rate_display.loc['Overall'] = overall_row
    
    print(tabulate(rate_display, headers='keys', tablefmt='grid'))
    
    return pivot

def multi_group_analyze(df, group_cols, target_col, sort_by='count', top_n=20):
    """
    Analyze target rates by multiple grouping columns
    
    Parameters:
    - group_cols: List of columns to group by
    - target_col: Binary target column
    - sort_by: Column to sort by ('count', 'rate', 'lift')
    - top_n: Show top N combinations
    """
    overall_rate = df[target_col].mean()
    
    # Group by multiple columns
    grouped = df.groupby(group_cols).agg({
        target_col: ['count', 'sum', 'mean']
    })
    
    # Flatten column names
    grouped.columns = ['count', 'positives', 'rate']
    
    # Add metrics
    grouped['pct_of_total'] = 100 * grouped['count'] / len(df)
    grouped['lift'] = grouped['rate'] / overall_rate
    
    # Sort and limit
    grouped = grouped.sort_values(sort_by, ascending=False).head(top_n)
    
    # Format for display
    display_df = grouped.copy()
    display_df['rate'] = display_df['rate'].apply(lambda x: f'{x:.2%}')
    display_df['pct_of_total'] = display_df['pct_of_total'].apply(lambda x: f'{x:.1f}%')
    display_df['lift'] = display_df['lift'].apply(lambda x: f'{x:.3f}')
    
    # Reset index for better display
    display_df = display_df.reset_index()
    
    print(f"\nTop {top_n} combinations by {sort_by}")
    print(f"Overall {target_col} rate: {overall_rate:.2%}")
    print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
    
    return grouped

def segment_profiler(df, segment_col, segment_value, compare_cols=None, target_col=None):
    """
    Profile a specific segment compared to the overall population
    
    Parameters:
    - segment_col: Column that defines segments
    - segment_value: Specific segment value to analyze
    - compare_cols: Columns to compare (if None, uses all numeric)
    - target_col: Optional target column for rate comparison
    """
    # Get segment and rest of population
    segment_mask = df[segment_col] == segment_value
    segment_df = df[segment_mask]
    rest_df = df[~segment_mask]
    
    segment_size = len(segment_df)
    total_size = len(df)
    segment_pct = 100 * segment_size / total_size
    
    print(f"\nSegment Profile: {segment_col} = {segment_value}")
    print(f"Segment size: {segment_size:,} ({segment_pct:.1f}% of total)")
    print("="*50)
    
    # If no columns specified, use all numeric columns
    if compare_cols is None:
        compare_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in compare_cols:
            compare_cols.remove(target_col)
    
    # Build comparison table
    comparison_data = []
    
    for col in compare_cols:
        if col not in df.columns:
            continue
            
        # Calculate metrics
        segment_mean = segment_df[col].mean()
        overall_mean = df[col].mean()
        rest_mean = rest_df[col].mean()
        
        # Calculate lift vs overall and rest
        lift_vs_overall = (segment_mean / overall_mean) - 1 if overall_mean != 0 else 0
        lift_vs_rest = (segment_mean / rest_mean) - 1 if rest_mean != 0 else 0
        
        comparison_data.append({
            'Column': col,
            'Segment': f'{segment_mean:.3f}',
            'Overall': f'{overall_mean:.3f}',
            'Rest': f'{rest_mean:.3f}',
            'Lift vs Overall': f'{lift_vs_overall:+.1%}',
            'Lift vs Rest': f'{lift_vs_rest:+.1%}'
        })
    
    # Add target column analysis if specified
    if target_col and target_col in df.columns:
        segment_rate = segment_df[target_col].mean()
        overall_rate = df[target_col].mean()
        rest_rate = rest_df[target_col].mean()
        
        comparison_data.append({
            'Column': f'{target_col} (rate)',
            'Segment': f'{segment_rate:.2%}',
            'Overall': f'{overall_rate:.2%}',
            'Rest': f'{rest_rate:.2%}',
            'Lift vs Overall': f'{(segment_rate/overall_rate - 1):+.1%}',
            'Lift vs Rest': f'{(segment_rate/rest_rate - 1):+.1%}'
        })
    
    # Display comparison table
    comparison_df = pd.DataFrame(comparison_data)
    print(tabulate(comparison_df, headers='keys', tablefmt='grid', showindex=False))
    
    # Categorical variables comparison
    cat_cols = df.select_dtypes(include=['object']).columns
    
    if len(cat_cols) > 0:
        print("\nCategorical Variables Distribution:")
        cat_comparison = []
        
        for col in cat_cols:
            if col == segment_col:
                continue
                
            # Get top 5 values for each
            segment_top5 = segment_df[col].value_counts(normalize=True).head(5)
            overall_top5 = df[col].value_counts(normalize=True).head(5)
            
            # Get unique values in segment top 5
            for value in segment_top5.index[:3]:  # Show top 3
                segment_pct = segment_top5.get(value, 0) * 100
                overall_pct = overall_top5.get(value, 0) * 100
                lift = (segment_pct / overall_pct - 1) if overall_pct > 0 else 0
                
                cat_comparison.append({
                    'Column': col,
                    'Value': str(value)[:30],  # Truncate long values
                    'Segment %': f'{segment_pct:.1f}%',
                    'Overall %': f'{overall_pct:.1f}%',
                    'Lift': f'{lift:+.1%}'
                })
        
        if cat_comparison:
            cat_df = pd.DataFrame(cat_comparison)
            print(tabulate(cat_df, headers='keys', tablefmt='grid', showindex=False))
    
    return segment_df

# ==================== GROUPING AND AGGREGATION FUNCTIONS ====================

# ==================== MODEL PREPARATION FUNCTIONS ====================

def prepare_features(df, target_col, test_size=0.2, random_state=42, scale=True):
    """
    Prepare features for modeling including encoding and scaling
    """
    df_prep = df.copy()
    
    # Separate features and target
    X = df_prep.drop(columns=[target_col])
    y = df_prep[target_col]
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if X[col].nunique() <= 10:  # One-hot encode if few unique values
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
        else:  # Label encode if many unique values
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.dtype == 'object' else None
    )
    
    # Scale features
    if scale:
        scaler = StandardScaler()
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    return X_train, X_test, y_train, y_test, None

def create_feature_importance_plot(feature_names, importances, top_n=20):
    """
    Create a feature importance plot
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# ==================== VISUALIZATION FUNCTIONS ====================

def create_dashboard(df, target_col=None):
    """
    Create a comprehensive dashboard with multiple visualizations
    """
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Missing values heatmap
    ax1 = plt.subplot(3, 3, 1)
    missing_df = df.isnull().sum().sort_values(ascending=False)
    missing_df = missing_df[missing_df > 0]
    if len(missing_df) > 0:
        missing_df.plot(kind='barh', ax=ax1)
        ax1.set_title('Missing Values by Column')
        ax1.set_xlabel('Count')
    else:
        ax1.text(0.5, 0.5, 'No missing values', ha='center', va='center')
        ax1.set_title('Missing Values')
    
    # 2. Data types distribution
    ax2 = plt.subplot(3, 3, 2)
    dtype_counts = df.dtypes.value_counts()
    dtype_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
    ax2.set_title('Data Types Distribution')
    ax2.set_ylabel('')
    
    # 3. Numeric distributions
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]
    for i, col in enumerate(numeric_cols):
        ax = plt.subplot(3, 3, 3 + i)
        df[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
        ax.set_title(f'{col} Distribution')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    
    # 4. Correlation heatmap (if numeric columns exist)
    if len(df.select_dtypes(include=[np.number]).columns) > 1:
        ax7 = plt.subplot(3, 3, 7)
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax7, cbar_kws={'shrink': 0.8})
        ax7.set_title('Correlation Heatmap')
    
    # 5. Target distribution (if specified)
    if target_col and target_col in df.columns:
        ax8 = plt.subplot(3, 3, 8)
        if df[target_col].dtype in ['int64', 'float64']:
            df[target_col].hist(bins=30, ax=ax8, edgecolor='black', alpha=0.7)
        else:
            df[target_col].value_counts().plot(kind='bar', ax=ax8)
        ax8.set_title(f'{target_col} Distribution')
        ax8.set_xlabel(target_col)
        ax8.set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

def plot_learning_curves(train_scores, val_scores, title='Learning Curves'):
    """
    Plot learning curves for model evaluation
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_scores) + 1)
    
    plt.plot(epochs, train_scores, 'b-', label='Training Score')
    plt.plot(epochs, val_scores, 'r-', label='Validation Score')
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==================== MAIN ANALYSIS FUNCTION ====================

def perform_eda(df, target_col=None, clean_data=True):
    """
    Perform complete EDA workflow
    """
    print("Starting Exploratory Data Analysis...")
    
    # 1. Data Discovery
    print("\n1. DATA DISCOVERY")
    quick_look(df)
    
    # 2. Data Cleaning (if requested)
    if clean_data:
        print("\n2. DATA CLEANING")
        df_clean = handle_missing_values(df, strategy='auto')
        df_clean = clean_column_names(df_clean)
        print(f"Cleaned data shape: {df_clean.shape}")
    else:
        df_clean = df
    
    # 3. Data Exploration
    print("\n3. DATA EXPLORATION")
    
    # Profile columns
    print("\nColumn Profiling:")
    profile_df = profile_columns(df_clean)
    display(profile_df)
    
    # Explore relationships
    print("\nExploring Relationships:")
    explore_relationships(df_clean, target_col=target_col)
    
    # Analyze categorical variables
    print("\nCategorical Analysis:")
    analyze_categorical(df_clean)
    
    # 4. Visualizations
    print("\n4. VISUALIZATIONS")
    
    # Create dashboard
    create_dashboard(df_clean, target_col=target_col)
    
    # Plot distributions
    plot_distributions(df_clean)
    
    return df_clean

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example usage - uncomment to test
    
    # # Load your data
    # df = pd.read_csv('test_data.csv')
    
    # # Perform complete EDA
    # df_clean = perform_eda(df, target_col='target', clean_data=True)
    
    # # Group analysis examples
    # # Simple group analysis
    # group_analyze(df_clean, 'membership_level', 'target')
    
    # # Group analysis with numeric columns
    # group_analyze(df_clean, 'state', 'target', numeric_cols=['annual_income', 'age'], top_n=10)
    
    # # Compare two categorical variables
    # group_compare(df_clean, 'gender', 'membership_level', 'target')
    
    # # Multi-group analysis
    # multi_group_analyze(df_clean, ['gender', 'membership_level'], 'target', sort_by='rate')
    
    # # Profile a specific segment
    # segment_profiler(df_clean, 'membership_level', 'Gold', target_col='target')
    
    # # Create quantile bins for analysis
    # df_quantiled = create_quantile_bins(df_clean, 'annual_income', n_quantiles=10)
    
    # # Analyze target rates by quantiles
    # income_analysis = analyze_by_quantiles(df_clean, 'annual_income', 'target', n_quantiles=10)
    # print(income_analysis)
    
    # # Plot quantile analysis
    # plot_quantile_analysis(df_clean, 'annual_income', 'target', n_quantiles=10)
    
    # # Find signal strength of all features
    # signal_strength = find_signal_strength(df_clean, 'target')
    # print(signal_strength)
    
    # # Prepare for modeling
    # X_train, X_test, y_train, y_test, scaler = prepare_features(
    #     df_clean, 
    #     target_col='target',
    #     test_size=0.2,
    #     scale=True
    # )
    
    print("Data Science Toolkit loaded successfully!")
    print("Available functions:")
    print("\nData Discovery & Cleaning:")
    print("- quick_look(df)")
    print("- profile_columns(df)")
    print("- handle_missing_values(df)")
    print("- remove_outliers(df)")
    print("- clean_column_names(df)")
    
    print("\nExploration & Analysis:")
    print("- explore_relationships(df)")
    print("- plot_distributions(df)")
    print("- analyze_categorical(df)")
    
    print("\nGrouping & Aggregation:")
    print("- group_analyze(df, group_col, target_col)")
    print("- group_compare(df, col1, col2, target_col)")
    print("- multi_group_analyze(df, group_cols, target_col)")
    print("- segment_profiler(df, segment_col, segment_value)")
    
    print("\nQuantile Analysis:")
    print("- create_quantile_bins(df, column)")
    print("- analyze_by_quantiles(df, value_column, target_column)")
    print("- plot_quantile_analysis(df, value_column, target_column)")
    print("- find_signal_strength(df, target_column)")
    
    print("\nModel Preparation:")
    print("- prepare_features(df, target_col)")
    
    print("\nVisualization:")
    print("- create_dashboard(df)")
    
    print("\nMain Workflow:")
    print("- perform_eda(df, target_col)")
