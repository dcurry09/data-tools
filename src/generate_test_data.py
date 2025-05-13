"""
Generate Test Data for Data Science Toolkit
Author: [Your Name]
Date: May 2025
Description: Creates a comprehensive test dataset with various data types and edge cases
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import string

# Set random seed for reproducibility
np.random.seed(42)

def generate_test_data(n_rows=500, target_rate=0.2):
    """
    Generate test data with various column types and edge cases
    
    Parameters:
    - n_rows: Number of rows to generate
    - target_rate: Rate of positive cases for binary target
    """
    
    # Initialize data dictionary
    data = {}
    
    # 1. Binary target variable (dependent on some features)
    # We'll create this last based on other features
    
    # 2. Numeric features with different characteristics
    # Clean numeric
    data['age'] = np.random.normal(35, 12, n_rows).clip(18, 80).astype(int)
    
    # Numeric with outliers
    income_base = np.random.lognormal(10.5, 0.8, n_rows)
    outlier_mask = np.random.rand(n_rows) < 0.05
    data['annual_income'] = np.where(outlier_mask, income_base * 10, income_base)
    
    # Numeric with missing values
    score = np.random.normal(50, 15, n_rows)
    missing_mask = np.random.rand(n_rows) < 0.1
    data['credit_score'] = np.where(missing_mask, np.nan, score)
    
    # Highly skewed numeric
    data['transaction_amount'] = np.random.exponential(100, n_rows)
    
    # Numeric with many zeros (sparse)
    sparse_data = np.random.normal(0, 1, n_rows)
    zero_mask = np.random.rand(n_rows) < 0.7
    data['product_views'] = np.where(zero_mask, 0, np.abs(sparse_data * 100))
    
    # 3. Categorical features
    # Low cardinality categorical
    data['gender'] = np.random.choice(['M', 'F', 'Other'], n_rows, p=[0.45, 0.45, 0.1])
    
    # Medium cardinality categorical
    states = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
    data['state'] = np.random.choice(states, n_rows)
    
    # High cardinality categorical
    data['customer_id'] = ['CUST' + str(i).zfill(6) for i in range(n_rows)]
    
    # Categorical with missing values
    categories = ['Bronze', 'Silver', 'Gold', 'Platinum']
    cat_data = np.random.choice(categories, n_rows, p=[0.4, 0.3, 0.2, 0.1])
    missing_mask = np.random.rand(n_rows) < 0.08
    data['membership_level'] = np.where(missing_mask, np.nan, cat_data)
    
    # 4. Date/time features
    start_date = datetime(2020, 1, 1)
    date_range = (datetime(2024, 12, 31) - start_date).days
    random_days = np.random.randint(0, date_range, n_rows)
    data['account_created'] = [start_date + timedelta(days=int(d)) for d in random_days]
    
    # 5. Boolean features
    data['is_active'] = np.random.choice([True, False], n_rows, p=[0.7, 0.3])
    data['has_discount'] = np.random.choice([True, False], n_rows, p=[0.3, 0.7])
    
    # 6. Text features (for testing string operations)
    first_names = ['John', 'Jane', 'Bob', 'Alice', 'Charlie', 'Diana', 'Eve', 'Frank']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Davis', 'Miller', 'Wilson']
    data['customer_name'] = [
        f"{np.random.choice(first_names)} {np.random.choice(last_names)}" 
        for _ in range(n_rows)
    ]
    
    # 7. Mixed type column (numbers stored as strings)
    data['zip_code'] = [str(np.random.randint(10000, 99999)) for _ in range(n_rows)]
    
    # 8. Column with special characters
    special_chars = ['Product_A', 'Product-B', 'Product C', 'Product/D', 'Product@E']
    data['product_type'] = np.random.choice(special_chars, n_rows)
    
    # 9. Highly correlated features
    data['feature_a'] = np.random.normal(100, 20, n_rows)
    data['feature_b'] = data['feature_a'] * 0.8 + np.random.normal(0, 10, n_rows)
    
    # 10. Create target variable based on other features
    # Target is influenced by age, income, credit_score, and membership_level
    target_prob = np.zeros(n_rows)
    
    # Age influence
    target_prob += (data['age'] > 30).astype(float) * 0.1
    target_prob += (data['age'] < 50).astype(float) * 0.1
    
    # Income influence
    income_norm = (data['annual_income'] - np.min(data['annual_income'])) / (np.max(data['annual_income']) - np.min(data['annual_income']))
    target_prob += income_norm * 0.3
    
    # Credit score influence (handle NaN)
    credit_filled = pd.Series(data['credit_score']).fillna(50)
    credit_norm = (credit_filled - credit_filled.min()) / (credit_filled.max() - credit_filled.min())
    target_prob += credit_norm * 0.2
    
    # Membership level influence
    membership_map = {'Bronze': 0.05, 'Silver': 0.1, 'Gold': 0.2, 'Platinum': 0.4}
    membership_probs = [membership_map.get(m, 0) for m in data['membership_level']]
    target_prob += membership_probs
    
    # Normalize probabilities and add noise
    target_prob = target_prob / target_prob.max()
    target_prob += np.random.normal(0, 0.1, n_rows)
    target_prob = np.clip(target_prob, 0, 1)
    
    # Adjust to achieve desired target rate
    threshold = np.percentile(target_prob, 100 * (1 - target_rate))
    data['target'] = (target_prob > threshold).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some completely empty columns for testing
    df['empty_column'] = np.nan
    
    # Add a column with a single unique value
    df['constant_column'] = 'constant_value'
    
    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

# Generate the test data
test_df = generate_test_data(n_rows=500, target_rate=0.2)

# Save to CSV
test_df.to_csv('test_data.csv', index=False)

# Display basic info about the generated data
print("Test Data Generated Successfully!")
print(f"Shape: {test_df.shape}")
print(f"Columns: {list(test_df.columns)}")
print(f"Target rate: {test_df['target'].mean():.2%}")
print("\nColumn types:")
print(test_df.dtypes)
print("\nFirst few rows:")
test_df.head()
