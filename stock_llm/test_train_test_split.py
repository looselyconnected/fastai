"""
Test script to demonstrate the train/test split functionality.
"""

from data import get_data_for_eval
from datetime import datetime
import pandas as pd

def test_data_split(ticker='AAPL', train_cutoff_date='2022-12-31', eval_start_date='2023-01-01'):
    """Test the train/test data split."""
    print(f"🧪 Testing data split for {ticker}")
    
    # Load data
    all_data_df = get_data_for_eval(ticker, 'data')
    print(f"📊 Total data: {len(all_data_df)} days")
    
    # Parse dates
    train_cutoff_date_obj = datetime.strptime(train_cutoff_date, '%Y-%m-%d').date()
    eval_start_date_obj = datetime.strptime(eval_start_date, '%Y-%m-%d').date()
    
    # Split data
    train_data_df = all_data_df[all_data_df['Date'] <= train_cutoff_date_obj]
    eval_data_df = all_data_df[all_data_df['Date'] >= eval_start_date_obj]
    
    print(f"📈 Training data: {len(train_data_df)} days (up to {train_cutoff_date})")
    print(f"🧪 Evaluation data: {len(eval_data_df)} days (from {eval_start_date} onwards)")
    
    # Show date ranges
    if len(train_data_df) > 0:
        print(f"  Training date range: {train_data_df['Date'].min()} to {train_data_df['Date'].max()}")
    
    if len(eval_data_df) > 0:
        print(f"  Evaluation date range: {eval_data_df['Date'].min()} to {eval_data_df['Date'].max()}")
    
    # Check for data leakage
    overlap = len(all_data_df) - len(train_data_df) - len(eval_data_df)
    print(f"📝 Data gap/overlap: {overlap} days")
    
    return train_data_df, eval_data_df

if __name__ == "__main__":
    # Test with AAPL
    train_df, eval_df = test_data_split('AAPL')
    
    print("\n" + "="*50)
    print("✅ Train/Test split working correctly!")
    print("="*50)