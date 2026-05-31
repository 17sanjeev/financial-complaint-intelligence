"""
Text preprocessing utilities for CFPB complaint narratives.
Author: Sanjeev Kumar | IIT Bombay
"""
import re
import pandas as pd
import numpy as np


# Product grouping map
PRODUCT_MAP = {
    'Credit reporting or other personal consumer reports': 'Credit Reporting',
    'Credit reporting, credit repair services, or other personal consumer reports': 'Credit Reporting',
    'Credit reporting': 'Credit Reporting',
    'Debt collection': 'Debt Collection',
    'Mortgage': 'Mortgage',
    'Checking or savings account': 'Bank Account',
    'Bank account or service': 'Bank Account',
    'Credit card': 'Credit Card',
    'Credit card or prepaid card': 'Credit Card',
    'Prepaid card': 'Credit Card',
    'Student loan': 'Loans',
    'Vehicle loan or lease': 'Loans',
    'Consumer Loan': 'Loans',
    'Payday loan, title loan, personal loan, or advance loan': 'Loans',
    'Payday loan, title loan, or personal loan': 'Loans',
    'Payday loan': 'Loans',
    'Money transfer, virtual currency, or money service': 'Money Transfer',
    'Money transfers': 'Money Transfer',
    'Debt or credit management': 'Debt Collection',
    'Other financial service': 'Other',
}

PRODUCT_CLASSES = [
    'Bank Account', 'Credit Card', 'Credit Reporting',
    'Debt Collection', 'Loans', 'Money Transfer', 'Mortgage'
]


def clean_text(text):
    """Clean complaint narrative text."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'x{2,}', 'XXXX', text)   # Redacted info
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_clean(url=None, filepath=None, nrows=200000):
    """
    Load CFPB data and apply all cleaning steps.
    Returns cleaned DataFrame with narrative text only.
    """
    if filepath:
        df = pd.read_csv(filepath, low_memory=False)
    else:
        if url is None:
            url = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
        df = pd.read_csv(url, compression='zip',
                         low_memory=False, nrows=nrows)

    # Keep only rows with narrative text
    df_text = df[df['Consumer complaint narrative'].notna()].copy()

    # Clean product categories
    df_text['product_clean'] = df_text['Product'].map(PRODUCT_MAP).fillna('Other')

    # Clean text
    df_text['narrative_clean'] = df_text['Consumer complaint narrative'].apply(clean_text)

    # Parse dates
    df_text['Date received'] = pd.to_datetime(df_text['Date received'])

    # Text length
    df_text['text_length'] = df_text['narrative_clean'].str.split().str.len()

    # Risk labels
    df_text['is_priority_review'] = (df_text['Timely response?'] == 'No').astype(int)
    df_text['is_untimely'] = (df_text['Timely response?'] == 'No').astype(int)
    df_text['is_monetary_relief'] = (
        df_text['Company response to consumer'].str.strip() == 'Closed with monetary relief'
    ).astype(int)

    return df_text


def time_based_split(df, train_pct=0.70, val_pct=0.15):
    """
    Chronological train/val/test split.
    NEVER use random split for time-series complaint data!
    """
    df_sorted = df.sort_values('Date received').reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_pct)
    val_end   = int(n * (train_pct + val_pct))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df   = df_sorted.iloc[train_end:val_end].copy()
    test_df  = df_sorted.iloc[val_end:].copy()

    print(f"Train: {len(train_df):,} | {train_df['Date received'].min().date()} - {train_df['Date received'].max().date()}")
    print(f"Val:   {len(val_df):,}   | {val_df['Date received'].min().date()} - {val_df['Date received'].max().date()}")
    print(f"Test:  {len(test_df):,}  | {test_df['Date received'].min().date()} - {test_df['Date received'].max().date()}")

    return train_df, val_df, test_df
