"""
data_processor.py
Will handle loading, validating, and cleaning transaction CSV data.
"""

import os
import pandas as pd


def load_sample_data(filepath="data/transactions.csv"):
    """
    Loads the sample transactions CSV using pandas.
    Returns a pandas DataFrame with columns: Date, Type, Category, Description, Amount.
    Converts the Date column to datetime.
    Converts Amount column to numeric, coercing errors to NaN, then drops rows where Amount is NaN.
    If the file does not exist, returns an empty DataFrame with the expected columns instead of raising an exception.
    """
    expected_columns = ["Date", "Type", "Category", "Description", "Amount"]

    # Step 1: Check if file exists; if not, return empty DataFrame with expected columns
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=expected_columns)

    try:
        # Step 2: Read CSV file using pandas
        df = pd.read_csv(filepath)

        # Ensure required columns exist in the DataFrame
        for col in expected_columns:
            if col not in df.columns:
                df[col] = None
        df = df[expected_columns]

        # Step 3: Convert Date column to datetime
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Step 4: Convert Amount column to numeric, coercing errors to NaN
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

        # Step 5: Drop rows where Amount or Date is NaN
        df = df.dropna(subset=["Amount", "Date"])

        return df

    except Exception:
        # Handle file read/parse errors gracefully by returning empty DataFrame
        return pd.DataFrame(columns=expected_columns)


def validate_columns(df):
    """
    Checks that the DataFrame contains the required columns:
    Date, Type, Category, Description, Amount (case-sensitive, exact match).
    Returns a tuple: (is_valid: bool, missing_columns: list of str).
    """
    required_columns = ["Date", "Type", "Category", "Description", "Amount"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    is_valid = len(missing_columns) == 0
    return (is_valid, missing_columns)


def process_uploaded_file(uploaded_file):
    """
    Takes a Streamlit UploadedFile object (from st.file_uploader).
    Reads it into a pandas DataFrame using pd.read_csv.
    Validates columns using validate_columns().
    If columns are missing, returns (None, error_message_string) where
    error_message_string clearly lists which columns are missing.
    If the file is empty (zero rows), returns (None, "The uploaded file is empty.").
    If valid, cleans the data the same way load_sample_data does:
      - Convert Date to datetime, coercing errors to NaT, drop rows where Date is NaT
      - Convert Amount to numeric, coercing errors to NaN, drop rows where Amount is NaN
      - Strip whitespace from Type and Category columns
    If any row-dropping happened due to invalid dates or amounts, include a warning
    message noting how many rows were dropped.
    Returns (cleaned_dataframe, message_string_or_None) on success, where message_string
    is None if no rows were dropped, or a warning string if some were dropped.
    If the file cannot be parsed as CSV at all, catch the exception and return
    (None, "Could not read this file as a CSV. Please check the format.").
    """
    # Step 1: Attempt to read uploaded file as CSV
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        return (None, "Could not read this file as a CSV. Please check the format.")

    # Step 2: Check if file contains zero rows
    if df.empty or len(df) == 0:
        return (None, "The uploaded file is empty.")

    # Step 3: Validate exact required columns
    is_valid, missing_cols = validate_columns(df)
    if not is_valid:
        missing_str = ", ".join(missing_cols)
        return (None, f"Uploaded CSV is missing required column(s): {missing_str}")

    initial_row_count = len(df)

    # Step 4: Convert Date to datetime, coercing invalid dates to NaT
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Step 5: Convert Amount to numeric, coercing non-numeric values to NaN
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    # Step 6: Drop rows where Date is NaT or Amount is NaN
    df = df.dropna(subset=["Date", "Amount"])

    # Step 7: Strip leading/trailing whitespace from string columns
    if "Type" in df.columns:
        df["Type"] = df["Type"].astype(str).str.strip()
    if "Category" in df.columns:
        df["Category"] = df["Category"].astype(str).str.strip()
    if "Description" in df.columns:
        df["Description"] = df["Description"].astype(str).str.strip()

    final_row_count = len(df)
    dropped_count = initial_row_count - final_row_count

    # Step 8: Build warning message if any rows were dropped
    warning_msg = None
    if dropped_count > 0:
        warning_msg = f"Warning: {dropped_count} row(s) were dropped due to invalid date or amount values."

    return (df, warning_msg)
