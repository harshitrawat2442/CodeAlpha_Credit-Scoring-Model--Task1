# ============================================================
# PREPROCESSING MODULE
# Credit Scoring Model
# ============================================================

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(file_path):
    """
    Load the credit card dataset from an Excel file.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found:\n{file_path}"
        )

    try:
        df = pd.read_excel(
            file_path,
            header=1
        )

    except Exception as error:
        raise RuntimeError(
            f"Unable to read Excel dataset.\n"
            f"File: {file_path}\n"
            f"Error: {error}"
        )

    # Remove completely empty columns
    df = df.dropna(
        axis=1,
        how="all"
    )

    # Remove completely empty rows
    df = df.dropna(
        axis=0,
        how="all"
    )

    # Clean column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# 2. CLEAN DATA
# ============================================================

def clean_data(df):
    """
    Clean dataset and separate features and target.
    """

    if df is None or df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    df = df.copy()

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Target column
    # --------------------------------------------------------

    target_column = "default payment next month"

    # Handle possible capitalization/spacing differences
    matching_target = None

    for column in df.columns:

        normalized = (
            column
            .strip()
            .lower()
        )

        if normalized == target_column.lower():
            matching_target = column
            break

    if matching_target is None:

        raise KeyError(
            "Target column not found.\n\n"
            f"Expected:\n"
            f"'{target_column}'\n\n"
            "Available columns:\n"
            + "\n".join(
                str(column)
                for column in df.columns
            )
        )

    # --------------------------------------------------------
    # Remove ID
    # --------------------------------------------------------

    id_columns = []

    for column in df.columns:

        if str(column).strip().upper() == "ID":
            id_columns.append(column)

    if id_columns:
        df = df.drop(
            columns=id_columns
        )

    # --------------------------------------------------------
    # Separate X and y
    # --------------------------------------------------------

    X = df.drop(
        columns=[matching_target]
    )

    y = df[matching_target]

    # --------------------------------------------------------
    # Convert values to numeric
    # --------------------------------------------------------

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    y = pd.to_numeric(
        y,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Handle missing values
    # --------------------------------------------------------

    if X.isnull().any().any():

        X = X.fillna(
            X.median()
        )

    # Remove rows where target is missing
    valid_target = y.notna()

    X = X.loc[
        valid_target
    ].reset_index(drop=True)

    y = y.loc[
        valid_target
    ].reset_index(drop=True)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if X.empty:
        raise ValueError(
            "No valid feature data available after cleaning."
        )

    if y.empty:
        raise ValueError(
            "No valid target data available after cleaning."
        )

    return X, y


# ============================================================
# 3. SPLIT AND SCALE DATA
# ============================================================

def split_and_scale(
    X,
    y,
    test_size=0.20,
    random_state=42
):
    """
    Split the dataset into training/testing sets
    and scale numerical features.

    Returns:
        X_train
        X_test
        X_train_scaled
        X_test_scaled
        y_train
        y_test
        scaler
    """

    if X is None or y is None:
        raise ValueError(
            "X and y cannot be None."
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of rows."
        )

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # --------------------------------------------------------
    # Standard Scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )


# ============================================================
# 4. TEST MODULE
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CREDIT SCORING - PREPROCESSING MODULE")
    print("=" * 70)

    # Automatically locate project root
    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    DATA_PATH = os.path.join(
        BASE_DIR,
        "data",
        "default_of_credit_card_clients.xls"
    )

    try:

        print("\nLoading dataset...")

        df = load_data(
            DATA_PATH
        )

        print(
            f"✓ Dataset loaded successfully."
        )

        print(
            f"✓ Dataset shape: {df.shape}"
        )

        # ----------------------------------------------------
        # Clean
        # ----------------------------------------------------

        X, y = clean_data(
            df
        )

        print(
            f"\n✓ Data cleaned successfully."
        )

        print(
            f"✓ Features shape: {X.shape}"
        )

        print(
            f"✓ Target shape: {y.shape}"
        )

        # ----------------------------------------------------
        # Split and scale
        # ----------------------------------------------------

        (
            X_train,
            X_test,
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            scaler
        ) = split_and_scale(
            X,
            y
        )

        print(
            f"\n✓ Training samples : {len(X_train)}"
        )

        print(
            f"✓ Testing samples  : {len(X_test)}"
        )

        print(
            f"✓ Scaled training data shape : "
            f"{X_train_scaled.shape}"
        )

        print(
            f"✓ Scaled testing data shape  : "
            f"{X_test_scaled.shape}"
        )

        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as error:

        print("\n" + "=" * 70)
        print("PREPROCESSING ERROR")
        print("=" * 70)

        print(
            f"\n{error}"
        )

        print("\nPlease check:")
        print("1. Dataset exists inside data/")
        print("2. Filename is correct")
        print("3. Required Python packages are installed")
        print("4. Excel file can be read by pandas")

        print("=" * 70)