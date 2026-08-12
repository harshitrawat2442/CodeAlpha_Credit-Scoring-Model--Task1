import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression


# ============================================================
# CREDIT SCORING MODEL - PREDICTION SYSTEM
# CODEALPHA - TASK 1
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "credit_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)


# ============================================================
# 2. TERMINAL UI HELPERS
# ============================================================

def print_line(character="=", length=70):
    print(character * length)


def print_header(title, subtitle=None):

    print("\n")
    print_line("=")
    print(f"{title:^70}")

    if subtitle:
        print(f"{subtitle:^70}")

    print_line("=")


# ============================================================
# 3. MODEL LOADING
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"\nModel file not found:\n{MODEL_PATH}\n\n"
            "Please run train.py first."
        )

    if not os.path.exists(SCALER_PATH):

        raise FileNotFoundError(
            f"\nScaler file not found:\n{SCALER_PATH}\n\n"
            "Please run train.py first."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


# ============================================================
# 4. INPUT VALIDATION
# ============================================================

def get_positive_float(message):

    while True:

        try:

            value = float(input(message))

            if value < 0:

                print(
                    "  ✗ Value cannot be negative."
                )

                continue

            return value

        except ValueError:

            print(
                "  ✗ Invalid input. Please enter a number."
            )


def get_integer(message):

    while True:

        try:

            return int(input(message))

        except ValueError:

            print(
                "  ✗ Invalid input. Please enter a whole number."
            )


def get_choice(message, valid_choices):

    while True:

        try:

            value = int(input(message))

            if value in valid_choices:

                return value

            print(
                f"  ✗ Please choose from: {valid_choices}"
            )

        except ValueError:

            print(
                "  ✗ Invalid input. Please enter a number."
            )


# ============================================================
# 5. BASIC CUSTOMER INFORMATION
# ============================================================

def get_basic_customer_information():

    print_header(
        "CUSTOMER PROFILE",
        "Enter basic customer information"
    )

    limit_bal = get_positive_float(
        "\n  Credit Limit (LIMIT_BAL) : "
    )

    sex = get_choice(
        "  Sex (1=Male, 2=Female)   : ",
        [1, 2]
    )

    education = get_choice(
        "\n  Education:\n"
        "    1 = Graduate School\n"
        "    2 = University\n"
        "    3 = High School\n"
        "    4 = Others\n"
        "  Select Education         : ",
        [1, 2, 3, 4]
    )

    marriage = get_choice(
        "\n  Marriage:\n"
        "    1 = Married\n"
        "    2 = Single\n"
        "    3 = Others\n"
        "  Select Marriage          : ",
        [1, 2, 3]
    )

    while True:

        age = get_integer(
            "\n  Age                      : "
        )

        if 18 <= age <= 100:

            break

        print(
            "  ✗ Age must be between 18 and 100."
        )

    return {
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age
    }


# ============================================================
# 6. REPAYMENT HISTORY
# ============================================================

def get_repayment_history():

    print_header(
        "REPAYMENT HISTORY",
        "Payment delay status for the previous months"
    )

    print(
        "\n  Meaning:"
    )

    print(
        "    0       = Payment made on time"
    )

    print(
        "    Positive = Payment delay"
    )

    print(
        "    Negative = Payment made earlier than expected"
    )

    print("\n")

    repayment = {}

    repayment_features = [
        ("PAY_0", "Most Recent Month"),
        ("PAY_2", "Month - 2"),
        ("PAY_3", "Month - 3"),
        ("PAY_4", "Month - 4"),
        ("PAY_5", "Month - 5"),
        ("PAY_6", "Month - 6")
    ]

    for feature, description in repayment_features:

        while True:

            value = get_integer(
                f"  {feature:<7} ({description:<20}) : "
            )

            # UCI dataset normally uses values
            # from -2 to 8
            if -2 <= value <= 8:

                repayment[feature] = value

                break

            print(
                "  ✗ Please enter a value between -2 and 8."
            )

    return repayment


# ============================================================
# 7. BILL AMOUNT INTERFACE
# ============================================================

def get_bill_amounts():

    print_header(
        "MONTHLY BILL AMOUNTS",
        "Credit card statement amount for each month"
    )

    print(
        "\n  Enter the amount shown on the customer's"
    )

    print(
        "  credit card statement for each month."
    )

    print("\n")

    bill_amounts = {}

    months = [
        ("BILL_AMT1", "Month 1"),
        ("BILL_AMT2", "Month 2"),
        ("BILL_AMT3", "Month 3"),
        ("BILL_AMT4", "Month 4"),
        ("BILL_AMT5", "Month 5"),
        ("BILL_AMT6", "Month 6")
    ]

    print(
        "  ┌───────────────┬────────────────────────┐"
    )

    print(
        "  │    MONTH      │     BILL AMOUNT        │"
    )

    print(
        "  ├───────────────┼────────────────────────┤"
    )

    for feature, month in months:

        value = get_positive_float(
            f"  │ {month:<13} │ ₹ "
        )

        print(
            "  │               │                        │"
        )

        bill_amounts[feature] = value

    print(
        "  └───────────────┴────────────────────────┘"
    )

    return bill_amounts


# ============================================================
# 8. PAYMENT AMOUNT INTERFACE
# ============================================================

def get_payment_amounts():

    print_header(
        "MONTHLY PAYMENT HISTORY",
        "Actual amount paid by the customer"
    )

    print(
        "\n  Enter the actual amount paid by the customer"
    )

    print(
        "  for each corresponding month."
    )

    print("\n")

    payment_amounts = {}

    months = [
        ("PAY_AMT1", "Month 1"),
        ("PAY_AMT2", "Month 2"),
        ("PAY_AMT3", "Month 3"),
        ("PAY_AMT4", "Month 4"),
        ("PAY_AMT5", "Month 5"),
        ("PAY_AMT6", "Month 6")
    ]

    print(
        "  ┌───────────────┬────────────────────────┐"
    )

    print(
        "  │    MONTH      │    PAYMENT MADE        │"
    )

    print(
        "  ├───────────────┼────────────────────────┤"
    )

    for feature, month in months:

        value = get_positive_float(
            f"  │ {month:<13} │ ₹ "
        )

        print(
            "  │               │                        │"
        )

        payment_amounts[feature] = value

    print(
        "  └───────────────┴────────────────────────┘"
    )

    return payment_amounts


# ============================================================
# 9. CREATE CUSTOMER DATA
# ============================================================

def collect_customer_data():

    basic_info = get_basic_customer_information()

    repayment = get_repayment_history()

    bill_amounts = get_bill_amounts()

    payment_amounts = get_payment_amounts()

    customer_data = {}

    customer_data.update(
        basic_info
    )

    customer_data.update(
        repayment
    )

    customer_data.update(
        bill_amounts
    )

    customer_data.update(
        payment_amounts
    )

    return customer_data


# ============================================================
# 10. CUSTOMER DATA SUMMARY
# ============================================================

def display_customer_summary(customer_data):

    print_header(
        "CUSTOMER DATA SUMMARY",
        "Information submitted for prediction"
    )

    print("\n  BASIC INFORMATION")
    print("  " + "-" * 60)

    print(
        f"  Credit Limit : ₹{customer_data['LIMIT_BAL']:,.2f}"
    )

    print(
        f"  Age          : {customer_data['AGE']}"
    )

    print(
        f"  Sex          : {customer_data['SEX']}"
    )

    print(
        f"  Education    : {customer_data['EDUCATION']}"
    )

    print(
        f"  Marriage     : {customer_data['MARRIAGE']}"
    )

    print("\n  REPAYMENT HISTORY")
    print("  " + "-" * 60)

    print(
        f"  PAY_0 : {customer_data['PAY_0']}"
    )

    print(
        f"  PAY_2 : {customer_data['PAY_2']}"
    )

    print(
        f"  PAY_3 : {customer_data['PAY_3']}"
    )

    print(
        f"  PAY_4 : {customer_data['PAY_4']}"
    )

    print(
        f"  PAY_5 : {customer_data['PAY_5']}"
    )

    print(
        f"  PAY_6 : {customer_data['PAY_6']}"
    )

    print("\n  BILL AMOUNTS")
    print("  " + "-" * 60)

    for i in range(1, 7):

        value = customer_data[
            f"BILL_AMT{i}"
        ]

        print(
            f"  Month {i} : ₹{value:,.2f}"
        )

    print("\n  PAYMENT AMOUNTS")
    print("  " + "-" * 60)

    for i in range(1, 7):

        value = customer_data[
            f"PAY_AMT{i}"
        ]

        print(
            f"  Month {i} : ₹{value:,.2f}"
        )


# ============================================================
# 11. PREDICTION FUNCTION
# ============================================================

def predict_credit_risk(
    customer_data,
    model,
    scaler
):

    customer_df = pd.DataFrame(
        [customer_data]
    )

    # --------------------------------------------------------
    # Feature validation
    # --------------------------------------------------------

    if hasattr(model, "feature_names_in_"):

        expected_features = list(
            model.feature_names_in_
        )

        missing_features = [
            feature
            for feature in expected_features
            if feature not in customer_df.columns
        ]

        if missing_features:

            raise ValueError(
                "Missing features: "
                + str(missing_features)
            )

        customer_df = customer_df[
            expected_features
        ]

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    if isinstance(
        model,
        LogisticRegression
    ):

        customer_input = scaler.transform(
            customer_df
        )

    else:

        customer_input = customer_df

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        customer_input
    )[0]

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        customer_input
    )[0]

    default_probability = probabilities[1]

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    if prediction == 1:

        risk = "HIGH RISK"

    else:

        risk = "LOW RISK"

    return {
        "prediction": int(prediction),
        "risk": risk,
        "probability": float(
            default_probability
        )
    }


# ============================================================
# 12. DISPLAY PREDICTION RESULT
# ============================================================

def display_result(result):

    probability = (
        result["probability"] * 100
    )

    print_header(
        "CREDIT RISK PREDICTION",
        "Machine Learning Assessment"
    )

    print("\n")

    print(
        f"  Credit Risk         : {result['risk']}"
    )

    print(
        f"  Default Probability : {probability:.2f}%"
    )

    print(
        f"  Prediction Class    : {result['prediction']}"
    )

    print("\n")

    print_line("-", 70)

    if result["prediction"] == 0:

        print(
            "\n  ✓ ASSESSMENT"
        )

        print(
            "  The model predicts a lower probability"
        )

        print(
            "  of credit default for this customer."
        )

    else:

        print(
            "\n  ! ASSESSMENT"
        )

        print(
            "  The model predicts a higher probability"
        )

        print(
            "  of credit default for this customer."
        )

    print("\n")

    if probability < 20:

        probability_level = "Very Low"

    elif probability < 40:

        probability_level = "Low"

    elif probability < 60:

        probability_level = "Moderate"

    elif probability < 80:

        probability_level = "High"

    else:

        probability_level = "Very High"

    print(
        f"  Probability Level   : {probability_level}"
    )

    print("\n")

    print_line("-", 70)

    print(
        "\n  NOTE:"
    )

    print(
        "  This prediction is generated by the trained"
    )

    print(
        "  machine learning model for project/demo purposes."
    )

    print(
        "  It should not be used as the sole basis for"
    )

    print(
        "  real-world financial or lending decisions."
    )

    print_line("=")


# ============================================================
# 13. MAIN PROGRAM
# ============================================================

def main():

    print("\n")

    print_line("=")

    print(
        "CREDIT SCORING MODEL".center(70)
    )

    print(
        "CODEALPHA - TASK 1".center(70)
    )

    print(
        "Credit Default Risk Prediction System".center(70)
    )

    print_line("=")

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    try:

        model, scaler = load_model()

    except Exception as error:

        print("\nMODEL LOADING ERROR")
        print("-" * 70)

        print(error)

        return

    print(
        "\n✓ Trained model loaded successfully."
    )

    print(
        f"✓ Model Type: {type(model).__name__}"
    )

    # --------------------------------------------------------
    # Prediction loop
    # --------------------------------------------------------

    while True:

        try:

            customer_data = collect_customer_data()

            # ------------------------------------------------
            # Summary
            # ------------------------------------------------

            display_customer_summary(
                customer_data
            )

            # ------------------------------------------------
            # Confirmation
            # ------------------------------------------------

            print("\n")

            confirmation = input(
                "Proceed with prediction? (y/n): "
            ).strip().lower()

            if confirmation != "y":

                print(
                    "\nPrediction cancelled."
                )

                continue

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            result = predict_credit_risk(
                customer_data,
                model,
                scaler
            )

            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            display_result(
                result
            )

        except ValueError as error:

            print("\n")
            print_line("=")

            print(
                "INPUT ERROR".center(70)
            )

            print_line("=")

            print(
                f"\n{error}"
            )

            print(
                "\nPlease enter valid values and try again."
            )

        except Exception as error:

            print("\n")
            print_line("=")

            print(
                "PREDICTION ERROR".center(70)
            )

            print_line("=")

            print(
                f"\n{error}"
            )

        # ----------------------------------------------------
        # Another prediction
        # ----------------------------------------------------

        print("\n")

        another = input(
            "Do you want to predict another customer? (y/n): "
        ).strip().lower()

        if another != "y":

            print("\n")

            print_line("=")

            print(
                "Thank you for using the Credit Scoring Model."
                .center(70)
            )

            print_line("=")

            break


# ============================================================
# 14. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()