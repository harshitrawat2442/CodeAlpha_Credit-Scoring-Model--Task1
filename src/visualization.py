import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from preprocessing import (
    load_data,
    clean_data,
    split_and_scale
)


# ============================================================
# PROJECT PATHS
# ============================================================

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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

MODEL_RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "model_results.csv"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def print_line(character="=", length=70):

    print(
        character * length
    )


def print_header(title):

    print("\n")

    print_line("=")

    print(
        f"{title:^70}"
    )

    print_line("=")


# ============================================================
# LOAD SAVED MODEL
# ============================================================

def load_saved_model():

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}\n\n"
            "Please run train.py first."
        )

    if not os.path.exists(
        SCALER_PATH
    ):

        raise FileNotFoundError(
            f"Scaler not found:\n{SCALER_PATH}\n\n"
            "Please run train.py first."
        )

    model = joblib.load(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    return (
        model,
        scaler
    )


# ============================================================
# PREPARE TEST DATA
# ============================================================

def prepare_test_data():

    print(
        "\nLoading dataset..."
    )

    df = load_data(
        DATA_PATH
    )

    print(
        "✓ Dataset loaded."
    )

    X, y = clean_data(
        df
    )

    print(
        "✓ Data cleaned."
    )

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
        f"✓ Test samples: {len(X_test)}"
    )

    return (
        X_test,
        y_test
    )


# ============================================================
# 1. CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(
    y_test,
    predictions
):

    print_header(
        "GENERATING CONFUSION MATRIX"
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No Default",
            "Default"
        ]
    )

    display.plot()

    plt.title(
        "Credit Default Prediction - Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        f"✓ Saved: {output_path}"
    )


# ============================================================
# 2. ROC CURVE
# ============================================================

def create_roc_curve(
    y_test,
    probabilities
):

    print_header(
        "GENERATING ROC CURVE"
    )

    (
        false_positive_rate,
        true_positive_rate,
        _
    ) = roc_curve(
        y_test,
        probabilities
    )

    roc_auc = auc(
        false_positive_rate,
        true_positive_rate
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"ROC Curve (AUC = {roc_auc:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Credit Default Prediction - ROC Curve"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "roc_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        f"✓ ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"✓ Saved: {output_path}"
    )


# ============================================================
# 3. FEATURE IMPORTANCE
# ============================================================

def create_feature_importance(
    model,
    X_test
):

    print_header(
        "GENERATING FEATURE IMPORTANCE"
    )

    if not hasattr(
        model,
        "feature_importances_"
    ):

        print(
            "⚠ This model does not provide feature importance."
        )

        return

    importances = model.feature_importances_

    feature_names = X_test.columns

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances
        }
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=True
    )

    # --------------------------------------------------------
    # Top 15 features
    # --------------------------------------------------------

    top_features = importance_df.tail(
        15
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 15 Features Influencing Credit Default Prediction"
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "feature_importance.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        f"✓ Saved: {output_path}"
    )

    # --------------------------------------------------------
    # Save feature importance data
    # --------------------------------------------------------

    csv_path = os.path.join(
        OUTPUT_DIR,
        "feature_importance.csv"
    )

    importance_df.sort_values(
        by="Importance",
        ascending=False
    ).to_csv(
        csv_path,
        index=False
    )

    print(
        f"✓ Saved: {csv_path}"
    )


# ============================================================
# 4. MODEL PERFORMANCE GRAPH
# ============================================================

def create_performance_chart(
    y_test,
    predictions,
    probabilities
):

    print_header(
        "GENERATING PERFORMANCE CHART"
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ]

    metric_values = [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]

    plt.figure(
        figsize=(10, 6)
    )

    bars = plt.bar(
        metric_names,
        metric_values
    )

    plt.ylim(
        0,
        1
    )

    plt.ylabel(
        "Score"
    )

    plt.xlabel(
        "Evaluation Metric"
    )

    plt.title(
        "Credit Scoring Model Performance"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    # --------------------------------------------------------
    # Display values above bars
    # --------------------------------------------------------

    for bar, value in zip(
        bars,
        metric_values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center"
        )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "model_performance.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        f"✓ Saved: {output_path}"
    )


# ============================================================
# 5. MODEL COMPARISON GRAPH
# ============================================================

def create_model_comparison():

    print_header(
        "GENERATING MODEL COMPARISON GRAPH"
    )

    # --------------------------------------------------------
    # Check model_results.csv
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_RESULTS_PATH
    ):

        print(
            "⚠ model_results.csv not found."
        )

        print(
            f"Expected file:\n{MODEL_RESULTS_PATH}"
        )

        print(
            "\nPlease run train.py first."
        )

        return False

    # --------------------------------------------------------
    # Load model results
    # --------------------------------------------------------

    results_df = pd.read_csv(
        MODEL_RESULTS_PATH
    )

    print(
        "✓ Model results loaded successfully."
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in results_df.columns
    ]

    if missing_columns:

        print(
            "\n⚠ Required columns missing:"
        )

        for column in missing_columns:

            print(
                f"   - {column}"
            )

        print(
            "\nPlease regenerate model_results.csv "
            "by running train.py."
        )

        return False

    # --------------------------------------------------------
    # Display model comparison
    # --------------------------------------------------------

    print(
        "\nMODEL PERFORMANCE COMPARISON"
    )

    print_line(
        "-",
        70
    )

    print(
        results_df[
            required_columns
        ].to_string(
            index=False
        )
    )

    print_line(
        "-",
        70
    )

    # --------------------------------------------------------
    # Convert scores to percentage
    # --------------------------------------------------------

    models = results_df[
        "Model"
    ].astype(str)

    accuracy = (
        results_df["Accuracy"] * 100
    )

    precision = (
        results_df["Precision"] * 100
    )

    recall = (
        results_df["Recall"] * 100
    )

    f1 = (
        results_df["F1-Score"] * 100
    )

    roc_auc = (
        results_df["ROC-AUC"] * 100
    )

    # --------------------------------------------------------
    # X positions
    # --------------------------------------------------------

    x = np.arange(
        len(models)
    )

    width = 0.15

    # --------------------------------------------------------
    # Create graph
    # --------------------------------------------------------

    plt.figure(
        figsize=(14, 8)
    )

    bars_accuracy = plt.bar(
        x - 2 * width,
        accuracy,
        width,
        label="Accuracy"
    )

    bars_precision = plt.bar(
        x - width,
        precision,
        width,
        label="Precision"
    )

    bars_recall = plt.bar(
        x,
        recall,
        width,
        label="Recall"
    )

    bars_f1 = plt.bar(
        x + width,
        f1,
        width,
        label="F1-Score"
    )

    bars_roc = plt.bar(
        x + 2 * width,
        roc_auc,
        width,
        label="ROC-AUC"
    )

    # --------------------------------------------------------
    # Graph formatting
    # --------------------------------------------------------

    plt.xlabel(
        "Machine Learning Model"
    )

    plt.ylabel(
        "Performance (%)"
    )

    plt.title(
        "Credit Scoring Model Comparison"
    )

    plt.xticks(
        x,
        models,
        rotation=15
    )

    plt.ylim(
        0,
        100
    )

    plt.legend(
        loc="upper right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    # --------------------------------------------------------
    # Add values above bars
    # --------------------------------------------------------

    bar_groups = [
        bars_accuracy,
        bars_precision,
        bars_recall,
        bars_f1,
        bars_roc
    ]

    for bars in bar_groups:

        for bar in bars:

            height = bar.get_height()

            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                height + 1,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save model comparison graph
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        "model_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        "\n✓ Model comparison graph generated successfully."
    )

    print(
        f"✓ Saved: {output_path}"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "CREDIT SCORING MODEL VISUALIZATION"
    )

    try:

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model, scaler = load_saved_model()

        print(
            f"\n✓ Model loaded: {type(model).__name__}"
        )

        # ----------------------------------------------------
        # Prepare test data
        # ----------------------------------------------------

        X_test, y_test = prepare_test_data()

        # ----------------------------------------------------
        # Prepare model input
        # ----------------------------------------------------

        if type(model).__name__ == "LogisticRegression":

            X_test_input = scaler.transform(
                X_test
            )

        else:

            X_test_input = X_test

        # ----------------------------------------------------
        # Generate predictions
        # ----------------------------------------------------

        predictions = model.predict(
            X_test_input
        )

        probabilities = model.predict_proba(
            X_test_input
        )[:, 1]

        print(
            "\n✓ Predictions generated successfully."
        )

        # ----------------------------------------------------
        # Generate confusion matrix
        # ----------------------------------------------------

        create_confusion_matrix(
            y_test,
            predictions
        )

        # ----------------------------------------------------
        # Generate ROC curve
        # ----------------------------------------------------

        create_roc_curve(
            y_test,
            probabilities
        )

        # ----------------------------------------------------
        # Generate feature importance
        # ----------------------------------------------------

        create_feature_importance(
            model,
            X_test
        )

        # ----------------------------------------------------
        # Generate performance chart
        # ----------------------------------------------------

        create_performance_chart(
            y_test,
            predictions,
            probabilities
        )

        # ----------------------------------------------------
        # Generate model comparison
        # ----------------------------------------------------

        create_model_comparison()

        # ----------------------------------------------------
        # Final message
        # ----------------------------------------------------

        print_header(
            "VISUALIZATION COMPLETE"
        )

        print(
            "\nAll graphs have been generated successfully."
        )

        print(
            "\nOutput folder:"
        )

        print(
            OUTPUT_DIR
        )

        print(
            "\nGenerated files:"
        )

        print(
            "  ✓ confusion_matrix.png"
        )

        print(
            "  ✓ roc_curve.png"
        )

        print(
            "  ✓ feature_importance.png"
        )

        print(
            "  ✓ feature_importance.csv"
        )

        print(
            "  ✓ model_performance.png"
        )

        print(
            "  ✓ model_comparison.png"
        )

        print_line("=")

    except Exception as error:

        print_header(
            "VISUALIZATION ERROR"
        )

        print(
            f"\n{error}"
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()