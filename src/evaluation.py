import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from preprocessing import (
    load_data,
    clean_data,
    split_and_scale
)


# ============================================================
# CREDIT SCORING MODEL - EVALUATION
# ============================================================
# This script:
#
# 1. Loads the trained model
# 2. Loads the scaler
# 3. Loads and cleans the dataset
# 4. Recreates the test split
# 5. Generates predictions
# 6. Calculates evaluation metrics
# 7. Generates classification report
# 8. Generates confusion matrix
# 9. Generates ROC curve
# 10. Generates metrics chart
# 11. Saves evaluation results
#
# Project:
# CODEALPHA-SCORING_MODEL
# ============================================================


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
    print(character * length)


def print_header(title):

    print("\n")
    print_line("=")
    print(f"{title:^70}")
    print_line("=")


def print_section(title):

    print("\n")
    print_line("-")
    print(f"{title:^70}")
    print_line("-")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_saved_model():

    print_section(
        "LOADING TRAINED MODEL"
    )

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "\nTrained model not found!\n\n"
            f"Expected location:\n{MODEL_PATH}\n\n"
            "Please run train.py first."
        )

    # --------------------------------------------------------
    # Check scaler
    # --------------------------------------------------------

    if not os.path.exists(SCALER_PATH):

        raise FileNotFoundError(
            "\nScaler not found!\n\n"
            f"Expected location:\n{SCALER_PATH}\n\n"
            "Please run train.py first."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = joblib.load(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Load scaler
    # --------------------------------------------------------

    scaler = joblib.load(
        SCALER_PATH
    )

    print(
        "\n✓ Trained model loaded successfully."
    )

    print(
        f"✓ Model type: {type(model).__name__}"
    )

    print(
        "✓ Scaler loaded successfully."
    )

    return model, scaler


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

def prepare_test_data():

    print_section(
        "PREPARING TEST DATA"
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        "\nLoading dataset..."
    )

    df = load_data(
        DATA_PATH
    )

    print(
        "✓ Dataset loaded successfully."
    )

    print(
        f"✓ Dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Clean dataset
    # --------------------------------------------------------

    X, y = clean_data(
        df
    )

    print(
        f"✓ Features shape: {X.shape}"
    )

    print(
        f"✓ Target shape: {y.shape}"
    )

    # --------------------------------------------------------
    # Split and scale
    # --------------------------------------------------------

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
        f"✓ Training samples: {len(X_train)}"
    )

    print(
        f"✓ Testing samples: {len(X_test)}"
    )

    print(
        f"✓ Number of features: {X_test.shape[1]}"
    )

    return (
        X_test,
        y_test
    )


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    scaler,
    X_test
):

    print_section(
        "GENERATING PREDICTIONS"
    )

    model_name = type(model).__name__

    # --------------------------------------------------------
    # Logistic Regression requires scaled data
    # --------------------------------------------------------

    if model_name == "LogisticRegression":

        print(
            "\n✓ Applying saved scaler..."
        )

        X_test_input = scaler.transform(
            X_test
        )

    # --------------------------------------------------------
    # Tree-based models use original features
    # --------------------------------------------------------

    else:

        print(
            "\n✓ Using original feature values..."
        )

        X_test_input = X_test

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test_input
    )

    # --------------------------------------------------------
    # Prediction probabilities
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            X_test_input
        )[:, 1]

    else:

        probabilities = None

    print(
        "✓ Predictions generated successfully."
    )

    if probabilities is not None:

        print(
            "✓ Prediction probabilities generated successfully."
        )

    return (
        predictions,
        probabilities
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_test,
    predictions,
    probabilities
):

    print_section(
        "CALCULATING EVALUATION METRICS"
    )

    # --------------------------------------------------------
    # Basic metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    if probabilities is not None:

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

    else:

        roc_auc = float("nan")

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    # Make sure matrix is 2 x 2
    if cm.shape == (2, 2):

        tn, fp, fn, tp = cm.ravel()

    else:

        tn = 0
        fp = 0
        fn = 0
        tp = 0

    # --------------------------------------------------------
    # Specificity
    # --------------------------------------------------------

    if (tn + fp) > 0:

        specificity = tn / (tn + fp)

    else:

        specificity = 0.0

    # --------------------------------------------------------
    # Store metrics
    # --------------------------------------------------------

    metrics = {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1-Score": f1,

        "ROC-AUC": roc_auc,

        "Specificity": specificity,

        "True Negative": tn,

        "False Positive": fp,

        "False Negative": fn,

        "True Positive": tp
    }

    # --------------------------------------------------------
    # Display metrics
    # --------------------------------------------------------

    print(
        f"\nAccuracy     : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision    : {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall       : {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1-Score     : {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"ROC-AUC      : {roc_auc:.4f}"
    )

    print(
        f"Specificity  : {specificity:.4f} "
        f"({specificity * 100:.2f}%)"
    )

    print(
        "\nConfusion Matrix Values:"
    )

    print(
        f"True Negative : {tn}"
    )

    print(
        f"False Positive: {fp}"
    )

    print(
        f"False Negative: {fn}"
    )

    print(
        f"True Positive : {tp}"
    )

    return metrics


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def create_classification_report(
    y_test,
    predictions
):

    print_section(
        "CLASSIFICATION REPORT"
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "No Default",
            "Default"
        ],
        zero_division=0
    )

    print("\n")
    print(report)

    # --------------------------------------------------------
    # Save text report
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "CREDIT SCORING MODEL - CLASSIFICATION REPORT\n"
        )

        file.write(
            "=" * 60
        )

        file.write(
            "\n\n"
        )

        file.write(
            report
        )

    print(
        f"✓ Classification report saved:"
    )

    print(
        f"  {output_path}"
    )

    return report


# ============================================================
# CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(
    y_test,
    predictions
):

    print_section(
        "GENERATING EVALUATION CONFUSION MATRIX"
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        "Credit Default Prediction - Evaluation Confusion Matrix"
    )

    plt.colorbar()

    class_names = [
        "No Default",
        "Default"
    ]

    tick_marks = [
        0,
        1
    ]

    plt.xticks(
        tick_marks,
        class_names
    )

    plt.yticks(
        tick_marks,
        class_names
    )

    # --------------------------------------------------------
    # Display values
    # --------------------------------------------------------

    threshold = cm.max() / 2.0

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                color="white"
                if cm[i, j] > threshold
                else "black",
                fontsize=14
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
        "evaluation_confusion_matrix.png"
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
# ROC CURVE
# ============================================================

def create_roc_curve(
    y_test,
    probabilities
):

    print_section(
        "GENERATING EVALUATION ROC CURVE"
    )

    if probabilities is None:

        print(
            "⚠ Probability predictions are unavailable."
        )

        return

    # --------------------------------------------------------
    # ROC values
    # --------------------------------------------------------

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            y_test,
            probabilities
        )
    )

    roc_auc = auc(
        false_positive_rate,
        true_positive_rate
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

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
        linewidth=1,
        label="Random Classifier"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Credit Default Prediction - Evaluation ROC Curve"
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
        "evaluation_roc_curve.png"
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
# METRICS CHART
# ============================================================

def create_metrics_chart(
    metrics
):

    print_section(
        "GENERATING EVALUATION METRICS CHART"
    )

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "Specificity"
    ]

    metric_values = [
        metrics["Accuracy"],
        metrics["Precision"],
        metrics["Recall"],
        metrics["F1-Score"],
        metrics["ROC-AUC"],
        metrics["Specificity"]
    ]

    # --------------------------------------------------------
    # Create chart
    # --------------------------------------------------------

    plt.figure(
        figsize=(11, 6)
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
        "Credit Scoring Model - Evaluation Metrics"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    # --------------------------------------------------------
    # Display values
    # --------------------------------------------------------

    for bar, value in zip(
        bars,
        metric_values
    ):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.02, 0.98),
            f"{value:.3f}",
            ha="center",
            fontsize=10
        )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "evaluation_metrics.png"
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
# SAVE METRICS CSV
# ============================================================

def save_metrics_csv(
    model,
    metrics
):

    print_section(
        "SAVING EVALUATION RESULTS"
    )

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    results = {

        "Model": [
            type(model).__name__
        ],

        "Accuracy": [
            metrics["Accuracy"]
        ],

        "Precision": [
            metrics["Precision"]
        ],

        "Recall": [
            metrics["Recall"]
        ],

        "F1-Score": [
            metrics["F1-Score"]
        ],

        "ROC-AUC": [
            metrics["ROC-AUC"]
        ],

        "Specificity": [
            metrics["Specificity"]
        ],

        "True Negative": [
            metrics["True Negative"]
        ],

        "False Positive": [
            metrics["False Positive"]
        ],

        "False Negative": [
            metrics["False Negative"]
        ],

        "True Positive": [
            metrics["True Positive"]
        ]
    }

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        "evaluation_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"✓ Evaluation results saved:"
    )

    print(
        f"  {output_path}"
    )

    return results_df


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

def save_evaluation_summary(
    model,
    metrics
):

    output_path = os.path.join(
        OUTPUT_DIR,
        "evaluation_summary.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "CREDIT SCORING MODEL - EVALUATION SUMMARY\n"
        )

        file.write(
            "=" * 60
        )

        file.write(
            "\n\n"
        )

        file.write(
            f"Model        : {type(model).__name__}\n"
        )

        file.write(
            f"Accuracy     : {metrics['Accuracy']:.4f}\n"
        )

        file.write(
            f"Precision    : {metrics['Precision']:.4f}\n"
        )

        file.write(
            f"Recall       : {metrics['Recall']:.4f}\n"
        )

        file.write(
            f"F1-Score     : {metrics['F1-Score']:.4f}\n"
        )

        file.write(
            f"ROC-AUC      : {metrics['ROC-AUC']:.4f}\n"
        )

        file.write(
            f"Specificity  : {metrics['Specificity']:.4f}\n"
        )

        file.write(
            "\n"
        )

        file.write(
            "CONFUSION MATRIX VALUES\n"
        )

        file.write(
            "-" * 30
        )

        file.write(
            "\n"
        )

        file.write(
            f"True Negative : {metrics['True Negative']}\n"
        )

        file.write(
            f"False Positive: {metrics['False Positive']}\n"
        )

        file.write(
            f"False Negative: {metrics['False Negative']}\n"
        )

        file.write(
            f"True Positive : {metrics['True Positive']}\n"
        )

    print(
        f"✓ Evaluation summary saved:"
    )

    print(
        f"  {output_path}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def display_final_summary(
    model,
    metrics
):

    print_header(
        "FINAL EVALUATION SUMMARY"
    )

    print(
        f"\nModel       : {type(model).__name__}"
    )

    print(
        f"Accuracy    : {metrics['Accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision   : {metrics['Precision'] * 100:.2f}%"
    )

    print(
        f"Recall      : {metrics['Recall'] * 100:.2f}%"
    )

    print(
        f"F1-Score    : {metrics['F1-Score'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC     : {metrics['ROC-AUC']:.4f}"
    )

    print(
        f"Specificity : {metrics['Specificity'] * 100:.2f}%"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        f"  True Negative : {metrics['True Negative']}"
    )

    print(
        f"  False Positive: {metrics['False Positive']}"
    )

    print(
        f"  False Negative: {metrics['False Negative']}"
    )

    print(
        f"  True Positive : {metrics['True Positive']}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "CREDIT SCORING MODEL EVALUATION"
    )

    print(
        "\nCODEALPHA - TASK 1"
    )

    print(
        "Credit Default Risk Prediction System"
    )

    try:

        # ----------------------------------------------------
        # 1. Load trained model
        # ----------------------------------------------------

        model, scaler = load_saved_model()

        # ----------------------------------------------------
        # 2. Prepare test data
        # ----------------------------------------------------

        (
            X_test,
            y_test
        ) = prepare_test_data()

        # ----------------------------------------------------
        # 3. Generate predictions
        # ----------------------------------------------------

        (
            predictions,
            probabilities
        ) = generate_predictions(
            model,
            scaler,
            X_test
        )

        # ----------------------------------------------------
        # 4. Calculate metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            y_test,
            predictions,
            probabilities
        )

        # ----------------------------------------------------
        # 5. Classification report
        # ----------------------------------------------------

        create_classification_report(
            y_test,
            predictions
        )

        # ----------------------------------------------------
        # 6. Confusion matrix
        # ----------------------------------------------------

        create_confusion_matrix(
            y_test,
            predictions
        )

        # ----------------------------------------------------
        # 7. ROC curve
        # ----------------------------------------------------

        create_roc_curve(
            y_test,
            probabilities
        )

        # ----------------------------------------------------
        # 8. Metrics chart
        # ----------------------------------------------------

        create_metrics_chart(
            metrics
        )

        # ----------------------------------------------------
        # 9. Save CSV
        # ----------------------------------------------------

        save_metrics_csv(
            model,
            metrics
        )

        # ----------------------------------------------------
        # 10. Save text summary
        # ----------------------------------------------------

        save_evaluation_summary(
            model,
            metrics
        )

        # ----------------------------------------------------
        # 11. Final summary
        # ----------------------------------------------------

        display_final_summary(
            model,
            metrics
        )

        # ----------------------------------------------------
        # 12. Completion message
        # ----------------------------------------------------

        print_header(
            "EVALUATION COMPLETE"
        )

        print(
            "\n✓ Model evaluation completed successfully."
        )

        print(
            "✓ All evaluation metrics calculated."
        )

        print(
            "✓ Classification report generated."
        )

        print(
            "✓ Confusion matrix generated."
        )

        print(
            "✓ ROC curve generated."
        )

        print(
            "✓ Evaluation metrics chart generated."
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
            "  ✓ evaluation_results.csv"
        )

        print(
            "  ✓ classification_report.txt"
        )

        print(
            "  ✓ evaluation_summary.txt"
        )

        print(
            "  ✓ evaluation_confusion_matrix.png"
        )

        print(
            "  ✓ evaluation_roc_curve.png"
        )

        print(
            "  ✓ evaluation_metrics.png"
        )

        print_line("=")

    except FileNotFoundError as error:

        print_header(
            "FILE ERROR"
        )

        print(
            f"\n{error}"
        )

    except Exception as error:

        print_header(
            "EVALUATION ERROR"
        )

        print(
            f"\n{type(error).__name__}: {error}"
        )

        print(
            "\nPlease check:"
        )

        print(
            "1. preprocessing.py"
        )

        print(
            "2. train.py"
        )

        print(
            "3. credit_model.pkl"
        )

        print(
            "4. scaler.pkl"
        )

        print(
            "5. Dataset file"
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()