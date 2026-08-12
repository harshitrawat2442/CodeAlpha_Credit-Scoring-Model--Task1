import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
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
# CREDIT SCORING MODEL TRAINING SYSTEM
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

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "default_of_credit_card_clients.xls"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "credit_model.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "model_results.csv"
)

COMPARISON_GRAPH_PATH = os.path.join(
    OUTPUT_DIR,
    "model_comparison.png"
)


# ============================================================
# 2. CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 3. DISPLAY HELPERS
# ============================================================

def print_line(character="=", length=70):
    print(character * length)


def print_header(title):

    print("\n")

    print_line("=")

    print(
        f"{title:^70}"
    )

    print_line("=")


# ============================================================
# 4. LOAD DATASET
# ============================================================

def load_dataset():

    print_header(
        "LOADING DATASET"
    )

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_PATH}"
        )

    print(
        "\nLoading dataset..."
    )

    df = load_data(
        DATA_PATH
    )

    print(
        "✓ Dataset loaded successfully!"
    )

    print(
        f"✓ Dataset shape: {df.shape}"
    )

    return df


# ============================================================
# 5. PREPROCESS DATA
# ============================================================

def preprocess_dataset(df):

    print_header(
        "DATA PREPROCESSING"
    )

    X, y = clean_data(
        df
    )

    print(
        f"\n✓ Features shape: {X.shape}"
    )

    print(
        f"✓ Target shape:   {y.shape}"
    )

    return X, y


# ============================================================
# 6. SPLIT AND SCALE DATA
# ============================================================

def prepare_data(X, y):

    print_header(
        "TRAIN TEST SPLIT"
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
        f"\n✓ Training samples: {len(X_train)}"
    )

    print(
        f"✓ Testing samples : {len(X_test)}"
    )

    print(
        f"✓ Training features: {X_train.shape[1]}"
    )

    print(
        "✓ Data splitting and scaling completed."
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
# 7. DEFINE MACHINE LEARNING MODELS
# ============================================================

def create_models():

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    }

    return models


# ============================================================
# 8. TRAIN AND EVALUATE MODELS
# ============================================================

def train_models(
    models,
    X_train,
    X_test,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
):

    print_header(
        "MODEL TRAINING AND EVALUATION"
    )

    results = []

    trained_models = {}

    for model_name, model in models.items():

        print(
            f"\nTraining: {model_name}"
        )

        print(
            "-" * 50
        )

        # ----------------------------------------------------
        # Logistic Regression uses scaled data
        # ----------------------------------------------------

        if model_name == "Logistic Regression":

            model.fit(
                X_train_scaled,
                y_train
            )

            predictions = model.predict(
                X_test_scaled
            )

            probabilities = model.predict_proba(
                X_test_scaled
            )[:, 1]

        # ----------------------------------------------------
        # Tree-based models use original data
        # ----------------------------------------------------

        else:

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            probabilities = model.predict_proba(
                X_test
            )[:, 1]

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append({

            "Model": model_name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1-Score": f1,

            "ROC-AUC": roc_auc
        })

        trained_models[
            model_name
        ] = model

        # ----------------------------------------------------
        # Display metrics
        # ----------------------------------------------------

        print(
            f"Accuracy  : {accuracy * 100:.2f}%"
        )

        print(
            f"Precision : {precision * 100:.2f}%"
        )

        print(
            f"Recall    : {recall * 100:.2f}%"
        )

        print(
            f"F1-Score  : {f1 * 100:.2f}%"
        )

        print(
            f"ROC-AUC   : {roc_auc:.4f}"
        )

        print(
            "✓ Training completed."
        )

    return (
        results,
        trained_models
    )


# ============================================================
# 9. CREATE RESULTS DATAFRAME
# ============================================================

def create_results_dataframe(
    results
):

    print_header(
        "MODEL PERFORMANCE COMPARISON"
    )

    results_df = pd.DataFrame(
        results
    )

    # Convert metrics to percentage
    display_df = results_df.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ]:

        display_df[column] = (
            display_df[column] * 100
        ).round(2)

    display_df["ROC-AUC"] = (
        display_df["ROC-AUC"]
        .round(4)
    )

    print()

    print(
        display_df.to_string(
            index=False
        )
    )

    return results_df


# ============================================================
# 10. SAVE MODEL RESULTS
# ============================================================

def save_results(
    results_df
):

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print(
        f"\n✓ Model results saved:"
    )

    print(
        RESULTS_PATH
    )


# ============================================================
# 11. CREATE MODEL COMPARISON GRAPH
# ============================================================

def create_model_comparison(
    results_df
):

    print_header(
        "GENERATING MODEL COMPARISON GRAPH"
    )

    model_names = results_df[
        "Model"
    ].tolist()

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ]

    x_positions = range(
        len(model_names)
    )

    width = 0.15

    plt.figure(
        figsize=(12, 7)
    )

    for index, metric in enumerate(metrics):

        values = results_df[
            metric
        ].tolist()

        positions = [
            x + (
                index - 2
            ) * width
            for x in x_positions
        ]

        bars = plt.bar(
            positions,
            values,
            width=width,
            label=metric
        )

        # Add metric values
        for bar, value in zip(
            bars,
            values
        ):

            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    plt.xticks(
        list(x_positions),
        model_names
    )

    plt.ylim(
        0,
        1.05
    )

    plt.ylabel(
        "Score"
    )

    plt.xlabel(
        "Machine Learning Model"
    )

    plt.title(
        "Credit Scoring Model Comparison"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        COMPARISON_GRAPH_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        f"\n✓ Model comparison graph saved:"
    )

    print(
        COMPARISON_GRAPH_PATH
    )


# ============================================================
# 12. SELECT BEST MODEL
# ============================================================

def select_best_model(
    results_df,
    trained_models
):

    print_header(
        "BEST MODEL SELECTION"
    )

    # --------------------------------------------------------
    # F1-Score is used because the dataset is imbalanced
    # --------------------------------------------------------

    best_index = results_df[
        "F1-Score"
    ].idxmax()

    best_model_name = results_df.loc[
        best_index,
        "Model"
    ]

    best_model = trained_models[
        best_model_name
    ]

    best_row = results_df.loc[
        best_index
    ]

    print(
        f"\n✓ Best Model: {best_model_name}"
    )

    print(
        f"\n  Accuracy  : "
        f"{best_row['Accuracy'] * 100:.2f}%"
    )

    print(
        f"  Precision : "
        f"{best_row['Precision'] * 100:.2f}%"
    )

    print(
        f"  Recall    : "
        f"{best_row['Recall'] * 100:.2f}%"
    )

    print(
        f"  F1-Score  : "
        f"{best_row['F1-Score'] * 100:.2f}%"
    )

    print(
        f"  ROC-AUC   : "
        f"{best_row['ROC-AUC']:.4f}"
    )

    return (
        best_model_name,
        best_model
    )


# ============================================================
# 13. SAVE BEST MODEL
# ============================================================

def save_best_model(
    best_model,
    scaler
):

    print_header(
        "SAVING BEST MODEL"
    )

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    print(
        "\n✓ Best model saved successfully!"
    )

    print(
        f"✓ Model: {MODEL_PATH}"
    )

    print(
        f"✓ Scaler: {SCALER_PATH}"
    )


# ============================================================
# 14. FINAL TRAINING SUMMARY
# ============================================================

def display_final_summary(
    best_model_name,
    results_df
):

    best_row = results_df[
        results_df["Model"]
        == best_model_name
    ].iloc[0]

    print_header(
        "TRAINING COMPLETE"
    )

    print(
        "\n✓ All models trained successfully."
    )

    print(
        "✓ All evaluation metrics calculated."
    )

    print(
        "✓ Model comparison generated."
    )

    print(
        "✓ Best model selected."
    )

    print(
        "✓ Best model saved."
    )

    print(
        "\nFINAL BEST MODEL"
    )

    print(
        "-" * 50
    )

    print(
        f"Model     : {best_model_name}"
    )

    print(
        f"Accuracy  : "
        f"{best_row['Accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{best_row['Precision'] * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{best_row['Recall'] * 100:.2f}%"
    )

    print(
        f"F1-Score  : "
        f"{best_row['F1-Score'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC   : "
        f"{best_row['ROC-AUC']:.4f}"
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"  ✓ {MODEL_PATH}"
    )

    print(
        f"  ✓ {SCALER_PATH}"
    )

    print(
        f"  ✓ {RESULTS_PATH}"
    )

    print(
        f"  ✓ {COMPARISON_GRAPH_PATH}"
    )

    print_line("=")


# ============================================================
# 15. MAIN PROGRAM
# ============================================================

def main():

    print_header(
        "CREDIT SCORING MODEL TRAINING"
    )

    try:

        # ----------------------------------------------------
        # Step 1 - Load dataset
        # ----------------------------------------------------

        df = load_dataset()

        # ----------------------------------------------------
        # Step 2 - Clean data
        # ----------------------------------------------------

        X, y = preprocess_dataset(
            df
        )

        # ----------------------------------------------------
        # Step 3 - Split and scale
        # ----------------------------------------------------

        (
            X_train,
            X_test,
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            scaler
        ) = prepare_data(
            X,
            y
        )

        # ----------------------------------------------------
        # Step 4 - Create models
        # ----------------------------------------------------

        models = create_models()

        print_header(
            "MODELS SELECTED"
        )

        for model_name in models:

            print(
                f"  ✓ {model_name}"
            )

        # ----------------------------------------------------
        # Step 5 - Train models
        # ----------------------------------------------------

        (
            results,
            trained_models
        ) = train_models(
            models,
            X_train,
            X_test,
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test
        )

        # ----------------------------------------------------
        # Step 6 - Create results table
        # ----------------------------------------------------

        results_df = create_results_dataframe(
            results
        )

        # ----------------------------------------------------
        # Step 7 - Save results
        # ----------------------------------------------------

        save_results(
            results_df
        )

        # ----------------------------------------------------
        # Step 8 - Create comparison graph
        # ----------------------------------------------------

        create_model_comparison(
            results_df
        )

        # ----------------------------------------------------
        # Step 9 - Select best model
        # ----------------------------------------------------

        (
            best_model_name,
            best_model
        ) = select_best_model(
            results_df,
            trained_models
        )

        # ----------------------------------------------------
        # Step 10 - Save best model
        # ----------------------------------------------------

        save_best_model(
            best_model,
            scaler
        )

        # ----------------------------------------------------
        # Step 11 - Final summary
        # ----------------------------------------------------

        display_final_summary(
            best_model_name,
            results_df
        )

    except Exception as error:

        print_header(
            "TRAINING ERROR"
        )

        print(
            f"\n{error}"
        )

        print(
            "\nPlease check the dataset, preprocessing code,"
        )

        print(
            "model files, and Python dependencies."
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()