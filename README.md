# 🌐CodeAlpha_Credit_Scoring_Model--Task1
# 💳 Credit Scoring Model

## 📌 PROJECT OVERVIEW

This project is a Machine Learning based Credit Scoring Model developed to predict the likelihood of credit card payment default.

The system uses historical credit card customer data and applies classification algorithms to analyze customer financial and repayment behavior.

Multiple Machine Learning models are trained, evaluated, and compared to select the best-performing model for credit risk prediction.

---

## 🎯 OBJECTIVES

The main objectives of this project are:

- To develop a Machine Learning model for credit default prediction.
- To preprocess and prepare historical credit card data.
- To train multiple classification algorithms.
- To compare model performance using standard evaluation metrics.
- To identify the best-performing model.
- To evaluate the final model using different performance metrics.
- To generate meaningful visualizations for model analysis.
- To save the final trained model for future predictions.
- To build a reusable credit-risk prediction system.

---

## 🛠 TECHNOLOGIES USED

### Programming Language

- Python

### Libraries & Frameworks

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

### Development Tools

- Jupyter Notebook
- VS Code
- Git
- GitHub

### Machine Learning Algorithms

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

## 📄 WEBSITE PAGES

This project is a Machine Learning application and does not contain traditional website pages.

The main project components are:

- Credit_Scoring_Model.ipynb – Complete Machine Learning workflow
- train.py – Model training and comparison
- predict.py – Credit risk prediction
- evaluation.py – Model evaluation
- visualization.py – Visualization generation
- preprocessing.py – Data preprocessing

---

## ⭐️ FEATURES

- Credit card default prediction
- Data preprocessing and cleaning
- Feature and target separation
- Train-test data splitting
- Feature scaling
- Multiple Machine Learning models
- Model performance comparison
- Accuracy calculation
- Precision calculation
- Recall calculation
- F1-Score calculation
- ROC-AUC evaluation
- Confusion Matrix
- ROC Curve
- Feature Importance visualization
- Model comparison visualization
- Final model selection
- Saved Machine Learning model
- Saved feature scaler
- New customer credit-risk prediction
- Classification report generation
- Final project conclusion report



##### 📂 PROJECT STRUCTURE

```text
Credit-Scoring-Project/
│
├── 📁 data/
│   └── default_of_credit_card_clients.xls
│
├── 📁 models/
│   ├── credit_model.pkl
│   └── scaler.pkl
│
├── 📁 notebooks/
│   └── Credit_Scoring_Model.ipynb
│
├── 📁 outputs/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── evaluation_confusion_matrix.png
│   ├── evaluation_metrics.png
│   ├── evaluation_results.csv
│   ├── evaluation_roc_curve.png
│   ├── evaluation_summary.txt
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── final_classification_metrics.csv
│   ├── final_classification_report.txt
│   ├── final_conclusion.txt
│   ├── final_confusion_matrix.png
│   ├── final_confusion_matrix_table.csv
│   ├── final_confusion_matrix_values.csv
│   ├── final_evaluation_metrics.csv
│   ├── final_model_information.csv
│   ├── final_model_selection.csv
│   ├── final_project_summary.csv
│   ├── final_roc_comparison.csv
│   ├── final_roc_curve_comparison.png
│   ├── model_comparison.png
│   ├── model_comparison_final.csv
│   ├── model_performance.png
│   ├── model_results.csv
│   ├── roc_curve.png
│   ├── saved_model_verification.csv
│   └── saved_model_verification.txt
│
├── 📁 src/
│   ├── evaluation.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── visualization.py
│   │
│   └── 📁 pycache/
│       └── preprocessing.cpython-314.pyc
```

---
## ▶️ HOW TO RUN
Follow these steps to run the Credit Scoring Model project locally.
# 1. Clone the Repository
git clone YOUR_GITHUB_REPOSITORY_URL
cd Credit-Scoring-Project
# 2. Install Required Libraries
Install the required Python libraries:
' ' 'bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib openpyxl xlrd
# 3.Run Model Training
From the project root directory:
' ' 'bash
python scr/train.py 
This will:

Load the credit card dataset.
Preprocess the data.
Split the data into training and testing sets.
Train Logistic Regression, Decision Tree, and Random Forest models.
Compare model performance.
Select the best-performing model.
Save the trained model and scaler.

The saved files will be available in:
models/
├── credit_model.pkl
└── scaler.pkl
# 4. Run Prediction
After training the model.run:
' ' 'bash
python scr/predict.py
This will load the saved model and generate:
-Credit Risk
-Default Probability
-Prediction Class
# 5.Run Model Evaluation
Run:
' ' 'bash
python scr/evaluation.py
This evaluation the trained model and generate evaluation results 
# 6. Generate Visualizations
Run:
' ' 'bash
python scr/visualization.py
This generates model comparison, confusion matrix, ROC curve, feature importance, and other visualization files inside the outputs/ folder.
# 7. Run the Jupyter Notebook
Start Jupyter Notebook:
' ' 'bash
jupyter notebook
Then open 
notebooks/Credit_Scoring_Model.ipynb
Run the notebook cells sequentially to reproduce the complete Machine Learning workflow.
## Complete Workflow
```text
 Dataset
   ↓
Data Preprocessing
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Model Evaluation
   ↓
Visualization
   ↓
Model Saving
   ↓
Credit Risk Prediction
```
---
## 🎓LEARNING OUTCOMES
Through this project, I gained practical experience in:
* Python programming for Machine Learning.
* Data loading and preprocessing.
* Data cleaning and feature preparation.
* Train-test data splitting.
* Feature scaling using StandardScaler.
* Classification Machine Learning algorithms.
* Logistic Regression.
* Decision Tree Classifier.
* Random Forest Classifier.
* Model performance comparison.
* Model selection using evaluation metrics.
* Accuracy, Precision, Recall and F1-Score analysis.
* ROC-AUC evaluation.
* Confusion Matrix analysis.
* ROC Curve visualization.
* Feature Importance analysis.
* Model persistence using Joblib.
* Building a reusable prediction system.
* Jupyter Notebook-based Machine Learning workflow.
* Generating CSV, TXT and PNG project reports.

---

## 👨‍💻 AUTHOR
Harshit Singh
Credit Scoring Machine Learning Project
This project was developed to demonstrate practical skills in:
Python
Machine Learning
Data Analysis
Data Visualization
Predictive Analytics

---

## 📜 LICENSE
This project is intended for educational and internship purposes.
You may use, modify, and learn from the source code for educational purposes.
For production or commercial use, appropriate validation, data licensing, security, privacy, fairness analysis, and regulatory 
requirements should be considered.
