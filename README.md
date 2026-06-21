# Fraudulent Claim Detection

Machine learning project to classify insurance claims as fraudulent or legitimate using historical claim, policyholder, incident, and vehicle details. The work is based on a Global Insure business case where early fraud identification can reduce manual review effort and financial losses from late fraud detection.

## Project Objective

Build and evaluate classification models that can flag potentially fraudulent insurance claims early in the approval process while balancing fraud recall, false positives, and overall predictive performance.

## Dataset

The dataset contains 1,000 insurance claim records with 40 original columns, including:

- Policyholder details such as age, months as customer, education, occupation, hobbies, and relationship.
- Policy details such as policy state, combined single limit, deductible, annual premium, and bind date.
- Incident details such as incident type, severity, state, city, collision type, authorities contacted, and incident date.
- Claim amounts for injury, property, vehicle, and total claim.
- Target variable: `fraud_reported`.

The notebook expects `insurance_claims.csv` to be available in the project directory.

## Workflow

1. Loaded and inspected the insurance claims dataset.
2. Cleaned missing, redundant, empty, and invalid values.
3. Converted date columns to datetime format and corrected data types.
4. Performed a stratified 70-30 train-validation split.
5. Conducted exploratory data analysis on numerical and categorical variables.
6. Addressed class imbalance with `RandomOverSampler`.
7. Engineered and selected features using RFECV, statistical significance checks, VIF, and Random Forest feature importance.
8. Built and evaluated Logistic Regression and Random Forest models.
9. Tuned Random Forest hyperparameters with GridSearchCV.

## Models

- Logistic Regression
- Random Forest Classifier
- Tuned Random Forest Classifier

## Results

| Model | Validation Accuracy | Sensitivity / Recall | Specificity | Precision | F1 Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.797 | 0.716 | 0.823 | 0.570 | 0.635 |
| Tuned Random Forest | 0.820 | 0.743 | 0.845 | 0.611 | 0.671 |

The tuned Random Forest model performed slightly better than Logistic Regression on validation data and detected more fraudulent claims, but it also showed signs of overfitting due to the gap between training and validation accuracy.

## Key Insights

- Fraud labels were imbalanced, with legitimate claims forming the majority class.
- `months_as_customer` and `age` showed high correlation.
- `total_claim_amount` and `injury_claim` showed high correlation.
- Important predictive signals included insured hobbies, incident severity, incident type, property damage, incident state, auto model, and claim amount fields.
- RandomOverSampler balanced the training set to 526 records for each class.

## Repository Contents

```text
.
├── Fraudulent_Claim_Detection__Ganesh_Babu_Kamma_Harshit_Kumar_Vaish.ipynb
├── Fraudulent_Claim_detection_Ganesh_Babu_Kamma_Harshit_Kumar_Vaish.pdf
└── README.md
```

## How to Run

1. Clone this repository.
2. Add `insurance_claims.csv` to the project directory.
3. Install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn statsmodels
```

4. Open the notebook:

```bash
jupyter notebook Fraudulent_Claim_Detection__Ganesh_Babu_Kamma_Harshit_Kumar_Vaish.ipynb
```

5. Run all cells to reproduce the data cleaning, EDA, feature engineering, model training, tuning, and evaluation steps.

## Authors

- Harshit Kumar Vaish
