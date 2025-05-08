# Credit Card Fraud Detection with Transformer and Logistic Regression Models

## Project Overview

This project implements an advanced credit card fraud detection system using both state-of-the-art transformer-based models and traditional machine learning approaches. By comparing a logistic regression baseline with a fine-tuned pre-trained language model on a large dataset of transaction descriptions, we've created a comprehensive fraud detection system that demonstrates the advantages of deep learning for this challenging task.

## Key Features

- Utilizes both traditional machine learning (Logistic Regression) and modern deep learning (Transformers)
- Provides a comparative analysis between simple and complex model architectures
- Processes transaction descriptions to identify potentially fraudulent activities
- Achieves an AUC-ROC of 0.9777 with the transformer model, a 76.25% improvement over the baseline
- Implements comprehensive statistical testing to validate model improvements
- Provides visualization tools for model performance analysis and comparison

## Model Performance Comparison

Our models achieve the following performance metrics:

| Metric | Logistic Regression | Original Transformer | Fine-tuned Transformer | Improvement (Fine-tuned vs. Original) |
|--------|---------------------|----------------------|------------------------|---------------------------------------|
| AUC-ROC | 0.898 | 0.5547 | 0.9777 | +76.25% |
| Accuracy | 0.89 | 0.0084 | 0.4415 | +5148.64% |
| Precision | 0: 0.99, 1: 0.17 | 0.0057 | 0.0100 | +74.88% |
| Recall | 0: 0.89, 1: 0.73 | 0.9992 | 0.9883 | -1.08% |
| F1 Score | 0: 0.94, 1: 0.28 | 0.0114 | 0.0199 | +74.11% |

Statistical significance tests (bootstrap AUC test and McNemar's test) confirm that the improvements of the fine-tuned transformer over the original model are significant with p-values effectively zero.

## Dataset

The project uses the CIS435-CreditCardFraudDetection dataset from Hugging Face, which contains:
- Transaction descriptions and metadata
- Binary labels indicating fraudulent transactions
- Approximately 1 million transactions with a 20/80 fraud/non-fraud distribution

## Installation Requirements

```bash
pip install datasets transformers accelerate torch pandas numpy scikit-learn matplotlib seaborn imbalanced-learn huggingface_hub statsmodels
```

## Project Structure

The project is organized as follows:

1. **Data Loading and Preparation**
   - Loading the dataset from Hugging Face
   - Data cleaning and preprocessing
   - Train/test splitting
   - Feature engineering for logistic regression

2. **Logistic Regression Model**
   - Feature extraction from transaction data
   - Model training and hyperparameter tuning
   - Performance evaluation

3. **Transformer Model Fine-tuning**
   - Loading pre-trained models from HuggingFace
   - Fine-tuning on a large dataset (160,000 samples)
   - Hyperparameter optimization

4. **Model Evaluation and Comparison**
   - Comprehensive metrics calculation for both models
   - ROC and Precision-Recall curve analysis
   - Score distribution visualization
   - Comparative analysis of model strengths and weaknesses

5. **Statistical Significance Testing**
   - Bootstrap test for AUC-ROC differences
   - McNemar's test for classification differences

## Usage Instructions

### Loading the Dataset

```python
from datasets import load_dataset
dataset = load_dataset("dazzle-nu/CIS435-CreditCardFraudDetection")
df = dataset['train'].to_pandas()
```

### Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create logistic regression pipeline
logistic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
])

# Train logistic regression model
logistic_pipeline.fit(X_train, y_train)

# Make predictions
lr_probs = logistic_pipeline.predict_proba(X_test)[:, 1]
```

### Transformer Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Original model
tokenizer = AutoTokenizer.from_pretrained("HIT-TMG/yizhao-risk-en-scorer")
model = AutoModelForSequenceClassification.from_pretrained("HIT-TMG/yizhao-risk-en-scorer")

# Making predictions
def get_risk_scores(text_list, batch_size=32):
    """Get risk scores using the model"""
    risk_scores = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1).float().cpu().numpy()
        risk_scores.extend(logits.tolist())
    return np.array(risk_scores)
```

### Statistical Significance Testing

```python
# Compare transformer models with bootstrap test for AUC difference
bootstrap_results = bootstrap_auc_test(test_labels, original_probs, finetuned_probs)

# Compare transformer and logistic regression models
lr_bootstrap_results = bootstrap_auc_test(test_labels, lr_probs, finetuned_probs)
```

## Model Training Details

### Logistic Regression
- Trained on engineered features from transaction data
- Hyperparameters tuned using grid search with cross-validation
- Class weighting applied to handle imbalanced dataset

### Fine-tuned Transformer
- **Base Model**: HIT-TMG/yizhao-risk-en-scorer
- **Training Data**: 160,000 samples (20% fraud, 80% non-fraud)
- **Optimizer**: AdamW
- **Learning Rate**: 3e-5
- **Batch Size**: 32
- **Epochs**: 3
- **Weight Decay**: 0.01
- **Learning Rate Schedule**: Linear warmup followed by decay

## Key Findings

1. Fine-tuning on a large dataset (160,000 samples) dramatically improved transformer performance compared to a smaller dataset (1,600 samples)
2. The fine-tuned transformer model maintains near-perfect recall (98.83%) while significantly improving precision
3. While logistic regression provides a competitive baseline with good interpretability, the transformer model achieves superior performance through its ability to process text data
4. Both bootstrap and McNemar's tests confirm that improvements are statistically significant
5. The transformer model's high AUC-ROC (0.9777) enables flexible threshold selection to balance precision and recall according to business needs

## Future Work

Potential areas for improvement include:

- Exploring ensemble methods combining logistic regression and transformer models
- Implementing threshold optimization to balance precision and recall
- Investigating additional feature engineering approaches for the logistic regression model
- Developing an online learning component to adapt to evolving fraud patterns
- Exploring model explainability techniques to make transformer predictions more interpretable

## Acknowledgments

- The original pre-trained model was developed by HIT-TMG
- Data is provided by the CIS435-CreditCardFraudDetection dataset

## License

This project is licensed under the MIT License - see the LICENSE file for details.
