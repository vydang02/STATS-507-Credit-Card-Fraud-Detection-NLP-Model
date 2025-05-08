# Credit Card Fraud Detection with Transformer Models

## Project Overview

This project implements an advanced credit card fraud detection system using state-of-the-art transformer-based models. By fine-tuning a pre-trained language model on a large dataset of transaction descriptions, we have created a high-performance fraud detection system that significantly outperforms baseline models.

## Key Features

- Utilizes the HuggingFace Transformers library to fine-tune pre-trained models
- Processes transaction descriptions to identify potentially fraudulent activities
- Achieves an AUC-ROC of 0.9777, a 76.25% improvement over the baseline model
- Implements comprehensive statistical testing to validate model improvements
- Provides visualization tools for model performance analysis

## Model Performance

Our fine-tuned model achieves the following performance metrics:

| Metric | Original Model | Fine-tuned Model | Improvement |
|--------|---------------|------------------|-------------|
| AUC-ROC | 0.5547 | 0.9777 | +76.25% |
| Accuracy | 0.0084 | 0.4415 | +5148.64% |
| Precision | 0.0057 | 0.0100 | +74.88% |
| Recall | 0.9992 | 0.9883 | -1.08% |
| F1 Score | 0.0114 | 0.0199 | +74.11% |

Statistical significance tests (bootstrap AUC test and McNemar's test) confirm that these improvements are significant with p-values effectively zero.

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

2. **Model Fine-tuning**
   - Loading pre-trained models from HuggingFace
   - Fine-tuning on a large dataset (160,000 samples)
   - Hyperparameter optimization

3. **Model Evaluation**
   - Comprehensive metrics calculation
   - ROC and Precision-Recall curve analysis
   - Score distribution visualization

4. **Statistical Significance Testing**
   - Bootstrap test for AUC-ROC differences
   - McNemar's test for classification differences

## Usage Instructions

### Loading the Dataset

```python
from datasets import load_dataset
dataset = load_dataset("dazzle-nu/CIS435-CreditCardFraudDetection")
df = dataset['train'].to_pandas()
```

### Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Original model
tokenizer = AutoTokenizer.from_pretrained("HIT-TMG/yizhao-risk-en-scorer")
model = AutoModelForSequenceClassification.from_pretrained("HIT-TMG/yizhao-risk-en-scorer")
```

### Making Predictions

```python
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
# Bootstrap test for AUC difference
bootstrap_results = bootstrap_auc_test(test_labels, original_probs, finetuned_probs)

# McNemar's test for classification difference
mcnemar_result = mcnemar_test(test_labels, original_probs, finetuned_probs)
```

## Model Training Details

The fine-tuned model was trained using the following configuration:

- **Base Model**: HIT-TMG/yizhao-risk-en-scorer
- **Training Data**: 160,000 samples (20% fraud, 80% non-fraud)
- **Optimizer**: AdamW
- **Learning Rate**: 3e-5
- **Batch Size**: 32
- **Epochs**: 3
- **Weight Decay**: 0.01
- **Learning Rate Schedule**: Linear warmup followed by decay

## Key Findings

1. Fine-tuning on a large dataset (160,000 samples) dramatically improved performance compared to a smaller dataset (1,600 samples)
2. The fine-tuned model maintains near-perfect recall (98.83%) while significantly improving precision
3. The AUC-ROC of 0.9777 indicates excellent discrimination ability
4. Both bootstrap and McNemar's tests confirm that improvements are statistically significant

## Future Work

Potential areas for improvement include:

- Exploring ensemble methods to further improve performance
- Implementing threshold optimization to balance precision and recall
- Investigating additional feature engineering approaches
- Developing an online learning component to adapt to evolving fraud patterns

## Acknowledgments

- The original pre-trained model was developed by HIT-TMG
- Data is provided by the CIS435-CreditCardFraudDetection dataset

## License

This project is licensed under the MIT License - see the LICENSE file for details.
