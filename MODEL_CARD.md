# Model Card — Financial Complaint Risk Intelligence System

## Model Details
- Author: Sanjeev Kumar | IIT Bombay
- Date: May 2026
- Version: 1.0
- Type: Text Classification + Triage

## Intended Use
- Route complaint narratives to product queues
- Prioritize complaints for SLA risk review
- Support operations teams in complaint handling
- Portfolio/demo project — NOT for production use

## NOT Intended For
- Automated complaint denial or regulatory action
- Replacing human review of serious complaints
- Production deployment without further validation

## Training Data
- CFPB Consumer Complaint Database (public)
- 33,980 training complaints (2015-2025)
- Only complaints WITH consumer narrative
- US financial product complaints only

## Evaluation Data
- 7,282 test complaints (Jul 2025 - May 2026)
- Chronological split (not random!)
- Never used during training or tuning

## Task 1: Product Classification
| Model | Test Macro-F1 | Decision |
|-------|--------------|---------|
| TF-IDF + LR | 0.7643 | Champion |
| TF-IDF + SVM | 0.7677 | Strong baseline |
| Word2Vec + XGBoost | 0.6352 | Underperformed |
| Frozen BERT + LR | 0.6737 | Underperformed |
| Fine-tuned DistilBERT | 0.7722 | Challenger |

## Per-Class F1 (Champion Model)
| Product | F1 |
|---------|-----|
| Credit Reporting | 0.93 |
| Mortgage | 0.84 |
| Bank Account | 0.78 |
| Money Transfer | 0.71 |
| Loans | 0.72 |
| Debt Collection | 0.70 |
| Credit Card | 0.68 |

## Task 2: Priority Review Triage
- Positive rate: 1.07% (untimely response)
- Test PR-AUC: 0.0470 (4.7x random baseline)
- Recall@Top 10%: 35.6% of all priority cases
- Lift@Top 10%: 3.6x over random review

## Prediction-Time Features
Allowed:
- Complaint narrative text
- Submission channel
- State
- Date received

Excluded (leakage risk!):
- Company response to consumer
- Timely response flag
- Consumer disputed flag
- Any post-resolution fields

## Known Limitations
1. CFPB is US-only — not India-specific
2. Only 24.3% have narratives (selection bias)
3. Risk label is proxy (untimely response)
4. Not representative per CFPB warning
5. Recent complaints may have incomplete outcomes
6. Fine-tuned BERT gave only +0.007 F1 lift
7. XGBoost triage used smaller feature set

## Ethical Considerations
- CFPB scrubs PII before publishing narratives
- Model is for prioritization only
- Not for automated adverse action
- Not validated for demographic fairness
