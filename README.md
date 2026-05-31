# Financial Complaint Risk Intelligence System

**Author:** Sanjeev Kumar | IIT Bombay M.Sc Mathematics 2025
**Domain:** Banking NLP — Complaint Routing & Priority Triage
**Dataset:** [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) — 200K rows
**Target Roles:** American Express, JPMorgan Chase, Goldman Sachs, Razorpay, PhonePe

---

## Business Problem

Financial institutions receive thousands of customer complaints daily. Two critical tasks:

1. **Route complaints** to the right product team automatically
2. **Prioritize complaints** likely to breach response SLAs or require urgent review

Wrong routing = delays + customer frustration  
Missed priority complaints = regulatory risk + SLA breach

---

## Solution

An end-to-end NLP pipeline that:
- Classifies complaint narratives into 7 product categories
- Scores each complaint for priority review probability
- Monitors emerging complaint themes
- Provides an interactive dashboard for operations teams

---

## Portfolio Connection

| Project | Risk Area | Key Method |
|---------|-----------|-----------|
| Fraud Detection | Transaction risk | XGBoost, PR-AUC |
| Credit Risk Scorecard | Borrower default risk | WOE/IV, ECL=PD×LGD×EAD |
| **Complaint NLP Intelligence** | **Operational/regulatory risk** | **BERT, TF-IDF, Triage** |

---

## Dataset

- Source: CFPB Consumer Complaint Database (public, US financial regulator)
- Size: 200,000 complaints loaded (full dataset = 3M+)
- Narratives available: 48,815 rows (24.4%) — consent required
- Date range: 2011–2026 (narratives from March 2015 onward)

---

## Label Design

### Task 1: Product Classification (7 classes)
Grouped 20 raw CFPB product categories:

| Clean Label | Raw Products |
|------------|-------------|
| Credit Reporting | "Credit reporting...", "Credit reporting or other..." |
| Debt Collection | "Debt collection", "Debt or credit management" |
| Credit Card | "Credit card", "Credit card or prepaid card", "Prepaid card" |
| Bank Account | "Checking or savings account", "Bank account or service" |
| Loans | "Student loan", "Vehicle loan", "Consumer Loan", "Payday loan..." |
| Mortgage | "Mortgage" |
| Money Transfer | "Money transfer...", "Money transfers" |

### Task 2: Priority Review Triage
```
is_priority_review = 1 if Timely response == "No"
                   = 0 otherwise

Positive rate: 1.07% (523 cases out of 48,815)
```

**Important:** This is a **proxy label**, not ground-truth complaint severity. Untimely response reflects SLA/operational risk, not complaint content severity per se.

---

## Time-Based Split (CRITICAL!)

Never random split complaint data — language patterns repeat across time!

| Split | Rows | Date Range |
|-------|------|-----------|
| Train | 34,170 | 2015 – 2025 |
| Val | 7,294 | 2025 |
| Test | 7,294 | 2025 – 2026 |

---

## Modeling Experiments

### Product Classification

| Model | Val Macro-F1 | Test Macro-F1 | Decision |
|-------|-------------|--------------|---------|
| **TF-IDF + LR** | **0.7696** | **0.7652** | **Champion ✅** |
| TF-IDF + Linear SVM | 0.7803 | 0.7591 | Strong baseline |
| Word2Vec + XGBoost | 0.6971 | 0.6383 | Underperformed ❌ |
| Frozen BERT CLS + LR | 0.6507 | 0.6716 | Underperformed ❌ |
| Frozen BERT Mean + LR | 0.6703 | 0.6737 | Underperformed ❌ |
| Fine-tuned DistilBERT | 0.7968 | 0.7722 | Challenger |

**Key insight:** TF-IDF beats frozen BERT because product categories are **keyword-driven** (credit report, mortgage, debt collector). Fine-tuned DistilBERT gives only **+0.007 lift** — not enough to justify GPU dependency, latency, and governance overhead.

### Priority Review Triage

| Model | PR-AUC | Note |
|-------|--------|------|
| Random baseline | 0.0107 | 1.07% positive rate |
| TF-IDF + LR (balanced) | TBD | Run notebook 05 |
| XGBoost (tuned spw) | TBD | Run notebook 05 |

**Metrics used:** PR-AUC, Recall@5%, Recall@10%, Precision@K, Lift@K
**NOT accuracy** — 97% accuracy by predicting all zeros is meaningless!

---

## Leakage Controls

**Allowed at prediction time (available at complaint submission):**
- Complaint narrative text
- Submission channel (Web/Phone/etc)
- State
- Date received

**Excluded (post-resolution — leakage risk!):**
- Company response to consumer
- Timely response flag
- Consumer disputed flag
- Date sent to company
- Any resolution fields

---

## Project Structure

```
financial_complaint_intelligence/
├── notebooks/
│   ├── 01_eda.ipynb                    EDA + data understanding
│   ├── 02_baseline_models.ipynb        TF-IDF + Word2Vec baselines
│   ├── 03_bert_embeddings.ipynb        Frozen BERT embeddings
│   ├── 04_bert_finetuning.ipynb        Fine-tune DistilBERT (GPU!)
│   └── 05_complaint_triage.ipynb       Priority review triage model
├── src/
│   ├── text_preprocessor.py            Data loading + cleaning
│   └── evaluation.py                   Metrics (PR-AUC, Recall@K etc)
├── app/
│   └── app.py                          Streamlit dashboard (5 pages)
├── models/                             Saved models (git-ignored)
├── reports/                            Charts + JSON results
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# Clone
git clone https://github.com/17sanjeev/financial-complaint-intelligence
cd financial-complaint-intelligence

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order (01 → 05)
# Notebook 04 requires GPU (Google Colab T4 recommended)

# Launch dashboard
streamlit run app/app.py
```

---

## Known Limitations

| Limitation | Impact | Fix in Production |
|-----------|--------|-----------------|
| CFPB is US-only | Not India-specific | Indian regulatory complaint data |
| 24.4% narrative coverage | Selection bias | Encourage narrative submission |
| Proxy risk label | Not true severity | Manual labeling + expert annotation |
| Marginal BERT lift | Keywords dominate | Domain-specific pre-training (FinBERT) |
| Recent data censoring | Incomplete outcomes | 60-day maturity window |
| Not representative | CFPB explicitly warns this | Multiple data sources |

---

## Interview Story (90 seconds)

> "I built a Financial Complaint Risk Intelligence System using CFPB complaint narratives. The system routes complaints to product queues and prioritizes those likely to breach response SLAs. I used chronological validation to simulate future performance. For product routing, I benchmarked TF-IDF, SVM, Word2Vec, frozen DistilBERT, and fine-tuned DistilBERT. Fine-tuned DistilBERT achieved Test Macro-F1=0.772 but only improved 0.7 points over TF-IDF + LR. Given the marginal lift and the added latency, governance, and deployment complexity, I selected TF-IDF + LR as champion. For triage, I modeled a rare-event target (1.07% positive) using PR-AUC, Recall@K, and Lift@K rather than accuracy. The key lesson: I didn't blindly use BERT — I benchmarked systematically and selected models based on generalization and business value."

---

*Sanjeev Kumar | M.Sc Mathematics, IIT Bombay | Associate Consultant DS @ Virtusa*
*Applying to Banking & Fintech DS roles | July 2026*
