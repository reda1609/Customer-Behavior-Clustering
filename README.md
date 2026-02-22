# Customer Behavior Clustering — Credit Card Customer Segmentation

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Methodology](#methodology)
  - [Data Exploration](#data-exploration)
  - [Data Preprocessing](#data-preprocessing)
  - [Null Imputation](#null-imputation)
  - [Dimensionality Reduction & Visualization](#dimensionality-reduction--visualization)
  - [Clustering with Gaussian Mixture Model](#clustering-with-gaussian-mixture-model)
- [Cluster Descriptions & Business Insights](#cluster-descriptions--business-insights)
- [Key Results](#key-results)
- [Technologies Used](#technologies-used)

---

## Project Overview

This project analyzes credit card usage data to identify **distinct customer segments** using unsupervised machine learning. Rather than stopping at technical clustering, the analysis goes further to provide **actionable business strategies** for each customer group — enabling targeted marketing, risk management, and product development.

The workflow covers end-to-end data science: data loading, exploration, preprocessing, visualization, clustering, and business interpretation.

---

## Dataset

The dataset used is the **[Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)** from Kaggle. It contains usage behavior of ~8,950 active credit card holders over a period of time, summarized across 18 behavioral features.

| Feature | Description |
|---|---|
| `CUST_ID` | Unique credit card holder ID |
| `BALANCE` | Account balance available for purchases |
| `BALANCE_FREQUENCY` | How frequently the balance is updated (0–1) |
| `PURCHASES` | Total amount of purchases made |
| `ONEOFF_PURCHASES` | Maximum purchase amount in one transaction |
| `INSTALLMENTS_PURCHASES` | Amount of purchases done in installments |
| `CASH_ADVANCE` | Cash in advance given by the user |
| `PURCHASES_FREQUENCY` | How frequently purchases are made (0–1) |
| `ONEOFF_PURCHASES_FREQUENCY` | Frequency of one-off purchases (0–1) |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | Frequency of installment purchases (0–1) |
| `CASH_ADVANCE_FREQUENCY` | Frequency of cash advance transactions (0–1) |
| `CASH_ADVANCE_TRX` | Number of cash advance transactions |
| `PURCHASES_TRX` | Number of purchase transactions |
| `CREDIT_LIMIT` | Credit card limit |
| `PAYMENTS` | Total amount of payments made |
| `MINIMUM_PAYMENTS` | Minimum amount of payments made |
| `PRC_FULL_PAYMENT` | Percentage of full payment paid by user |
| `TENURE` | Tenure of credit card service (in months) |

---

## Project Structure

```
Customer-Behavior-Clustering/
├── main.ipynb          # Full analysis notebook (EDA → Clustering → Insights)
├── README.md           # Project documentation
└── data/
    └── CC GENERAL.csv  # Raw credit card dataset
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- Jupyter Notebook or VS Code with Jupyter extension

### Install Dependencies

```bash
pip install numpy pandas seaborn matplotlib scikit-learn plotly
```

### Run the Notebook

```bash
jupyter notebook main.ipynb
```

Or open `main.ipynb` directly in VS Code.

---

## Methodology

### Data Exploration

- Inspected data types, missing values, and basic statistics.
- Separated features into **discrete** (≤100 unique values) and **continuous** (>100 unique values) categories.
- Generated pair plots, histograms, box plots, and count plots to understand distributions and relationships.
- Observed that most continuous features are **heavily right-skewed**, and discrete features exhibit very low variance.

### Data Preprocessing

- **Dropped** `CUST_ID` (unique identifier with no predictive value).
- **Removed** the single row with a missing `CREDIT_LIMIT` value.
- Applied **log transformation** (`log(x + 1)`) to all continuous features to reduce skewness and normalize distributions.
- Verified improvement via post-transformation histograms, box plots, pair plots, and a correlation heatmap.

### Null Imputation

Handled missing values in `MINIMUM_PAYMENTS` with a two-step strategy:

1. **Zero-balance rule**: If `BALANCE == 0`, set `MINIMUM_PAYMENTS = 0`.
2. **Regression-based imputation**: Trained a `LinearRegression` model on non-null rows (using all other features as predictors) and imputed the remaining missing values with the model's predictions.

### Dimensionality Reduction & Visualization

- Used **t-SNE** (t-distributed Stochastic Neighbor Embedding) to project the high-dimensional feature space into 2D.
- The t-SNE scatter plot revealed **7 visually distinct clusters**, guiding the choice of `n_components=7` for the clustering model.

### Clustering with Gaussian Mixture Model

- Chose **Gaussian Mixture Model (GMM)** with `covariance_type='full'` because the clusters have different shapes and orientations — requiring per-cluster covariance matrices.
- Performed a **seed search** across 50 random initializations to select the best-performing seed (highest log-likelihood).
- Validated clustering quality by overlaying predicted cluster labels on the t-SNE visualization — the model correctly recovered all 7 visually apparent groups.

---

## Cluster Descriptions & Business Insights

| Cluster | Name | Size | Key Behavior | Suggested Strategy |
|:---:|---|:---:|---|---|
| **0** | Occasional One-Off Shoppers | 1,070 | One-off purchases only; no installments or cash advances; low frequency; 14% full-payment rate | Targeted promotions on one-off purchases; introduce installment options |
| **1** | Active All-Round Purchasers | 1,769 | Highest purchases & frequency (0.81); both one-off & installment; zero cash advances; highest credit limits; 26% full-payment rate | Premium rewards & loyalty programs; credit limit increases |
| **2** | Cash-Advance-Only Users | 2,039 | Zero purchases; rely entirely on cash advances; high balances; lowest credit limits among high-balance groups; 4% full-payment rate | Financial counseling; balance transfer options; incentivize purchase spending |
| **3** | High-Value Power Users | 1,021 | Use every feature (one-off, installment, cash advance); highest balances, credit limits, & payments; ~7.7 cash advance TRX; 7% full-payment rate | Premium services; monitor credit risk; higher limits with guardrails |
| **4** | One-Off + Cash Advance Users | 804 | Occasional one-off purchases combined with heavy cash advances; high balances; 6% full-payment rate | Structured repayment plans; shift behavior from cash advances to purchases |
| **5** | Budget-Conscious Installment Shoppers | 1,788 | Installment purchases only; lowest balances & credit limits; **highest full-payment rate (31%)** — most financially disciplined | Reward responsible behavior; gradual credit limit increases; installment-friendly offers |
| **6** | Installment + Cash Advance Users | 458 | Installment purchases + heaviest cash advance usage; highest minimum payments; 5% full-payment rate | Debt consolidation products; lower-rate balance transfers; close monitoring |

---

## Key Results

- **7 distinct customer segments** were identified, each with clearly differentiated spending and repayment behaviors.
- **Gaussian Mixture Model** with full covariance successfully captured clusters of varying shapes and densities.
- **t-SNE** visualization confirmed that the clustering in high-dimensional space aligns with the natural structure visible in 2D.
- Business strategies were derived for each segment, covering **retention**, **risk management**, **cross-selling**, and **financial support** initiatives.

---

## Technologies Used

| Library | Purpose |
|---|---|
| **NumPy** | Numerical operations |
| **Pandas** | Data manipulation & analysis |
| **Seaborn** | Statistical visualizations |
| **Matplotlib** | Plotting framework |
| **Scikit-learn** | GMM clustering, t-SNE, Linear Regression, train/test split, metrics |
| **Plotly** | Interactive visualizations with dropdown controls |