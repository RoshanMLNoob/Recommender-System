# Recommender-System
# Anime Recommender System

A custom end-to-end anime recommendation engine built from scratch in Python using NumPy and Pandas[cite: 2, 4, 5]. This project features both an **Alternating Least Squares (ALS) Collaborative Filtering** module[cite: 4] and a **Content-Based Filtering** module[cite: 5] to provide personalized recommendations using real-world data.

## Dataset

This project uses the **Cooper Union Anime Recommendations Database**:
* **Source**: [Kaggle - Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)[cite: 1]
* **Files**: `anime.csv` (metadata and anime names) and `rating.csv` (user interaction and rating records)[cite: 2].

## How to Use

### 0. Using the Installed PyPI Package in Your Own Projects

Once you have installed the package via pip:
```bash
pip install anime-recommender-system
```
## Project Architecture

### 1. Collaborative Filtering (`recommender_sys_ColabFilter.py`)
* **Matrix Factorization**: Implements a custom Alternating Least Squares (ALS) algorithm from scratch with configurable latent factors ($k$)[cite: 4].
* **Optimized Solvers**: Utilizes analytical least squares updates via `np.linalg.solve` with LU decomposition for fast convergence[cite: 4].
* **Regularization**: Integrates $L_2$ regularization ($\lambda$) to prevent overfitting[cite: 4].
* **Cold-Start Support**: Includes dynamic latent vector computation for new users (`Show_Result_NewUser`)[cite: 4].

### 2. Content-Based Filtering (`recommender_sys_ContentBased.py`)
* **Feature Weighting**: Computes feature weights using regularized linear regression on rated items ($W = (X_R^T X_R + \lambda I)^{-1} X_R^T y_R$)[cite: 5].
* **Scoring & Ranking**: Evaluates model cost and ranks unrated items based on feature similarities[cite: 5].

## Pipeline & Preprocessing (`anime_recommendation.py`)

* Loads and parses raw dataset files using Pandas[cite: 2].
* Constructs a user-item interaction matrix $Y$[cite: 2].
* Applies a soft-imputation strategy for unrated items, scaling missing values relative to individual user means ($\alpha = 0.5$)[cite: 2].
* Trains the ALS model (e.g., $k=50$, $\lambda=0.05$) across multiple epochs to generate top-$N$ personalized recommendations[cite: 2].

## Project Structure

```text
Recommender_System/
├── anime_recommendation.py            # Main execution pipeline and training script[cite: 2]
├── recommender_sys_ColabFilter.py     # Custom ALS collaborative filtering engine[cite: 4]
├── recommender_sys_ContentBased.py    # Content-based recommendation module[cite: 5]
└── data/                              # Dataset directory (anime.csv, rating.csv)[cite: 2]
