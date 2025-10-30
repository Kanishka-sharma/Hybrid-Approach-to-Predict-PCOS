# PCOS Detection - ML vs Hybrid (Autoencoder + Random Forest)

## Project Overview
Polycystic Ovary Syndrome (PCOS) is a common endocrine disorder.  
This project compares **traditional ML models** against a **hybrid deep-learning + ML pipeline** (Autoencoder for feature extraction followed by Random Forest) to detect PCOS from **clinical, biochemical, and lifestyle features** (Kerala Hospitals dataset - *541 samples, 44 features*).

---

## Objective
Compare multiple ML models and a **hybrid Autoencoder + Random Forest** pipeline using **stratified 5-fold cross-validation**, and report **reliable, reproducible performance metrics**.

---

## Data & Preprocessing (Summary)
- **Records:** 541  
- **Features:** 44 (clinical / biochemical / lifestyle)  
- **Target:** PCOS (Y/N)

### Preprocessing Highlights
- Missing value imputation  
  - Median for numeric  
  - Mode for categorical  
- Categorical encoding: `OrdinalEncoder`  
- Class balancing: `SMOTENC` (handles mixed numeric + categorical)  
- Numeric scaling: `StandardScaler`  
- Stratified 5-fold cross-validation for all comparisons  
- Final feature split used in pipeline:  
  - **25 numeric**
  - **16 categorical** (after treating low-cardinality numerics as categorical)

---

## Models Evaluated
- **Random Forest (ensemble)**
- **Logistic Regression (linear baseline)**
- **Decision Tree**
- **Gaussian Naive Bayes**
- **Hybrid:** Autoencoder (latent numeric features) -> Random Forest

All results shown are **cross-validated (Stratified 5-fold).**

---

## Cross-Validation — Traditional ML Models (5-fold CV Summary)

| Model               | Accuracy | F1-score | Precision | Recall |
|---------------------|-----------|-----------|------------|---------|
| Random Forest       | 0.9075    | 0.9070    | 0.9071     | 0.9075  |
| Logistic Regression | 0.8891    | 0.8907    | 0.8967     | 0.8891  |
| Decision Tree       | 0.8170    | 0.8177    | 0.8205     | 0.8170  |
| Gaussian NB         | 0.8024    | 0.8056    | 0.8314     | 0.8024  |

> **Note:** Random Forest achieves the highest CV averages among ML-only models (~90.75% accuracy).

---

## Hybrid Model Results - Autoencoder + Random Forest (5-fold CV)

These numbers are from the hybrid pipeline where **numeric features** are encoded using an **Autoencoder** and concatenated with **categorical features** before training Random Forest.

### Per-fold Hybrid Results (Accuracy / F1)
| Fold | Accuracy | F1-score |
|------|-----------|----------|
| 1 | 0.8899 | 0.8899 |
| 2 | 0.8796 | 0.8792 |
| 3 | 0.9259 | 0.9259 |
| 4 | 0.8889 | 0.8870 |
| 5 | 0.8889 | 0.8871 |

### Cross-Validation Averages
- **Average Accuracy:** 0.8946  
- **Average F1-score:** 0.8938  

### Overall Classification (aggregated predictions from CV)
- accuracy 0.89 
- weighted avg precision 0.89, recall 0.89, f1-score 0.89


### Per-class (aggregated)
| Class | Description | Precision | Recall | F1-score | Support |
|--------|--------------|------------|---------|-----------|----------|
| 0 | Non-PCOS | 0.91 | 0.93 | 0.92 | 364 |
| 1 | PCOS | 0.86 | 0.81 | 0.83 | 177 |

---

## Interpretation & Key Takeaways
- **Standalone Random Forest (ML-only)** shows slightly higher CV averages (~90.75% accuracy) than the **Hybrid Autoencoder + RF (~89.46%)**.
- The Hybrid model achieved **~89.5% accuracy** and **~0.894 F1**, which is competitive.
- **Random Forest (ML-only)** gave the best CV-average accuracy on this dataset.
- The Hybrid approach may help with **dimensionality reduction** and **robustness** - benefits likely to show more strongly with **larger or noisier datasets**.
- **SMOTENC** successfully balanced mixed-type data and improved minority-class (PCOS) recall.

---

## Limitations
- Dataset size is modest (**541 samples**) - external validation is recommended.  
- Results depend on **preprocessing choices** (encoding, imputation, SMOTE variant) and **CV split randomness (seed = 42)**.  
- Hybrid benefits might be clearer with more **numeric-only features** or **larger datasets**.

---

## Next Steps / Recommended Experiments
- Compare **Hybrid vs ML-only** on an **external test set** (if available).  
- Evaluate **model interpretability** (e.g., SHAP/LIME), especially for Random Forest and hybrid latent features.  
- Perform **hyperparameter tuning** (RandomizedSearchCV / Optuna) for both RF and AE.  
- Experiment with **different latent sizes** or **denoising/sparse autoencoders**.  
- Consider **nested cross-validation** for unbiased hyperparameter tuning.

---

## Tech Stack
- **Python**, `pandas`, `numpy`  
- **scikit-learn** (models, metrics)  
- **imbalanced-learn** (`SMOTE` / `SMOTENC`)  
- **TensorFlow / Keras** (Autoencoder)  
- **matplotlib**, **seaborn** (plots)  
- **joblib** (model saving)

---
## Conclusion
- This project successfully implemented and rigorously compared traditional ML classifiers against a hybrid Autoencoder-Random Forest pipeline for PCOS detection.
- The key finding is that for this specific (modest-sized) dataset, the standalone **Random Forest** (trained on SMOTENC-resampled and scaled features) achieved the highest cross-validated performance (**~90.75% accuracy**).
- The **Hybrid (Autoencoder + RF) model** performed exceptionally well (**~89.46% accuracy**) and demonstrated a valid, modern approach to feature engineering. However, the overhead and complexity of the deep learning feature extractor did not translate to a performance gain over the classical ML-only pipeline in this instance. This underscores the importance of baselining and demonstrates that a well-processed dataset can often be modeled effectively by robust ensemble methods like Random Forest.
