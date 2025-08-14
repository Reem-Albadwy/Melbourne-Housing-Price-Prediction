#  Melbourne Housing Price Prediction

Predict Melbourne house prices like a pro using **the art of machine learning**.  
**Final Performance (CatBoost):** R² = 0.838  | MAE ≈ 157k | RMSE ≈ 249k

---

## Project Overview

**Objective:** Accurately predict house sale prices using the Melbourne housing dataset.  

**Why it stands out:**  
1) **Full EDA** exploring distributions, correlations, outliers, and insights via Pandas Profiling.
 
2) **Smart Data Cleaning & Imputation:**  
  - Corrected invalid zeros in `BuildingArea`, `Landsize`, `Bedroom2`, `Bathroom`.  
  - Missing values filled using **median/mode conditioned on related features** (e.g., grouped by `Type`, `Regionname`, `Rooms`).  
  - Numeric outliers capped to maintain realistic ranges.

3) **Advanced Feature Engineering**:
  - `Age` from `YearBuilt`.  
  - `geo_cluster` created with KMeans on latitude & longitude.  
  - One-hot encoding: `Type`, `Method`.  
  - Frequency encoding: `CouncilArea`.  
  - KFold target encoding: `Suburb` & `geo_cluster` (prevents data leakage).
    
4) **Modeling & Evaluation:** Multiple regressors tested: Random Forest, XGBoost, LGBM, Extra Trees, Gradient Boosting, Decision Tree, CatBoost.

5) **Hyperparameter Tuning:** CatBoost optimized via RandomizedSearchCV for robust performance.

---

##  Top Model Results
| Model           | R²     | MAE      | RMSE     |
|-----------------|--------|----------|----------|
| **CatBoost**    | 0.838  | 156,205  | 249,511  |
| Extra Trees     | 0.828  | 159,682  | 256,855  |
| RandomForest    | 0.827  | 160,650  | 257,773  |
| LightGBM        | 0.826  | 162,484  | 258,616  |
| XGBoost         | 0.818  | 163,989  | 264,751  |
| GradientBoosting | 0.794 | 180,118  | 281,690  |
| Decision Tree   | 0.655  | 222,801  | 364,476  |

---

🔗 [Kaggle Profile](https://www.kaggle.com/reemalbadwii)
