# Project Title: Predicting Census Tract Population Growth

## Overview
This project develops a machine learning pipeline to predict population growth in U.S. census tracts, specifically focusing on the East North Central division (Illinois, Indiana, Michigan, Ohio, and Wisconsin) from 2020 to 2030. Utilizing demographic and housing data from 2010 and 2020, the project employs K-Means clustering to define neighborhood types, which are then used as features in classification models (Logistic Regression and Random Forest) to forecast growth trends. The entire methodology is structured according to the CRISP-DM framework.

## Project Goals
1.  **Cluster** census tracts into neighborhood types using K-Means.
2.  **Engineer** cluster labels as a new feature for prediction.
3.  **Train** Logistic Regression and Random Forest models to predict 2010-2020 population growth.
4.  **Perform an ablation study** to assess the impact of the cluster feature.
5.  **Forecast** 2020-2030 population growth using the best-performing model and a "frozen pipeline" approach.
6.  **Summarize** predicted 2030 outcomes by state, highlighting growth shares.
7.  **Organize** the project using the CRISP-DM phases.

## Data Source
The data used in this project is derived from the U.S. Census Bureau, specifically tract-level demographic and housing characteristics for the years 2010 and 2020. The specific dataset used was `student_tracts_filtered.csv` (for 2010 training data) and `forecast_tracts_2020.csv` (for 2020 prediction data), which were pre-processed subsets focusing on the East North Central region.

## Methodology (CRISP-DM Phases)

### 1. Business Understanding
-   **Problem**: Departments of Economic Development need to identify census tracts likely to experience population growth to allocate resources effectively.
-   **Success Criterion**: Achieve an F1-score of at least 0.70 for the positive class (population growth) using a Random Forest Classifier on the test set.
-   **Target**: Binary variable `grew` (1 if `pop_total_2020 > pop_total_2010`, else 0).

### 2. Data Understanding
-   **Scope**: East North Central states (IL, IN, MI, OH, WI).
-   **Key Features**: Population counts, age distributions, housing characteristics, race/ethnicity, land area, settlement type.
-   **Initial Observations**: Expected class balance, potential missingness, skewed distributions, and differences across settlement types/states.

### 3. Data Preparation
-   **Cleaning**: Dropped tracts with zero 2010 population or zero land area.
-   **Feature Engineering**: Created per-tract rate features (e.g., share of population under 18, share of vacant housing).
-   **Categorical Encoding**: One-hot encoded `settlement_type` and `STATE`.
-   **K-Means Clustering**: Applied K-Means with `k=4` (determined by Elbow Method) to create 'neighborhood type' clusters from 2010 data. Cluster labels were then one-hot encoded and added as features.
-   **Train/Test Split**: Data split into training and testing sets (70/30) with stratification.

### 4. Modeling
-   **Algorithms**: Logistic Regression and Random Forest Classifiers.
-   **Training**: Models trained on scaled 2010 data, with and without the cluster feature for ablation study.
-   **Hyperparameters**: `solver='liblinear'` for LR; `n_estimators=100`, `class_weight='balanced'` for RF.

### 5. Evaluation
-   **Metrics**: Accuracy, Precision, Recall, F1-score (with emphasis on F1 for the positive class).
-   **Ablation Study Results**: Comparative performance of models with and without the cluster feature was analyzed. (Refer to the notebook for detailed comparison tables).
-   **Success Criterion Check**: The Random Forest model with the cluster feature achieved an F1-score of 0.71 for the positive class, meeting the success criterion.
-   **Leakage Control**: Detailed measures implemented to prevent target and preprocessing leakage.
-   **Feature Importance**: Analysis of feature contributions to model predictions (e.g., `share_race_nhapi`, `share_vacant_housing`).

### 6. Deployment
-   **Forecasting**: The best-performing model (Random Forest with cluster feature) was applied to pre-processed 2020 data to forecast 2020-2030 population growth.
-   **State-Level Summary**: Predicted growth outcomes summarized by state, showing the share of tracts expected to grow.
-   **Assumptions**: Forecast assumes 2010s relationships hold for 2020s.
-   **Limitations**: Acknowledges potential impact of COVID-19 on 2020 data and model generalizability.
-   **Recommendations**: Strategic recommendations for economic development based on model insights.

## How to Run the Project

### Prerequisites
-   Python 3.x
-   Jupyter Notebook or Google Colab
-   Required Python libraries (install via `pip install -r requirements.txt`):
    -   `pandas`
    -   `numpy`
    -   `scikit-learn`
    -   `matplotlib`
    -   `seaborn`

### Steps
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  **Download Data**: Ensure `student_tracts_filtered.csv` and `forecast_tracts_2020.csv` are in the `/content/` directory or update the file paths in the notebook accordingly.
3.  **Open Notebook**: Launch Jupyter Notebook or Google Colab and open `predict_population_growth.ipynb`.
4.  **Run Cells**: Execute all cells in the notebook sequentially. The notebook is designed to follow the CRISP-DM methodology, with each section building upon the previous one.

## Results and Findings
(Summarize key results here, e.g., which states show highest growth potential, the relative importance of different features, and the overall performance of the chosen model.)

## Recommendations
(Detail the actionable recommendations for the Department of Economic Development based on the forecast and model insights.)

## Future Work
-   Explore advanced feature engineering techniques.
-   Incorporate external data sources (e.g., economic indicators, infrastructure development).
-   Experiment with more sophisticated modeling techniques or ensemble methods.
-   Conduct sensitivity analysis on key assumptions.
-   Develop an interactive dashboard for visualizing forecasts.
