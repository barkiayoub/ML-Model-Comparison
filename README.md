# Heart Disease UCI - Model Comparison

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-orange)

A machine learning project comparing multiple supervised algorithms to predict heart disease using the [Heart Disease UCI dataset](https://www.kaggle.com/ronitf/heart-disease-uci). Includes data preprocessing, exploratory analysis, model implementation, and performance evaluation.

<!-- ## 📋 Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Model Comparison](#model-comparison)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments) -->

## Dataset
The **Heart Disease UCI Dataset** contains 14 attributes related to patient health metrics (e.g., age, cholesterol levels, chest pain type) and a binary target variable indicating the presence of heart disease.  
**Download Instructions**:
1. Create a Kaggle account [here](https://www.kaggle.com/account/login?phase=startSignInTab&returnUrl=%2F).
2. Navigate and Download the `ronitf/heart-disease-uci` dataset 
3. Unzip the file and place `heart.csv` in the `data/` directory.

## Models Used
The following machine learning models were trained and evaluated

1. Logistic Regression - Optimized for linear decision boundaries.
2. Decision Tree Classifier - Optimized for interpretability and capturing complex patterns.
3. Random Forest Classifier - Optimized for reducing overfitting and improving accuracy.
4. K-Nearest Neighbors (KNN) - Optimized for non-linear decision boundaries and handling small datasets.

## Preprocessing Steps
- Handling Missing Values Numerical values were imputed using the median, and categorical values were imputed using the most frequent value.
- Feature Scaling Standardization was applied to numerical features.
- Encoding Categorical Data Label encoding was used for categorical variables.
- Train-Test Split The dataset was split into training and testing sets.

## Results
The models were evaluated using accuracy as the primary metric. The obtained results are

- Logistic Regression 83% accuracy
- Decision Tree Classifier 77% accuracy
- Random Forest Classifier 86% accuracy
- KNN Classifier 85% accuracy

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/barkiayoub/ML-Model-Comparison.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   **`requirements.txt`**:
   ```
   pandas==1.3.3
   numpy==1.21.2
   scikit-learn==1.0
   jupyter==1.0.0
   ```

## License
Distributed under the MIT License. See `LICENSE` for details.

## Acknowledgments
- Dataset: [Heart Disease UCI](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) from UCI, hosted on Kaggle.

