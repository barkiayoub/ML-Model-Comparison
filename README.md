# Heart Disease UCI - Model Comparison

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-orange)

A machine learning project comparing multiple supervised algorithms to predict heart disease using the [Heart Disease UCI dataset](https://www.kaggle.com/ronitf/heart-disease-uci). Includes data preprocessing, exploratory analysis, model implementation, and performance evaluation.

## 📋 Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Model Comparison](#model-comparison)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🗂 Dataset
The **Heart Disease UCI Dataset** contains 14 attributes related to patient health metrics (e.g., age, cholesterol levels, chest pain type) and a binary target variable indicating the presence of heart disease.  
**Download Instructions**:
1. Create a Kaggle account [here](https://www.kaggle.com/account/login?phase=startSignInTab&returnUrl=%2F).
2. Navigate and Download the `ronitf/heart-disease-uci` dataset 
3. Unzip the file and place `heart.csv` in the `data/` directory.

<!-- ## 🛠 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Heart-Disease-UCI-Model-Comparison.git
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
   matplotlib==3.4.3
   seaborn==0.11.2
   jupyter==1.0.0
   kaggle==1.5.12
   ``` -->

<!-- ## 🚀 Usage
1. **Data Preparation**: Clean missing values, encode categorical features, and normalize data.  
   See `notebooks/data_preprocessing.ipynb`.
2. **Exploratory Analysis**: Visualize distributions, correlations, and heatmaps.  
   See `notebooks/exploratory_analysis.ipynb`.
3. **Model Implementation**:  
   - Algorithms: KNN, SVM, Decision Tree, Logistic Regression, Linear Regression.  
   - 5-fold cross-validation for evaluation.  
   - Metrics: Accuracy, F1-Score (classification); MSE, R² (regression).  
   Code in `scripts/train_models.py`.

## 📊 Results
**Best Performing Models**:
- **Classification**: Logistic Regression achieved the highest F1-score (0.85) and accuracy (88%).  
- **Regression**: Linear Regression had the lowest MSE (0.12) but was less interpretable for binary classification.  

Key insights from EDA:
- Strong correlation between `thalach` (max heart rate) and target.
- Age and cholesterol showed moderate predictive power.

## 📈 Model Comparison
| Model              | Accuracy (%) | F1-Score | MSE   | Training Time (s) |
|--------------------|--------------|----------|-------|-------------------|
| Logistic Regression| 88           | 0.85     | -     | 0.8               |
| KNN                | 82           | 0.79     | -     | 1.2               |
| Decision Tree      | 78           | 0.75     | -     | 0.5               |
| SVM                | 84           | 0.81     | -     | 3.1               |
| Linear Regression  | -            | -        | 0.12  | 0.6               |

## 🤝 Contributing
Contributions are welcome! Open an issue or submit a pull request. Ensure code follows PEP8 guidelines. -->

## 📜 License
Distributed under the MIT License. See `LICENSE` for details.

## 🙏 Acknowledgments
- Dataset: [Heart Disease UCI](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) from UCI, hosted on Kaggle.
- Libraries: `pandas`, `scikit-learn`, `matplotlib`.
``` 

---

**Notes**:
1. Replace `yourusername` in the clone URL with your GitHub username.
2. Adjust the `results` and `model comparison` sections based on your actual findings.
3. Add badges or additional sections (e.g., "FAQ") as needed.