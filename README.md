# Credit Card Fraud Detection Using Machine Learning

This project focuses on identifying fraudulent credit card transactions using machine learning techniques. Given the significant imbalance in the dataset, various preprocessing, sampling, and modeling strategies were implemented to improve detection accuracy, especially for minority (fraudulent) classes.

---

## 📂 Project Structure

- `2401262.ipynb`: Jupyter Notebook containing all data processing, modeling, and evaluation steps.
- `2401262.pdf`: Report with detailed explanation and results.
- `README.md`: Overview of the project.

---

## 📊 Dataset

The dataset used in this project was sourced from [Hugging Face](https://huggingface.co/datasets/dazzle-nu/CIS435-CreditCardFraudDetection). It contains anonymized credit card transactions labeled as fraudulent or legitimate.

- **Total records:** 1,048,575 transactions  
- **Fraudulent records:** 1,825  
- **Legitimate records:** 1,046,750

Each transaction includes:
- Anonymized features (V1–V28)
- `Time` and `Amount`
- `Class`: the target label (0 = legitimate, 1 = fraud)

---

## ⚙️ Workflow

### 1. **Data Exploration (EDA)**
- Analyzed class distribution and imbalance
- Visualized feature distributions using histograms and boxplots

### 2. **Preprocessing**
- Normalized `Amount` and `Time`
- Removed original `Time` column after transformation
- Split data into training and testing sets

### 3. **Handling Imbalance**
- Applied **Under-sampling** to balance the training dataset

### 4. **Modeling**
Implemented and evaluated multiple machine learning models:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### 5. **Evaluation Metrics**
- Confusion Matrix
- Precision, Recall, F1-score
- ROC-AUC
- Classification Report

---

## 📈 Results

| Model               | Precision | Recall | F1-Score  | ROC-AUC   |
| ------------------- | --------- | ------ | --------- | --------- |
| Logistic Regression | High      | High   | Good      | Good      |
| Decision Tree       | Very High | High   | Excellent | Excellent |
| Random Forest       | Very High | High   | Excellent | Excellent |
| XGBoost             | Very High | High   | Excellent | Excellent |

- Tree-based models (especially XGBoost) showed superior performance in detecting fraud.
- Focus was placed on optimizing **recall** to minimize false negatives.

---

## ✅ Requirements

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - xgboost

Install dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## 📌 Conclusion

By addressing data imbalance and using powerful models like XGBoost, this project successfully improved fraud detection performance. The findings highlight the importance of precision, recall, and model selection in critical applications like financial fraud prevention.
