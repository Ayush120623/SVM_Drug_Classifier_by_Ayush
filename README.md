# 🧠 Drug Classification using SVM  
_A Machine Learning Project by Ayush_

---

## 📌 Project Overview

This project aims to predict the most suitable **drug type** for a patient based on their medical profile. Using a Support Vector Machine (SVM) classifier, we classify patients into one of five drug categories using features like age, sex, blood pressure, cholesterol levels, and the sodium-to-potassium ratio.

The model demonstrates strong performance after proper preprocessing and hyperparameter tuning, achieving **93% accuracy** on test data.

---

## 🧬 Problem Statement

Pharmaceutical companies often need to recommend drugs based on patient-specific factors. Automating this prediction process can:
- Reduce human error
- Improve healthcare efficiency
- Personalize treatment plans

---

## 📁 Dataset Information

- **Source:** UCI ML Repository
- **Size:** 200 patient records
- **Features:**
  - `Age`: Numeric
  - `Sex`: Categorical (F/M)
  - `BP`: Blood Pressure (LOW, NORMAL, HIGH)
  - `Cholesterol`: (NORMAL, HIGH)
  - `Na_to_K`: Sodium to Potassium ratio (numeric)
  - `Drug`: Target label (Drug A, B, C, X, Y)

---

## ⚙️ Project Workflow

1. **Data Cleaning & Exploration**
   - Checked for null values (none found)
   - Analyzed distributions and outliers
   - Identified skewness in features

2. **Feature Engineering**
   - Label encoding for categorical variables (`Sex`, `BP`, `Cholesterol`)
   - Correlation analysis:
     - `Na_to_K` negatively correlated with `Drug` (-0.69)
     - `BP` moderately positively correlated (0.42)

3. **Model Building**
   - Trained an SVM (Support Vector Classifier)
   - Hyperparameter tuning using `GridSearchCV`
   - Split data using `train_test_split` (70% train, 30% test)

4. **Model Evaluation**
   - **Accuracy:** 93%
   - **Precision:** 0.93
   - **Recall:** 0.93
   - **F1 Score:** 0.92
   - Evaluated using classification report and confusion matrix

---

## 📊 Results Summary

| Metric     | Value |
|------------|-------|
| Accuracy   | 93%   |
| Precision  | 0.93  |
| Recall     | 0.93  |
| F1 Score   | 0.92  |

✅ The model demonstrates reliable and consistent predictions across all drug classes.  
⚠️ Slight misclassification in underrepresented classes (like `DrugX`) due to class imbalance.

---

## 🔍 Sample Visualizations

- Countplot of target classes
- Feature distribution plots
- Confusion matrix heatmap
- Pairplot for multivariate pattern analysis

---

## 💡 Conclusion

This project showcases how machine learning can assist in clinical decision-making. By using SVM and structured patient data, we created a reliable model to classify drug recommendations. The project demonstrates high accuracy and interpretability, making it a good baseline for further development or integration into a clinical system.

---

## 🚀 Future Improvements

- Apply feature scaling (StandardScaler) to enhance SVM margin performance
- Use ensemble models like Random Forest or XGBoost for comparison
- Handle class imbalance using SMOTE or similar techniques
- Deploy the model using Streamlit or Flask

---

## 🛠️ Tech Stack

- Python 🐍
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- Jupyter Notebook / Google Colab

---

## 📂 Project Structure

📦 svm-drug-classification
├── 📜 Enhanced_svm_drug_project__by_Ayush.ipynb
├── 📜 README.md
├── 📊 visuals/
└── 📁 data/

#👋 Connect

Want to chat, collaborate, or hire me?

📬 Email: ayushsharma.mee@gmail.com

💼 LinkedIn : www.linkedin.com/in/ayush-sharma-a975862b5



