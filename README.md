# 🫀 Cardiovascular Risk Prediction – End-to-End ML Deployment 

This project presents a production-grade machine learning pipeline for predicting cardiovascular disease risk. It encompasses everything from data preprocessing, EDA, model training and evaluation, to CI/CD-powered containerized deployment using Docker, AWS ECR, and EC2, with an optional Streamlit frontend.

---

## 📌 Table of Contents

1. [Problem Statement](#problem-statement)  
2. [Dataset Overview](#dataset-overview)  
3. [Data Preprocessing](#data-preprocessing)  
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
5. [Feature Engineering](#feature-engineering)  
6. [Modeling Strategy](#modeling-strategy)  
7. [Model Evaluation](#model-evaluation)  
8. [Model Selection](#model-selection)  
9. [FastAPI Deployment](#fastapi-deployment)  
10. [Dockerization](#dockerization)  
11. [CI/CD with GitHub Actions](#cicd-with-github-actions)  
12. [AWS ECR + EC2 Deployment](#aws-ecr--ec2-deployment)  
13. [Streamlit Frontend (Optional)](#streamlit-frontend-optional)  
14. [Project Structure](#project-structure)  
15. [Author](#author)

---

## 🧠 Problem Statement

Early detection of cardiovascular disease (CVD) is critical to patient survival. This project aims to develop a predictive model that identifies patients at high risk for CVD based on clinical attributes.

---

## 📊 Dataset Overview

The dataset contains anonymized patient health metrics such as:

| Feature            | Description                                         |
|--------------------|-----------------------------------------------------|
| `age`              | Age of the patient in years                         |
| `education`        | Education level (1–4)                               |
| `sex`              | Gender (`M` or `F`)                                 |
| `is_smoking`       | Smoking status (`YES` or `NO`)                      |
| `cigsPerDay`       | Number of cigarettes smoked per day                 |
| `BPMeds`           | On blood pressure medication (0 or 1)              |
| `prevalentStroke`  | History of stroke                                   |
| `prevalentHyp`     | History of hypertension                             |
| `diabetes`         | Has diabetes (0 or 1)                               |
| `totChol`          | Total cholesterol                                   |
| `sysBP`            | Systolic blood pressure                             |
| `diaBP`            | Diastolic blood pressure                            |
| `BMI`              | Body Mass Index                                     |
| `heartRate`        | Resting heart rate                                  |
| `glucose`          | Blood glucose level                                 |
| `TenYearCHD`       | Target variable – CHD risk within 10 years (0 or 1) |

Total rows: 3,390  
Missing values handled for: `education`, `cigsPerDay`, `BPMeds`, `totChol`, `BMI`, `glucose`

---

## 🧹 Data Preprocessing

- Imputed missing values with median/mode
- Capped or removed extreme outliers using IQR
- Removed multicollinear features based on VIF > 10
- Categorical encoding for `sex`, `is_smoking`
- Applied `StandardScaler` for normalization

---

## 📈 Exploratory Data Analysis (EDA)

- Histograms & boxplots for numeric features
- Violin plots for CHD vs features
- Correlation matrix to detect redundancy
- Stratified count plots for categorical variables

---

## 🧠 Modeling Strategy

| Version | Model Type         | Sampling Method | Baseline Recall | Tuned Recall |
|---------|--------------------|------------------|------------------|--------------|
| v1      | Random Forest       | SMOTE            | 19%            | 62%          |
| v2      | Logistic Regression | SMOTE            | 7%             | **69%** ✅    |

- Used `GridSearchCV` for hyperparameter tuning
- Used best threshold (`best_threshold.txt`) for final decision boundary

---

## 🧪 Evaluation Metrics

- Precision, Recall, F1-Score, AUC
- Confusion matrix analysis
- ROC curve comparison

---

## ✅ Final Model Artifacts

- `modelv2.pkl`: Final logistic regression model
- `scaler.pkl`: StandardScaler instance
- `best_threshold.txt`: Threshold for classification

---

## 🚀 FastAPI Deployment

API built using FastAPI to expose a `/predict` endpoint.

### Sample Input:
```json
{
  "age": 55,
  "sex": 1,
  "cp": 2,
  "trestbps": 135,
  "chol": 250,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.2,
  "slope": 2,
  "ca": 0,
  "thal": 3
}
```

### Response:
```json
{
  "prediction": 1,
  "probability": 0.82,
  "threshold": 0.43
}
```

---

## 🐳 Dockerization

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## ⚙️ CI/CD with GitHub Actions

Workflow (`.github/workflows/deploy.yml`) automates:
- Dependency installation
- Test execution (`pytest`)
- Docker image build
- Push to AWS ECR

Requires GitHub Secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

---

## ☁️ AWS ECR + EC2 Deployment

1. Push image:
```bash
docker tag cardio-api:v2 <account>.dkr.ecr.us-east-1.amazonaws.com/cardio-api1:v2
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/cardio-api1:v2
```

2. On EC2:
```bash
docker pull <...>:v2
docker stop cardio-api || true
docker rm cardio-api || true
docker run -d -p 8000:8000 --name cardio-api <...>:v2
```

---

## 🌐 Streamlit Frontend (Optional)

- A `streamlitapp.py` file serves as a lightweight frontend
- Collects patient inputs and shows prediction output

---

## 🧪 Unit Tests

Run with:
```bash
pytest test.py
```

Tests include:
- Model loading
- Scaler validation
- Predictive output shape and range

---

## 📁 Project Structure

```
cardio-api/
├── app.py
├── modelv2.pkl
|-- modelv1.pkl 
├── scaler.pkl
├── best_threshold.txt
├── requirements.txt
├── Dockerfile
├── test.py
├── streamlitapp.py
└── .github/workflows/deploy.yml
```

---
## Project Flow Diagram
![Project Flow](download.png)

## 👨‍💻 Author

**Kumar Baibhav**  
🎓 MS Data Science, SUNY Buffalo  
🔗 [GitHub](https://github.com/kumarbaibhav6)

---

## 📬 Contact

Raise an issue or connect via GitHub for suggestions or improvements.