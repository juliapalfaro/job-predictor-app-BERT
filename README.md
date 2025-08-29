
## 📌 Project Overview  
This project, developed as my **final year dissertation (COMP1682)**, is a **machine learning–based tool** that predicts missing information in job advertisements.  

Many job adverts omit essential details such as **experience level** or **salary range**, which creates uncertainty for job seekers. By leveraging **Natural Language Processing (NLP)** and machine learning techniques (Logistic Regression, Random Forest, and **BERT**), the tool analyses job descriptions and predicts these missing attributes, giving job seekers clearer and more complete information.  

---

## 🎯 Aim & Objectives  

**Aim**  
Develop a predictive ML tool to enrich job adverts by automatically inferring missing information (e.g., experience level, salary range).  

**Objectives**  
- Conduct a literature review of ML methods for job advert analysis  
- Design a system architecture and NLP pipeline for text preprocessing  
- Train and compare Logistic Regression, Random Forest, and BERT models  
- Build a **Flask web app** for user interaction with predictions  
- Evaluate the system through black-box and white-box testing  
- Assess ethical, social, and professional issues (GDPR, bias, transparency)  

---

## 🛠️ Features  
- **NLP pipeline** for text cleaning, tokenization, and feature extraction  
- **Multiple ML models**: Logistic Regression, Random Forest, fine-tuned BERT  
- **Prediction outputs**: job advert’s experience level + salary mapping  
- **Confusion matrix & metrics** for model evaluation  
- **Web app interface**:  
  - Upload/paste a job advert  
  - View predicted attributes & confidence score  
  - Simple HTML/CSS templates for user interaction

## 📂 Repository Structure  

```plaintext
job_predictor_app_BERT/
│── job_predictor.py             # Main app script
│── labeled_bert.py              # Model training with BERT
│── correlation_matrix.py        # Exploratory data analysis
│── labeled_jobs_ads.csv         # Manually labelled dataset
│── requirements.txt             # Project dependencies
│── .gitignore                   # Ignore rules for venv, cache, large files
│
├── static/                      # Frontend assets
│   ├── style.css
│   ├── confusion_matrix.png
│   └── other images...
│
└── templates/                   # HTML templates
    ├── index.html
    ├── predict.html
    ├── about.html
    └── how.html
`````

## 📊 Model Development & Results  

- **Dataset**: 300 manually labelled job adverts across 5 experience levels (Graduate, Junior, Mid, Senior, Lead)  
- **Preprocessing**: TF-IDF for traditional models, Hugging Face tokenizer for BERT  
- **Models compared**:  
  - Logistic Regression (baseline)  
  - Random Forest (non-linear)  
  - BERT (transformer, fine-tuned on adverts)  
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-score  
- **Best result**: BERT achieved the highest macro F1-score (0.66 with 150 samples)  

---

## 📌 Skills Demonstrated  

- **Machine Learning**: Logistic Regression, Random Forest, BERT  
- **NLP**: Tokenization, TF-IDF, Hugging Face Transformers  
- **Data Science**: Exploratory data analysis, feature engineering, model evaluation  
- **Web Development**: Flask, HTML/CSS templates  
- **Software Engineering**: Version control, Agile (DSDM), UML design, MoSCoW prioritisation  
- **Professionalism**: Legal/ethical considerations (GDPR compliance, bias awareness)  

---
