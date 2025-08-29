# job_predictor.py — Predict salary using experience-based BERT model

import re
from flask import Flask, render_template, request
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import joblib
import torch.nn.functional as F

app = Flask(__name__)

# Load tokenizer and experience model
tokenizer = BertTokenizerFast.from_pretrained("./bert_experience_gold")
exp_model = BertForSequenceClassification.from_pretrained("./bert_experience_gold")
exp_encoder = joblib.load("bert_gold_label_encoder.pkl")
exp_model.eval()

# Predict experience label using BERT model
def predict_label(text, model, encoder):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
        predicted_class_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class_id].item()  # Confidence score
        label = encoder.inverse_transform([predicted_class_id])[0]
        return label, confidence
    
# Home route (can be landing page or info page)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")

def about():
    return render_template("about.html")

@app.route("/how")

def how():
    return render_template("how.html")

# Prediction route (Make a Prediction page)
@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    error_message = None
    accuracy = 0.67
    f1_score = 0.66

    if request.method == "POST":
        job_text = request.form["job_text"].strip()

        # Basic input validation
        if len(job_text) < 10 or not re.search(r"[a-zA-Z]", job_text):
            error_message = "Please enter a meaningful job advertisement."
        else:
            experience, confidence = predict_label(job_text, exp_model, exp_encoder)
            experience_to_salary = {
                "Graduate": "20K-40K",
                "Junior": "40K-60K",
                "Mid": "60K-80K",
                "Senior": "80K-100K",
                "Lead": "100K+",
            }

            salary = experience_to_salary.get(experience, "Unknown")
            
            result = {"experience": experience, 
                      "salary": salary,
                      "confidence": round (confidence * 100, 2)}

    return render_template("predict.html", result=result, error=error_message, accuracy=accuracy, f1_score=f1_score)

if __name__ == "__main__":
    app.run(debug=True)
