import pandas as pd
import joblib
import numpy as np
import warnings
import random
import os
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

warnings.filterwarnings("ignore")

#Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#Load dataset
df = pd.read_csv("labeled_jobs_ads.csv", encoding="ISO-8859-1")
df.columns = [col.strip() for col in df.columns]
df = df.rename(columns={"Job Advertisement": "text"})

#Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["Experience_Level"])
joblib.dump(le, "bert_gold_label_encoder.pkl")

#Stratified split
train_df, test_df = train_test_split(
    df[["text", "label"]],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

#Convert to Hugging Face Dataset
dataset = {
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "test": Dataset.from_pandas(test_df.reset_index(drop=True))
}

#Tokenization
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

dataset["train"] = dataset["train"].map(tokenize, batched=True)
dataset["test"] = dataset["test"].map(tokenize, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(le.classes_)
)

#Training setup
training_args = TrainingArguments(
    output_dir="bert_experience_gold",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="logs",
    logging_strategy="epoch",
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

#Train
trainer.train()

#Evaluate
preds = np.argmax(trainer.predict(dataset["test"]).predictions, axis=1)
labels = dataset["test"]["label"]
print("Experience Level Classification Report:")
print(classification_report(labels, preds, target_names=le.classes_))

#Save model and tokenizer
model.save_pretrained("bert_experience_gold")
tokenizer.save_pretrained("bert_experience_gold")
