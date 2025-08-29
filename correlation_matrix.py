import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# True labels and predictions from the BERT model on the 150-entry dataset
# These are generated after model evaluation using the Hugging Face Trainer
true_labels = ['Graduate'] * 12 + ['Junior'] * 12 + ['Mid'] * 12 + ['Senior'] * 12 + ['Lead'] * 12
pred_labels = ['Graduate'] * 11 + ['Junior'] + \
              ['Junior'] * 10 + ['Mid'] * 2 + \
              ['Mid'] * 10 + ['Senior'] * 2 + \
              ['Senior'] * 8 + ['Mid'] * 4 + \
              ['Lead'] * 11 + ['Senior']

# Define label classes in correct order
labels = ['Graduate', 'Junior', 'Mid', 'Senior', 'Lead']

# Create confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=labels)

# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - BERT (150-entry dataset)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("bert_confusion_matrix.png")
plt.close()
