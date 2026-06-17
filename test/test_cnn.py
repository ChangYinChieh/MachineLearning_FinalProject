# ======================================================
# Environment: Google Colab / Local
# Project: Hand Gesture Recognition - Testing & Evaluation
# Author: Chang, Yin-Chieh. U11216024
# Description: 
# This script loads the trained model from model/,
# evaluates it on the testing dataset, and saves the evaluation visual plots to test/output/.
# ======================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns 

# ===== 1. Path and Hyperparameter Settings =====
dataset_path = "./dataSet" 
model_path = "../model/Exp3.keras"
IMG_SIZE = (128, 128) 
BATCH_SIZE = 32

# ===== 2. Load Testing Dataset =====
test_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    shuffle=False,  
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = test_ds.class_names
print(f"Detected classes: {class_names}")

# ===== 3. Load Trained Model =====
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}. Please execute the training script in the training section first!")

print(f"\n--- Loading Model from: {model_path} ---")
model = tf.keras.models.load_model(model_path)

# ===== 4. Evaluate Model on Test Set =====
print("\n--- Evaluating on Test Dataset ---")
loss, accuracy = model.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# ===== 5. Predict and Generate Confusion Matrix =====
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true = []
for images, labels in test_ds:
    y_true.extend(np.argmax(labels.numpy(), axis=1))
y_true = np.array(y_true)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ===== 6. Plot and Save Testing Evaluation Results =====
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Exp3 (MiniVGGNet) - Test Confusion Matrix\n(Total Accuracy: {accuracy*100:.1f}%)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "Exp3_result.png")
plt.savefig(plot_path)
print(f"\n[Success] Testing evaluation plot successfully saved to: {plot_path}")
plt.show()