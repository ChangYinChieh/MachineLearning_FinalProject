# ======================================================
# Environment: Google Colab / Local
# Project: Hand Gesture Recognition - MediaPipe Evaluation
# Author: Chang, Yin-Chieh. U11216024
# Description: 
# This script loads the pre-trained MediaPipe (.task) model,
# evaluates it on the local testing dataset, prints the classification report,
# and saves the confusion matrix plot to ./output/.
# ======================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===== 1. Path and Hyperparameter Settings =====
dataset_path = "./dataSet" 
model_path = "../model/gesture_recognizer.task"
class_names = ["1", "2", "none"] 

# ===== 2. Load Trained Model (Initialize MediaPipe) =====
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}. Please ensure the .task file exists!")

print(f"\n--- Loading MediaPipe Model from: {model_path} ---")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
recognizer = vision.GestureRecognizer.create_from_options(options)

# ===== 3. Load Testing Dataset & Run Inference =====
y_true = []
y_pred = []
valid_ext = ('.png', '.jpg', '.jpeg', '.bmp')

print("\n--- Start MediaPipe Batch Testing ---")
for category in class_names:
    folder_path = os.path.join(dataset_path, category)
    if not os.path.exists(folder_path):
        print(f"Skipping folder [{category}]: Path does not exist")
        continue
    
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_ext)]
    print(f"Processing: Category [{category}], {len(files)} images found")

    for img_name in files:
        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)
        if image is None: 
            continue

        # 1. Record the ground truth label
        y_true.append(category)

        try:
            # 2. Image conversion and recognition
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            prediction = recognizer.recognize(mp_image)

            # 3. Extract and map prediction label back to class_names ("1", "2", "none")
            mapped_label = "none"
            if prediction.gestures and len(prediction.gestures) > 0:
                top_gesture = prediction.gestures[0][0].category_name
                
                # 建立轉換映射，確保與類別名稱字串完全一致
                if top_gesture == "gesture_1":
                    mapped_label = "1"
                elif top_gesture == "gesture_2":
                    mapped_label = "2"
                elif top_gesture == "none":
                    mapped_label = "none"
            
            y_pred.append(mapped_label)

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            y_pred.append("none")

# Close the recognizer to release system resources
recognizer.close()

# Convert lists to numpy arrays for evaluation
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ===== 4. Evaluate Model on Test Set =====
accuracy = accuracy_score(y_true, y_pred)
print("\n" + "="*50)
print(f"MediaPipe Test Accuracy: {accuracy * 100:.2f}%")
print("="*50)

# ===== 5. Predict and Generate Confusion Matrix =====
cm = confusion_matrix(y_true, y_pred, labels=class_names)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, labels=class_names))

# ===== 6. Plot and Save Testing Evaluation Results =====
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'MediaPipe Model - Test Confusion Matrix\n(Total Accuracy: {accuracy*100:.1f}%)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "MediaPipe_result.png")
plt.savefig(plot_path)
print(f"\n[Success] Testing evaluation plot successfully saved to: {plot_path}")
plt.show()