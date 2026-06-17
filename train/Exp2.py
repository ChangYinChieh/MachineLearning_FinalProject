# ======================================================
# Project: Hand Gesture Recognition
# Author: Chang, Yin-Chieh. U11216024
# Description:
# A pure CNN-based hand gesture recognition model trained
# using edge-feature-extracted image datasets.
# ======================================================

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# ===== 1. Path and Hyperparameter Settings =====
dataset_path = "./dataSet_preprocessed" 
BATCH_SIZE = 32
IMG_SIZE = (128, 128) 

# ===== 2. Load and Split Dataset =====
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42, 
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42, 
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

# ===== 3. Data Augmentation(Preserving binary edges) =====
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.15),       
    layers.RandomZoom(0.15)          
])

# ===== 4. Build Final Optimized CNN Architecture =====
model_final = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    data_augmentation,
    layers.Rescaling(1./255),
    
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5), 
    
    layers.Dense(3, activation="softmax")
])

# ===== 5. Compile & Train =====
model_final.compile(
    optimizer=optimizers.Adam(learning_rate=0.0003), 
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model_final.summary()

EPOCHS = 50 
print("\n--- Start Training ---")
history_final = model_final.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ===== 6. Save Plot and Model =====
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_final.history['accuracy'], label='Train Acc', color='blue')
plt.plot(history_final.history['val_accuracy'], label='Val Acc', color='orange')
plt.title('Exp2 - Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_final.history['loss'], label='Train Loss', color='blue')
plt.plot(history_final.history['val_loss'], label='Val Loss', color='orange')
plt.title('Exp2 - Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "Exp2.png"))
plt.show()

# Ensure model directory exists before saving
model_dir = "../model"
os.makedirs(model_dir, exist_ok=True)
model_final.save(os.path.join(model_dir, "Exp2.keras"))
print("[Success] Final optimized model saved successfully!")