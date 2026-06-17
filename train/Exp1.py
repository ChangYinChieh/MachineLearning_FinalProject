# ======================================================
# Project: Hand Gesture Recognition - BASELINE
# Author: Chang, Yin-Chieh. U11216024
# Description:
# Baseline CNN model trained on RAW, un-preprocessed images.
# ======================================================

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# ===== 1. Path Settings =====
dataset_path = "./dataSet" 
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

# ===== 3. Data Augmentation =====
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.15),       
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15)
])

# ===== 4. Build Architecture =====
model_baseline = models.Sequential([
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
model_baseline.compile(
    optimizer=optimizers.Adam(learning_rate=0.0003), 
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

EPOCHS = 50 
history_baseline = model_baseline.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS
)

# ===== 6. Save Plot and Model =====
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_baseline.history['accuracy'], label='Train Acc', color='blue')
plt.plot(history_baseline.history['val_accuracy'], label='Val Acc', color='orange')
plt.title('Exp1 - Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_baseline.history['loss'], label='Train Loss', color='blue')
plt.plot(history_baseline.history['val_loss'], label='Val Loss', color='orange')
plt.title('Exp1 - Loss')
plt.legend()
plt.tight_layout()

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "Exp1.png"))
plt.show()

model_dir = "../model"
os.makedirs(model_dir, exist_ok=True)
model_baseline.save(os.path.join(model_dir, "Exp1.keras"))