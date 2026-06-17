# ======================================================
# Project: Hand Gesture Recognition - MiniVGGNet
# Author: Chang, Yin-Chieh. U11216024
# Description:
# A MiniVGGNet-based architecture trained trained on RAW, un-preprocessed images.
# ======================================================

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# ===== 1. Path and Hyperparameter Settings =====
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

# ===== 4. Build MiniVGGNet Architecture =====
model_vgg = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    data_augmentation,
    layers.Rescaling(1./255),
    
    # Block 1: Conv -> Conv -> Pool -> Dropout
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    # Block 2: Conv -> Conv -> Pool -> Dropout
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    # Block 3: Fully Connected Layer
    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5), 
    
    layers.Dense(3, activation="softmax")
])

# ===== 5. Compile & Train =====
model_vgg.compile(
    optimizer=optimizers.Adam(learning_rate=0.0003), 
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model_vgg.summary()

EPOCHS = 50 
print("\n--- Start Training MiniVGGNet ---")
history_vgg = model_vgg.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ===== 6. Save Plot and Model =====
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_vgg.history['accuracy'], label='Train Acc', color='blue')
plt.plot(history_vgg.history['val_accuracy'], label='Val Acc', color='orange')
plt.title('Exp3 (MiniVGGNet) - Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_vgg.history['loss'], label='Train Loss', color='blue')
plt.plot(history_vgg.history['val_loss'], label='Val Loss', color='orange')
plt.title('Exp3 (MiniVGGNet) - Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "Exp3.png"))
plt.show()

model_dir = "../model"
os.makedirs(model_dir, exist_ok=True)
model_vgg.save(os.path.join(model_dir, "Exp3.keras"))
print("[Success] MiniVGGNet model saved successfully as Exp3.keras!")