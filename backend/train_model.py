import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import json

# ==============================
# CONFIG (OPTIMIZED)
# ==============================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30   # keep small for speed + size

train_dir = r"C:\projects\Cnn with XAI deploy\backend\train"


# ==============================
# DATA GENERATOR
# ==============================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_data.class_indices)

print("✅ Classes:", train_data.class_indices)

# ==============================
# SAVE CLASS NAMES (IMPORTANT)
# ==============================
with open("class_names.json", "w") as f:
    json.dump(train_data.class_indices, f)

# ==============================
# MODEL (LIGHTWEIGHT)
# # ==============================
# base_model = MobileNetV2(
#     weights='imagenet',
#     include_top=False,
#     input_shape=(IMG_SIZE, IMG_SIZE, 3)
# )

# # 🔥 Freeze base (IMPORTANT for small size)
# # base_model.trainable = False

# x = base_model.output
# x = layers.GlobalAveragePooling2D()(x)

# # 🔥 Small dense layer (keeps model small)
# x = layers.Dense(64, activation='relu')(x)

# # Output layer
# output = layers.Dense(num_classes, activation='softmax')(x)

# model = models.Model(inputs=base_model.input, outputs=output)

# # ==============================
# # COMPILE
# # ==============================
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.summary()

# ==============================
# MODEL (FINE-TUNED)
# ==============================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# 🔥 Enable training
base_model.trainable = True

# 🔥 Freeze most layers (IMPORTANT)
for layer in base_model.layers[:-30]:
    layer.trainable = False

# ==============================
# CUSTOM HEAD
# ==============================
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)

# 🔥 Dense + Dropout (prevents overfitting)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)

output = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# ==============================
# COMPILE (LOW LR IMPORTANT)
# ==============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),  # 🔥 VERY IMPORTANT
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ==============================
# TRAIN
# ==============================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ==============================
# SAVE MODEL (SMALL FORMAT)
# ==============================
model.save("bird_model.keras")

print("✅ Model saved as bird_model.keras")
