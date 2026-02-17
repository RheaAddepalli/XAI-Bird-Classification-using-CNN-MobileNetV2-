import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# LOAD MODEL
# ==============================
model = tf.keras.models.load_model("bird_model.keras")

IMG_SIZE = 224

# 🔴 CHANGE PATH IF NEEDED
TEST_DIR = r"C:\projects\Cnn with XAI deploy\backend\test"

# ==============================
# LOAD TEST DATA
# ==============================
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    shuffle=False,
    label_mode='categorical'   # IMPORTANT
)

class_names = test_ds.class_names

# ==============================
# 🔥 NORMALIZATION FIX (CRITICAL)
# ==============================
test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

# ==============================
# EVALUATE
# ==============================
loss, accuracy = model.evaluate(test_ds)
print("✅ Test Accuracy:", accuracy)

# ==============================
# PREDICTIONS
# ==============================
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# 🔥 FIX: convert one-hot → class index
y_true = np.argmax(y_true, axis=1)

y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# ==============================
# CLASSIFICATION REPORT
# ==============================
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=False,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ==============================
# LATENCY TEST
# ==============================
sample_batch = next(iter(test_ds))[0]

start = time.time()
model.predict(sample_batch)
end = time.time()

latency = (end - start) / len(sample_batch)

print("⚡ Avg Inference Time per Image (seconds):", latency)
