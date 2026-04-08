from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import numpy as np
import os
import cv2

# Load model
model = load_model("deepfake_model.h5")

# Load test data
X_test = []
y_test = []

test_path = "dataset/test"

for label in ["real", "fake"]:
    folder = os.path.join(test_path, label)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))   # same size as training
        img = img / 255.0
        X_test.append(img)
        y_test.append(0 if label == "real" else 1)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Predict
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))