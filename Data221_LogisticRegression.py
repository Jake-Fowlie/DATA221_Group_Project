from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import os
import cv2

def load_images_from_folder(folder, label_map, img_size=128):
    X = []
    y = []

    for label_name, label in label_map.items():
        path = os.path.join(folder, label_name)

        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(label)

    return np.array(X), np.array(y)

label_map = {
    "glioma": 0,
    "meningioma": 1,
    "pituitary": 2,
    "notumor": 3
}

X_train, y_train = load_images_from_folder("Data/Training", label_map)
X_test, y_test = load_images_from_folder("Data/Testing", label_map)

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, solver="lbfgs")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="macro"))
print("Recall:", recall_score(y_test, y_pred, average="macro"))
print("F1 Score:", f1_score(y_test, y_pred, average="macro"))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)