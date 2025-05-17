import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def extract_lbp_features(image):
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def load_images_from_folder(folder_path, label):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder_path, filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            features = extract_lbp_features(image)
            data.append((features, label))
    return data

# Load dataset
first = load_images_from_folder('data/first_print', 0)
second = load_images_from_folder('data/second_print', 1)
dataset = first + second

X = np.array([f for f, _ in dataset])
y = np.array([l for _, l in dataset])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


import joblib

# After model.fit(...)
joblib.dump(model, 'traditional_model.pkl')
print("SVM model saved as traditional_model.pkl")
