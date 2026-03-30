import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# FUNCTION: Load and preprocess images
def load_images_from_folder(folder_path, label_map, image_size=128):
    """
    Loads images from folder structure and converts them into feature vectors.
    Steps:
    - Read each image in grayscale
    - Resize to fixed size (128x128)
    - Normalize pixel values (0–255 → 0–1)
    - Flatten image into 1D vector (required for classical ML models)
    Returns:
    - X: numpy array of features (images)
    - y: numpy array of labels
    """
    features = []
    labels = []

    # Loop through each class (glioma, meningioma, etc.)
    for class_name, class_label in label_map.items():
        class_folder = os.path.join(folder_path, class_name)

        # Loop through images inside each class folder
        for file_name in os.listdir(class_folder):
            image_path = os.path.join(class_folder, file_name)

            # Read image in grayscale (reduces complexity)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                # Resize image to 128x128
                image = cv2.resize(image, (image_size, image_size))

                # Normalize pixel values (0–255 → 0–1)
                image = image / 255.0

                # Flatten image into 1D vector (KNN requires this)
                image = image.flatten()

                features.append(image)
                labels.append(class_label)

    return np.array(features), np.array(labels)


# MAIN FUNCTION
def main():

    # Mapping class names to numerical labels
    label_map = {
        "glioma": 0,
        "meningioma": 1,
        "pituitary": 2,
        "notumor": 3
    }

    # Used for readable output in classification report
    class_names = ["glioma", "meningioma", "pituitary", "notumor"]

    # Get base directory (avoids path errors across machines)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define dataset paths (predefined train-test split)
    train_path = os.path.join(base_dir, "Data", "Training")
    test_path = os.path.join(base_dir, "Data", "Testing")

    # LOAD DATA
    print("Loading training data...")
    X_train, y_train = load_images_from_folder(train_path, label_map)

    print("Loading testing data...")
    X_test, y_test = load_images_from_folder(test_path, label_map)

    # Check shapes
    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    # Standardize features so all dimensions contribute equally
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)  # fit only on training data
    X_test = scaler.transform(X_test)        # apply same scaling to test data


    # KNN MODEL
    # k = number of neighbors used for classification
    knn_model = KNeighborsClassifier(n_neighbors=5)

    # Train model
    knn_model.fit(X_train, y_train)

    # Predict on test set
    y_pred = knn_model.predict(X_test)


    # EVALUATION METRICS
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)


    # OUTPUT RESULTS
    print("\n=== KNN Results ===")
    print("Accuracy:", accuracy)
    print("Macro Precision:", precision)
    print("Macro Recall:", recall)
    print("Macro F1 Score:", f1)

    print("\nConfusion Matrix:")
    print(conf_matrix)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))




# RUN PROGRAM
if __name__ == "__main__":
    main()

